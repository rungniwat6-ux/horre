import os
import sys
import datetime
import zipfile

# Force ONNX Runtime and cuDNN to disable the buggy cuDNN Frontend graph engine
# This completely prevents CUDNN_BACKEND_API_FAILED crashes on long video runs on Ada/Hopper architectures.
os.environ["ORT_CUDA_FLAGS"] = "0"
os.environ["CUDNN_FRONTEND_DISABLE_CONV_ENGINE_PLAN"] = "1"
os.environ["CUDNN_FRONTEND_DISABLE_GRAPH"] = "1"

# Programmatically detect and inject pip-installed CUDA/cuDNN libraries into LD_LIBRARY_PATH
# This fixes dynamic library loading for onnxruntime-gpu on Linux cloud servers
try:
    import site
    site_packages_dirs = site.getsitepackages()
except Exception:
    site_packages_dirs = []

for path in sys.path:
    if 'site-packages' in path and path not in site_packages_dirs:
        site_packages_dirs.append(path)

nvidia_libs = []
nvidia_bins = []
for sp in site_packages_dirs:
    nvidia_dir = os.path.join(sp, 'nvidia')
    if os.path.exists(nvidia_dir):
        for root, dirs, files in os.walk(nvidia_dir):
            if 'lib' in dirs:
                nvidia_libs.append(os.path.join(root, 'lib'))
            if 'bin' in dirs:
                nvidia_bins.append(os.path.join(root, 'bin'))

# 1. Windows DLL search path injection (Python 3.8+ requirement)
if sys.platform.startswith('win') and nvidia_bins:
    for bin_path in nvidia_bins:
        abs_bin = os.path.abspath(bin_path)
        try:
            os.add_dll_directory(abs_bin)
            os.environ['PATH'] = abs_bin + os.pathsep + os.environ.get('PATH', '')
            print(f"[Info] Programmatically loaded Windows CUDA DLL path: {abs_bin}")
        except Exception:
            pass

# 2. Linux LD_LIBRARY_PATH environment injection and self-restart
if not sys.platform.startswith('win') and nvidia_libs:
    cuda_lib_paths = ":".join(nvidia_libs)
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    
    needs_restart = False
    for lib_path in nvidia_libs:
        if lib_path not in current_ld:
            needs_restart = True
            break
            
    if needs_restart:
        if current_ld:
            os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_paths}:{current_ld}"
        else:
            os.environ['LD_LIBRARY_PATH'] = cuda_lib_paths
        print(f"[Info] Dynamically injected CUDA search paths to environment.")
        print(f"[Info] Restarting Python process to apply LD_LIBRARY_PATH dynamically for GPU acceleration...")
        try:
            os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
        except Exception as e:
            print(f"[Warning] Restart failed: {e}. Please run: export LD_LIBRARY_PATH={cuda_lib_paths}:$LD_LIBRARY_PATH before running python app.py")
    else:
        print(f"[Info] CUDA search paths are already loaded: {current_ld[:120]}...")

import gradio_client.utils
# Monkey patch gradio_client to fix "TypeError: argument of type 'bool' is not iterable" with pydantic v2
orig_get_type = gradio_client.utils.get_type
def patched_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return orig_get_type(schema)
gradio_client.utils.get_type = patched_get_type

import gradio as gr
import cv2
import numpy as np
import onnxruntime as ort
import numpy as np

orig_init = ort.InferenceSession.__init__
def patched_init(self, model_path, sess_options=None, *args, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
        
    # 1. Optimize thread options dynamically based on execution provider to avoid CPU thread contention
    providers = kwargs.get('providers', None)
    args_list = list(args)
    
    # Extract providers parameter positional or keyword
    is_keyword = 'providers' in kwargs
    if providers is None and len(args_list) >= 1:
        providers = args_list[0]
        
    has_gpu = False
    new_providers = []
    if providers:
        for p in providers:
            # Check if provider is string or tuple
            p_name = p[0] if isinstance(p, tuple) else p
            if isinstance(p_name, str) and ('cuda' in p_name.lower() or 'tensorrt' in p_name.lower()):
                has_gpu = True
                
                # Dynamic GPU-only tuning:
                # 1. Route lightweight detector/landmarks to DEFAULT to completely bypass the cuDNN frontend graph planner crash.
                # 2. Keep heavy models (w600k recognition, inswapper, gfpgan, occluder) on HEURISTIC for full GPU speed.
                # Set HEURISTIC search for all models. Safe from crashes due to disabled graph flags.
                algo_search = 'HEURISTIC'
                print(f"[ONNX Init] Path: {model_path} -> Selected Algo Search: {algo_search}")
                
                cuda_opts = {
                    'cudnn_conv_algo_search': algo_search
                }
                
                # Merge with any existing options
                if isinstance(p, tuple) and len(p) >= 2:
                    existing_opts = dict(p[1])
                    existing_opts.update(cuda_opts)
                    new_providers.append(('CUDAExecutionProvider', existing_opts))
                else:
                    new_providers.append(('CUDAExecutionProvider', cuda_opts))
            else:
                new_providers.append(p)
                
        # Re-inject the updated providers list
        if is_keyword:
            kwargs['providers'] = new_providers
        elif len(args_list) >= 1:
            args_list[0] = new_providers
            args = tuple(args_list)
            
    if has_gpu:
        # GPU handles the heavy math, so we only need 1 CPU thread per session to feed it.
        # This completely avoids thread contention / context switching bottlenecks on 96-core systems.
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
    else:
        # CPU fallback uses all logical cores dynamically
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0
    
    # 2. Enable all graph & memory optimizations
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    
    # Run original initialization with unmodified providers/options for safety and reliability
    orig_init(self, model_path, sess_options, *args, **kwargs)
ort.InferenceSession.__init__ = patched_init

import tempfile
from httfacelond.utils.downloader import check_and_download_models

_HEAD3D_RENDER_CACHE = {}

# Ensure models are downloaded before importing SwapEngine
print("[Info] Checking models...")
check_and_download_models()

# Import SwapEngine
from httfacelond.core.swap_engine import SwapEngine

# Initialize SwapEngine
engine = SwapEngine()

hardware_info = engine.get_hardware_status()

# Define swapper and restorer options directly (with fallback checks to keep it robust)
available_swappers = ["inswapper_128_fp16.onnx", "inswapper_128.onnx"]
default_swapper = "inswapper_128_fp16.onnx"

available_restorers = ["None", "GFPGANv1.4.onnx", "codeformer.onnx", "GPEN-BFR-512.onnx", "GPEN-BFR-1024.onnx"]
default_restorer = "GFPGANv1.4.onnx"

available_gpus = engine.get_available_gpus()
print(f"[Info] Available GPUs detected: {available_gpus}")

def load_video_info(video_path, device_mode):
    if not video_path:
        return gr.update(maximum=0, value=0, visible=True), "No video uploaded.", "### Estimated Processing Time: No video uploaded."
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return gr.update(maximum=0, value=0, visible=True), "Error loading video file.", "### Estimated Processing Time: Error reading video."
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Calculate time estimate
    estimate_text = calculate_estimated_time(video_path, device_mode)
    
    return gr.update(maximum=max(0, total_frames - 1), value=0, visible=True), f"Video loaded: {total_frames} frames.", estimate_text

def calculate_estimated_time(video_path, device_mode):
    if not video_path:
        return "### Estimated Processing Time: No video uploaded."
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "### Estimated Processing Time: Error reading video."
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Heuristic speeds (seconds per frame) based on device select
    is_gpu = hardware_info["is_gpu"] and device_mode == "GPU Mode"
    speed_factor = 0.07 if is_gpu else 0.45
    
    total_seconds = total_frames * speed_factor
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    fps_estimate = 1.0 / speed_factor
    device_name = "GPU (Accelerated)" if is_gpu else "CPU (Standard)"
    
    return f"⏱️ **Estimated Swap Duration ({device_name}):** ~{minutes} min {seconds} sec (at ~{fps_estimate:.1f} frames/sec for {total_frames} frames)"

def preview_selected_frame(source_img_path, video_path, frame_index, enhance, enhance_strength, match_color, match_scale, custom_scale, det_thresh, face_upscale_resolution, handle_occlusions, swapper_model, restorer_model, device_mode, selected_gpus, swap_blend_strength=1.0, match_face_shape=True, target_detector="SCRFD (Default)", face_mask_type="InsightFace 106-Point"):
    if not source_img_path or not video_path:
        return None, "[System] Upload both Source Image and Target Video to preview."
        
    active_engine = engine
        
    # Determine the execution provider based on device_mode and selected_gpus
    if device_mode == "GPU Mode" and selected_gpus:
        execution_device = selected_gpus[0] if isinstance(selected_gpus, (list, tuple)) else selected_gpus
    else:
        execution_device = "CPU Only"
        
    # Set execution provider dynamically
    active_engine.set_execution_provider(execution_device)
    
    # Load selected models dynamically
    active_engine.load_swapper(swapper_model)
    active_engine.load_restorer(restorer_model)
    
    # Extract frame
    frame_bgr = active_engine.extract_frame(video_path, frame_index)
    if frame_bgr is None:
        return None, f"[Error] Failed to extract frame {frame_index}."
        
    # Read source image
    source_img = cv2.imread(source_img_path)
    if source_img is None:
        return None, "[Error] Failed to read source image."
        
    logs = f"[Preview] Swapping face on frame index: {frame_index}...\n"
    def log_callback(msg):
        nonlocal logs
        logs += f"{msg}\n"
        
    try:
        result_cv = active_engine.face_swap(
            source_img, 
            frame_bgr, 
            enhance=enhance, 
            enhance_strength=enhance_strength, 
            match_color=match_color,
            match_scale=match_scale,
            custom_scale=custom_scale,
            det_thresh=det_thresh,
            face_upscale_resolution=face_upscale_resolution,
            handle_occlusions=handle_occlusions,
            log_callback=log_callback,
            swap_blend_strength=swap_blend_strength,
            match_face_shape=match_face_shape,
            target_detector=target_detector,
            face_mask_type=face_mask_type
        )
        result_rgb = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)
        logs += "[System] Preview generated successfully!\n"
        return result_rgb, logs
    except Exception as e:
        import traceback
        return None, f"[Error] Preview execution failed: {str(e)}\n{traceback.format_exc()}"

def clear_vram_callback():
    msg = engine.unload_models()
    return msg

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def _uploaded_file_path(uploaded_file):
    if isinstance(uploaded_file, os.PathLike):
        return os.fspath(uploaded_file)
    if isinstance(uploaded_file, str):
        return uploaded_file
    if isinstance(uploaded_file, (list, tuple)) and uploaded_file:
        return _uploaded_file_path(uploaded_file[0])
    if hasattr(uploaded_file, "path"):
        return uploaded_file.path
    if hasattr(uploaded_file, "name"):
        return uploaded_file.name
    if isinstance(uploaded_file, dict):
        return uploaded_file.get("path") or uploaded_file.get("name")
    return None

def _read_image_bgr(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return cv2.imread(path)

def _write_image(path, image):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(path)
    return True

def _collect_image_uploads(uploaded_files, folder_path=""):
    paths = []

    folder_path = folder_path.strip() if folder_path else ""
    if folder_path:
        if os.path.isdir(folder_path):
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    path = os.path.join(root, filename)
                    ext = os.path.splitext(path)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        paths.append(path)
        elif os.path.isfile(folder_path):
            ext = os.path.splitext(folder_path)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                paths.append(folder_path)

    if not uploaded_files:
        return sorted(set(paths), key=lambda p: os.path.basename(p).lower())

    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]
    for uploaded_file in uploaded_files:
        path = _uploaded_file_path(uploaded_file)
        if not path:
            continue
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    nested_path = os.path.join(root, filename)
                    ext = os.path.splitext(nested_path)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        paths.append(nested_path)
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            paths.append(path)
    return sorted(set(paths), key=lambda p: os.path.basename(p).lower())

def _batch_output_dir(save_path):
    dest = save_path.strip() if save_path and save_path.strip() else ""
    if dest:
        root, ext = os.path.splitext(dest)
        if ext.lower() in IMAGE_EXTENSIONS or ext.lower() == ".zip":
            parent = os.path.dirname(dest) or "output"
            folder = os.path.basename(root) or "batch_images"
            return os.path.join(parent, f"{folder}_batch")
        return dest
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("output", f"batch_images_{timestamp}")

def _result_filename(input_path, index):
    base = os.path.splitext(os.path.basename(input_path))[0].strip() or f"image_{index:04d}"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in base)
    return f"{index:04d}_{safe}_swapped.jpg"

def perform_image_swap(source_img, target_img, enhance, enhance_strength, match_color, match_scale, custom_scale, det_thresh, face_upscale_resolution, handle_occlusions, swapper_model, restorer_model, device_mode, selected_gpu, save_path="", swap_blend_strength=1.0, match_face_shape=True, target_detector="SCRFD (Default)", face_mask_type="InsightFace 106-Point"):
    logs = "[System] Initializing image face swap pipeline...\n"
    yield None, logs
    
    if source_img is None or target_img is None:
        logs += "[Error] Missing source or target image.\n"
        yield None, logs
        return
        
    active_engine = engine
        
    def log_callback(msg):
        nonlocal logs
        logs += f"{msg}\n"
        
    # Determine the execution provider based on device_mode and selected_gpu
    if device_mode == "GPU Mode" and selected_gpu:
        execution_device = selected_gpu
    else:
        execution_device = "CPU Only"
        
    # Set execution provider dynamically
    logs += f"[Hardware] Selecting processing device: {execution_device}...\n"
    yield None, logs
    active_engine.set_execution_provider(execution_device)
    
    # Load selected models dynamically
    logs += f"[Model] Selected Swapper: {swapper_model}\n"
    logs += f"[Model] Selected Restorer: {restorer_model}\n"
    yield None, logs
    active_engine.load_swapper(swapper_model)
    active_engine.load_restorer(restorer_model)
    
    # Convert from PIL to BGR OpenCV format
    logs += "[Process] Preprocessing image dimensions and formats...\n"
    yield None, logs
    src_cv = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
    
    logs += "[Process] Executing core face swap algorithm...\n"
    yield None, logs
    
    # Run swapper
    try:
        result_cv = active_engine.face_swap(
            src_cv, 
            tgt_cv, 
            enhance=enhance, 
            enhance_strength=enhance_strength, 
            match_color=match_color,
            match_scale=match_scale,
            custom_scale=custom_scale,
            det_thresh=det_thresh,
            face_upscale_resolution=face_upscale_resolution,
            handle_occlusions=handle_occlusions,
            log_callback=log_callback,
            swap_blend_strength=swap_blend_strength,
            match_face_shape=match_face_shape,
            target_detector=target_detector,
            face_mask_type=face_mask_type
        )
        
        # Convert back to RGB for Gradio
        result_rgb = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)
        
        # Save output image (defaults to output/ directory if path is empty)
        dest = save_path.strip() if (save_path and save_path.strip()) else ""
        if not dest:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("output", exist_ok=True)
            dest = os.path.join("output", f"swapped_image_{timestamp}.jpeg")
            
        try:
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            cv2.imwrite(dest, result_cv)
            logs += f"[System] Output image successfully saved to: {os.path.abspath(dest)}\n"
        except Exception as e:
            logs += f"[Warning] Could not save output image: {str(e)}\n"
                
        logs += "[System] Swap completed successfully!\n"
        yield result_rgb, logs
    except Exception as e:
        import traceback
        traceback.print_exc()
        logs += f"[Error] Execution failed: {str(e)}\n"
        yield None, logs

def perform_batch_image_swap(source_img, target_files, target_folder_path, enhance, enhance_strength, match_color, match_scale, custom_scale, det_thresh, face_upscale_resolution, handle_occlusions, swapper_model, restorer_model, device_mode, selected_gpu, save_path="", swap_blend_strength=1.0, match_face_shape=True, target_detector="SCRFD (Default)", face_mask_type="InsightFace 106-Point"):
    logs = "[System] Initializing batch image face swap pipeline...\n"
    yield [], None, logs

    target_paths = _collect_image_uploads(target_files, target_folder_path)
    if source_img is None or not target_paths:
        logs += "[Error] Missing source image or target image folder/files. Upload a folder, select images, or paste a local folder path.\n"
        yield [], None, logs
        return

    active_engine = engine

    if device_mode == "GPU Mode" and selected_gpu:
        execution_device = selected_gpu
    else:
        execution_device = "CPU Only"

    logs += f"[Hardware] Selecting processing device: {execution_device}...\n"
    logs += f"[Batch] Found {len(target_paths)} image(s) to process.\n"
    yield [], None, logs
    active_engine.set_execution_provider(execution_device)

    logs += f"[Model] Selected Swapper: {swapper_model}\n"
    logs += f"[Model] Selected Restorer: {restorer_model}\n"
    yield [], None, logs
    active_engine.load_swapper(swapper_model)
    active_engine.load_restorer(restorer_model)

    src_cv = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
    output_dir = _batch_output_dir(save_path)
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    failed = []

    for index, target_path in enumerate(target_paths, start=1):
        logs += f"[Batch] ({index}/{len(target_paths)}) Processing: {os.path.basename(target_path)}\n"
        yield saved_paths, None, logs

        tgt_cv = _read_image_bgr(target_path)
        if tgt_cv is None:
            failed.append((target_path, "Could not read image"))
            logs += f"[Warning] Skipped unreadable image: {target_path}\n"
            continue

        item_logs = []
        def log_callback(msg):
            item_logs.append(msg)

        try:
            result_cv = active_engine.face_swap(
                src_cv,
                tgt_cv,
                enhance=enhance,
                enhance_strength=enhance_strength,
                match_color=match_color,
                match_scale=match_scale,
                custom_scale=custom_scale,
                det_thresh=det_thresh,
                face_upscale_resolution=face_upscale_resolution,
                handle_occlusions=handle_occlusions,
                log_callback=log_callback,
                swap_blend_strength=swap_blend_strength,
                match_face_shape=match_face_shape,
                target_detector=target_detector,
                face_mask_type=face_mask_type
            )

            dest = os.path.join(output_dir, _result_filename(target_path, index))
            if _write_image(dest, result_cv):
                saved_paths.append(os.path.abspath(dest))
                logs += f"[Batch] Saved: {os.path.abspath(dest)}\n"
            else:
                failed.append((target_path, "Could not encode output image"))
                logs += f"[Warning] Could not save output for: {target_path}\n"
        except Exception as e:
            failed.append((target_path, str(e)))
            logs += f"[Error] Failed: {os.path.basename(target_path)} -> {str(e)}\n"

    zip_path = None
    if saved_paths:
        zip_path = os.path.abspath(os.path.join(output_dir, "swapped_batch_results.zip"))
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for path in saved_paths:
                zip_file.write(path, arcname=os.path.basename(path))
        logs += f"[System] Batch output folder: {os.path.abspath(output_dir)}\n"
        logs += f"[System] Zip package created: {zip_path}\n"

    logs += f"[System] Batch completed. Success: {len(saved_paths)}, Failed: {len(failed)}.\n"
    if failed:
        logs += "[System] Failed files:\n"
        for path, reason in failed[:20]:
            logs += f"  - {os.path.basename(path)}: {reason}\n"
        if len(failed) > 20:
            logs += f"  - ...and {len(failed) - 20} more.\n"

    yield saved_paths, zip_path, logs

def perform_video_swap(source_img, target_video, enhance, enhance_strength, match_color, match_scale, custom_scale, det_thresh, face_upscale_resolution, handle_occlusions, swapper_model, restorer_model, device_mode, selected_gpus, frame_step, det_size_val, batch_size, save_path="", swap_blend_strength=1.0, match_face_shape=True, progress=gr.Progress(track_tqdm=False), target_detector="SCRFD (Default)", face_mask_type="InsightFace 106-Point"):
    import time
    start_swap_time = time.time()
    logs = "[System] Initializing video face swap pipeline...\n"
    initial_html = """
    <div class="kpi-container" style="display: flex; gap: 15px; margin-bottom: 15px;">
        <div class="kpi-card" style="flex: 1; background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 11px; color: #8a99ad;">⚡ PROCESSING SPEED</div>
            <div style="font-size: 20px; font-weight: bold; color: #3b82f6; margin-top: 5px;">Initializing...</div>
        </div>
        <div class="kpi-card" style="flex: 1; background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 11px; color: #8a99ad;">⏱️ TIME REMAINING (ETA)</div>
            <div style="font-size: 20px; font-weight: bold; color: #eab308; margin-top: 5px;">--:--</div>
        </div>
        <div class="kpi-card" style="flex: 1; background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 11px; color: #8a99ad;">📊 PROGRESS</div>
            <div style="font-size: 20px; font-weight: bold; color: #10b981; margin-top: 5px;">0 / 0 (0%)</div>
        </div>
    </div>
    """
    yield None, logs, initial_html
    
    if source_img is None or target_video is None:
        logs += "[Error] Missing source image or target video.\n"
        yield None, logs, initial_html
        return
        
    active_engine = engine
        
    def log_callback(msg):
        nonlocal logs
        logs += f"{msg}\n"
        
    # Determine the execution provider based on device_mode and selected_gpus
    if device_mode == "GPU Mode" and selected_gpus:
        selected_gpu = selected_gpus[0] if isinstance(selected_gpus, (list, tuple)) else selected_gpus
        gpus_to_use = [selected_gpu]
        provider_log = f"GPU Mode (Selected GPU: {selected_gpu})"
    else:
        gpus_to_use = []
        provider_log = "CPU Only"
        
    logs += f"[Hardware] Selecting processing devices: {provider_log}...\n"
    yield None, logs, initial_html
    
    # Load selected models dynamically
    logs += f"[Model] Selected Swapper: {swapper_model}\n"
    logs += f"[Model] Selected Restorer: {restorer_model}\n"
    yield None, logs, initial_html
    active_engine.load_swapper(swapper_model)
    active_engine.load_restorer(restorer_model)
    
    # Create temporary file for output video
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "swapped_output.mp4")
    
    # Progress callback
    def progress_callback(ratio):
        progress(ratio, desc=f"Processing frames ({int(ratio*100)}%)...")
        
    try:
        # Get generator for real-time progress yielding
        generator = active_engine.process_video(
            source_img_path=source_img,
            target_video_path=target_video,
            output_path=output_path,
            enhance=enhance,
            enhance_strength=enhance_strength,
            match_color=match_color,
            match_scale=match_scale,
            custom_scale=custom_scale,
            det_thresh=det_thresh,
            face_upscale_resolution=face_upscale_resolution,
            handle_occlusions=handle_occlusions,
            frame_step=int(frame_step),
            det_size=int(det_size_val),
            batch_size=int(batch_size),
            progress_callback=None,
            log_callback=log_callback,
            swap_blend_strength=swap_blend_strength,
            match_face_shape=match_face_shape,
            selected_gpus=gpus_to_use,
            target_detector=target_detector,
            face_mask_type=face_mask_type
        )
        
        for update in generator:
            if isinstance(update, tuple) and len(update) == 5:
                f_idx, tot_frames, speed, elapsed, eta = update
                
                # Update Gradio visual progress bar
                if progress is not None and callable(progress):
                    progress(f_idx / tot_frames, desc="")
                
                # Format time string nicely
                el_m, el_s = divmod(int(elapsed), 60)
                et_m, et_s = divmod(int(eta), 60)
                
                status_block = f"\n⏱️ Elapsed Time: {el_m:02d}m {el_s:02d}s | ETA: {et_m:02d}m {et_s:02d}s\n"
                status_block += f"⚡ Speed: {speed:.1f} FPS (Frames per second)\n"
                status_block += f"📊 Total Progress: {f_idx}/{tot_frames} frames ({f_idx/tot_frames*100:.1f}%)\n"
                
                status_html = f"""
                <div class="kpi-container" style="display: flex; gap: 15px; margin-bottom: 15px;">
                    <div class="kpi-card" style="flex: 1; background: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(59, 130, 246, 0.3);">
                        <div style="font-size: 11px; color: #93c5fd; font-weight: 600;">⚡ PROCESSING SPEED</div>
                        <div style="font-size: 20px; font-weight: bold; color: #3b82f6; margin-top: 5px;">{speed:.1f} FPS</div>
                    </div>
                    <div class="kpi-card" style="flex: 1; background: rgba(234, 179, 8, 0.1); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(234, 179, 8, 0.3);">
                        <div style="font-size: 11px; color: #fde047; font-weight: 600;">⏱️ TIME REMAINING (ETA)</div>
                        <div style="font-size: 20px; font-weight: bold; color: #eab308; margin-top: 5px;">{et_m:02d}m {et_s:02d}s</div>
                    </div>
                    <div class="kpi-card" style="flex: 1; background: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(16, 185, 129, 0.3);">
                        <div style="font-size: 11px; color: #6ee7b7; font-weight: 600;">📊 PROGRESS</div>
                        <div style="font-size: 20px; font-weight: bold; color: #10b981; margin-top: 5px;">{f_idx} / {tot_frames} ({f_idx/tot_frames*100:.1f}%)</div>
                    </div>
                </div>
                """
                yield None, logs, status_html
            elif isinstance(update, tuple) and len(update) == 2 and update[0] == "COMPLETED":
                final_path = update[1]
                
                # Copy output video (defaults to output/ directory if path is empty)
                dest = save_path.strip() if (save_path and save_path.strip()) else ""
                if not dest:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("output", exist_ok=True)
                    dest = os.path.join("output", f"swapped_video_{timestamp}.mp4")
                    
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
                    import shutil
                    shutil.copy(final_path, dest)
                    logs += f"[System] Output video successfully saved to: {os.path.abspath(dest)}\n"
                except Exception as e:
                    logs += f"[Warning] Could not save output video: {str(e)}\n"
                        
                total_duration = time.time() - start_swap_time
                logs += f"[System] Video swap completed successfully in {total_duration:.2f} seconds!\n"
                completed_html = """
                <div class="kpi-container" style="display: flex; gap: 15px; margin-bottom: 15px;">
                    <div class="kpi-card" style="flex: 1; background: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(16, 185, 129, 0.3);">
                        <div style="font-size: 11px; color: #6ee7b7; font-weight: 600;">⚡ PROCESSING SPEED</div>
                        <div style="font-size: 20px; font-weight: bold; color: #10b981; margin-top: 5px;">Done</div>
                    </div>
                    <div class="kpi-card" style="flex: 1; background: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(16, 185, 129, 0.3);">
                        <div style="font-size: 11px; color: #6ee7b7; font-weight: 600;">⏱️ TIME REMAINING (ETA)</div>
                        <div style="font-size: 20px; font-weight: bold; color: #10b981; margin-top: 5px;">Completed</div>
                    </div>
                    <div class="kpi-card" style="flex: 1; background: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(16, 185, 129, 0.3);">
                        <div style="font-size: 11px; color: #6ee7b7; font-weight: 600;">📊 PROGRESS</div>
                        <div style="font-size: 20px; font-weight: bold; color: #10b981; margin-top: 5px;">100%</div>
                    </div>
                </div>
                """
                yield final_path, logs, completed_html
    except Exception as e:
        import traceback
        logs += f"[Error] Video execution failed: {str(e)}\n"
        logs += traceback.format_exc() + "\n"
        error_html = f"""
        <div class="kpi-container" style="display: flex; gap: 15px; margin-bottom: 15px;">
            <div class="kpi-card" style="flex: 1; background: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(239, 68, 68, 0.3);">
                <div style="font-size: 11px; color: #fca5a5; font-weight: 600;">❌ STATUS</div>
                <div style="font-size: 20px; font-weight: bold; color: #ef4444; margin-top: 5px;">Failed: {str(e)[:40]}...</div>
            </div>
        </div>
        """
        yield None, logs, error_html

def _resolve_uploaded_path(upload_value):
    if isinstance(upload_value, dict):
        return upload_value.get("path") or upload_value.get("name")
    return getattr(upload_value, "name", upload_value)

def _make_proxy_head_rgba(width, height):
    head = np.zeros((height, width, 4), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (max(4, int(width * 0.38)), max(6, int(height * 0.46)))
    cv2.ellipse(head, center, axes, 0, 0, 360, (178, 142, 118, 235), -1)
    cv2.ellipse(head, (center[0] - width // 10, center[1] - height // 10), (max(2, width // 16), max(2, height // 22)), 0, 0, 360, (55, 42, 38, 240), -1)
    cv2.ellipse(head, (center[0] + width // 10, center[1] - height // 10), (max(2, width // 16), max(2, height // 22)), 0, 0, 360, (55, 42, 38, 240), -1)
    cv2.ellipse(head, (center[0], center[1] + height // 8), (max(3, width // 9), max(2, height // 35)), 0, 0, 180, (88, 48, 48, 230), 2)
    alpha = head[:, :, 3].astype(np.float32)
    blur = max(5, (min(width, height) // 12) | 1)
    head[:, :, 3] = cv2.GaussianBlur(alpha, (blur, blur), 0).astype(np.uint8)
    return head

def _overlay_rgba(frame, overlay, center_x, center_y):
    h, w = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    x1 = int(center_x - ow / 2)
    y1 = int(center_y - oh / 2)
    x2 = x1 + ow
    y2 = y1 + oh
    ox1 = max(0, -x1)
    oy1 = max(0, -y1)
    ox2 = ow - max(0, x2 - w)
    oy2 = oh - max(0, y2 - h)
    bx1 = max(0, x1)
    by1 = max(0, y1)
    bx2 = min(w, x2)
    by2 = min(h, y2)
    if bx2 <= bx1 or by2 <= by1 or ox2 <= ox1 or oy2 <= oy1:
        return frame
    patch = overlay[oy1:oy2, ox1:ox2]
    rgb = patch[:, :, :3].astype(np.float32)
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    base = frame[by1:by2, bx1:bx2].astype(np.float32)
    frame[by1:by2, bx1:bx2] = np.clip(rgb * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)
    return frame

def _read_glb_chunks(model_path):
    import json
    import struct

    with open(model_path, "rb") as f:
        data = f.read()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("Not a valid GLB file")

    offset = 12
    json_chunk = None
    bin_chunk = None
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            json_chunk = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk

    if json_chunk is None or bin_chunk is None:
        raise ValueError("GLB must contain JSON and BIN chunks")
    return json_chunk, bin_chunk

def _accessor_array(gltf, bin_chunk, accessor_index):
    import numpy as _np

    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    comp_type = accessor["componentType"]
    dtype_map = {
        5120: _np.int8,
        5121: _np.uint8,
        5122: _np.int16,
        5123: _np.uint16,
        5125: _np.uint32,
        5126: _np.float32,
    }
    comp_count_map = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }
    dtype = dtype_map[comp_type]
    comp_count = comp_count_map[accessor["type"]]
    count = accessor["count"]
    byte_offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    byte_stride = view.get("byteStride")
    item_size = _np.dtype(dtype).itemsize * comp_count

    raw = memoryview(bin_chunk)[byte_offset:byte_offset + view["byteLength"]]
    if byte_stride and byte_stride != item_size:
        rows = []
        for i in range(count):
            start = i * byte_stride
            rows.append(_np.frombuffer(raw[start:start + item_size], dtype=dtype, count=comp_count))
        arr = _np.vstack(rows)
    else:
        arr = _np.frombuffer(raw[:count * item_size], dtype=dtype, count=count * comp_count).reshape((count, comp_count))
    return arr.copy()

def _embedded_image_array(gltf, bin_chunk, image_index):
    from io import BytesIO
    from PIL import Image

    image_info = gltf["images"][image_index]
    view = gltf["bufferViews"][image_info["bufferView"]]
    start = view.get("byteOffset", 0)
    end = start + view["byteLength"]
    image = Image.open(BytesIO(bytes(bin_chunk[start:end]))).convert("RGBA")
    return np.array(image, dtype=np.uint8)

def _glb_texture_images(gltf, bin_chunk):
    images = {}
    for tex_index, texture in enumerate(gltf.get("textures", [])):
        source_index = texture.get("source")
        if source_index is None:
            continue
        images[tex_index] = _embedded_image_array(gltf, bin_chunk, source_index)
    return images

def _sample_texture_color(texture_img, uv):
    if texture_img is None:
        return None
    h, w = texture_img.shape[:2]
    u = float(uv[0] % 1.0)
    v = float(uv[1] % 1.0)
    x = int(np.clip(u * (w - 1), 0, w - 1))
    y = int(np.clip((1.0 - v) * (h - 1), 0, h - 1))
    rgba = texture_img[y, x].astype(np.float32)
    alpha = rgba[3] / 255.0
    return rgba[:3], alpha

def _node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float32).reshape(4, 4).T

    t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    q = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
    x, y, z, w = q
    rot = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w, 0],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w, 0],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot[:3, :3] * s.reshape(1, 3)
    mat[:3, 3] = t
    return mat

def _collect_glb_triangles(model_path):
    gltf, bin_chunk = _read_glb_chunks(model_path)
    nodes = gltf.get("nodes", [])
    scene_index = gltf.get("scene", 0)
    scene_nodes = gltf.get("scenes", [{}])[scene_index].get("nodes", list(range(len(nodes))))
    triangles = []
    texture_images = _glb_texture_images(gltf, bin_chunk)

    def material_diffuse(material_index):
        if material_index is None:
            return np.array([190, 160, 135], dtype=np.float32), 1.0, None
        material = gltf.get("materials", [{}])[material_index]
        pbr = material.get("pbrMetallicRoughness", {})
        spec_gloss = material.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness", {})

        color_factor = pbr.get("baseColorFactor") or spec_gloss.get("diffuseFactor") or [0.75, 0.62, 0.52, 1.0]
        color = np.array(color_factor[:3], dtype=np.float32)
        alpha = float(color_factor[3]) if len(color_factor) > 3 else 1.0

        tex_info = pbr.get("baseColorTexture") or spec_gloss.get("diffuseTexture")
        tex_img = texture_images.get(tex_info.get("index")) if tex_info else None
        return np.clip(color * 255.0, 0, 255), alpha, tex_img

    def walk(node_index, parent_mat):
        node = nodes[node_index]
        mat = parent_mat @ _node_matrix(node)
        if "mesh" in node:
            mesh = gltf["meshes"][node["mesh"]]
            for primitive in mesh.get("primitives", []):
                attrs = primitive.get("attributes", {})
                if "POSITION" not in attrs:
                    continue
                pos = _accessor_array(gltf, bin_chunk, attrs["POSITION"]).astype(np.float32)
                pos_h = np.hstack([pos, np.ones((pos.shape[0], 1), dtype=np.float32)])
                pos = (pos_h @ mat.T)[:, :3]

                indices = primitive.get("indices")
                if indices is not None:
                    idx = _accessor_array(gltf, bin_chunk, indices).reshape(-1).astype(np.int64)
                else:
                    idx = np.arange(pos.shape[0], dtype=np.int64)

                material_color, material_alpha, texture_img = material_diffuse(primitive.get("material"))
                uvs = None
                if texture_img is not None and "TEXCOORD_0" in attrs:
                    uvs = _accessor_array(gltf, bin_chunk, attrs["TEXCOORD_0"]).astype(np.float32)

                for i in range(0, len(idx) - 2, 3):
                    tri_idx = idx[i:i + 3]
                    tri = pos[tri_idx]
                    if tri.shape == (3, 3):
                        color = material_color.copy()
                        alpha = material_alpha
                        if uvs is not None:
                            sampled = _sample_texture_color(texture_img, np.mean(uvs[tri_idx], axis=0))
                            if sampled is not None:
                                tex_color, tex_alpha = sampled
                                color = color * (tex_color / 255.0)
                                alpha *= tex_alpha
                        if alpha > 0.02:
                            triangles.append((tri, color, alpha))
        for child in node.get("children", []):
            walk(child, mat)

    for root in scene_nodes:
        walk(root, np.eye(4, dtype=np.float32))
    if not triangles:
        raise ValueError("No triangles found in GLB")
    return triangles

def _render_model_rgba(model_path, width, height):
    cache_key = (model_path, int(width), int(height), os.path.getmtime(model_path))
    if cache_key in _HEAD3D_RENDER_CACHE:
        return _HEAD3D_RENDER_CACHE[cache_key].copy()

    ext = os.path.splitext(model_path)[1].lower()
    if ext != ".glb":
        return _make_proxy_head_rgba(width, height)

    triangles = _collect_glb_triangles(model_path)
    all_pts = np.vstack([tri for tri, _, _ in triangles])
    center = (all_pts.min(axis=0) + all_pts.max(axis=0)) * 0.5
    span = np.maximum(all_pts.max(axis=0) - all_pts.min(axis=0), 1e-6)
    scale = 0.82 * min(width / span[0], height / max(span[1], span[2], 1e-6))

    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    light = np.array([-0.25, -0.55, 1.0], dtype=np.float32)
    light /= np.linalg.norm(light)
    projected = []

    for tri, color, alpha in triangles:
        pts = (tri - center) * scale
        x = pts[:, 0] + width * 0.5
        y = -pts[:, 1] + height * 0.52
        z = pts[:, 2]
        p2 = np.stack([x, y], axis=1).astype(np.int32)
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        shade = 0.42 + 0.58 * max(0.0, float(np.dot(normal, light)))
        projected.append((float(np.mean(z)), p2, np.clip(color * shade, 0, 255).astype(np.uint8), alpha))

    for _, p2, color, alpha in sorted(projected, key=lambda item: item[0]):
        if np.max(p2[:, 0]) < 0 or np.min(p2[:, 0]) >= width or np.max(p2[:, 1]) < 0 or np.min(p2[:, 1]) >= height:
            continue
        cv2.fillConvexPoly(canvas, p2, (int(color[0]), int(color[1]), int(color[2]), int(np.clip(alpha * 245, 0, 245))))

    alpha = canvas[:, :, 3]
    if np.count_nonzero(alpha) == 0:
        return _make_proxy_head_rgba(width, height)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    canvas[:, :, 3] = alpha
    _HEAD3D_RENDER_CACHE[cache_key] = canvas.copy()
    return canvas

def _detect_head_anchor(frame, anchor_mode, head_scale, vertical_offset, pose_result=None):
    height, width = frame.shape[:2]
    if pose_result is not None and pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks.landmark
        left_shoulder = lm[11]
        right_shoulder = lm[12]
        shoulder_vis = min(left_shoulder.visibility, right_shoulder.visibility)
        if shoulder_vis > 0.25:
            lx, ly = left_shoulder.x * width, left_shoulder.y * height
            rx, ry = right_shoulder.x * width, right_shoulder.y * height
            sx = (lx + rx) * 0.5
            sy = (ly + ry) * 0.5
            shoulder_w = max(float(np.hypot(lx - rx, ly - ry)), width * 0.08)

            if anchor_mode == "Face Re-entry Assist" and lm[0].visibility > 0.35:
                anchor_x = lm[0].x * width
                anchor_y = lm[0].y * height
            elif anchor_mode == "Manual Center Anchor":
                anchor_x = width * 0.5
                anchor_y = height * 0.38
            else:
                anchor_x = sx
                anchor_y = sy - shoulder_w * (0.95 - vertical_offset * 0.45)

            head_h = shoulder_w * 1.18 * float(head_scale)
            return np.array([anchor_x, anchor_y, head_h * 0.78, head_h], dtype=np.float32)

    if anchor_mode == "Manual Center Anchor":
        head_h = height * 0.28 * float(head_scale)
        return np.array([width * 0.5, height * (0.38 + vertical_offset * 0.15), head_h * 0.78, head_h], dtype=np.float32)
    return None

def _apply_proxy_head(frame, anchor, model_path=None):
    if anchor is None:
        return frame
    ow = max(24, int(round(anchor[2] / 8.0) * 8))
    oh = max(32, int(round(anchor[3] / 8.0) * 8))
    try:
        head_rgba = _render_model_rgba(model_path, ow, oh) if model_path else _make_proxy_head_rgba(ow, oh)
    except Exception as e:
        print(f"[3D Head] GLB software render failed, using proxy head: {e}")
        head_rgba = _make_proxy_head_rgba(ow, oh)
    return _overlay_rgba(frame, head_rgba, anchor[0], anchor[1])

def perform_3d_head_image_composite(head_model, target_image, anchor_mode, render_backend, head_scale, vertical_offset):
    logs = "[3D Head Image] Initializing image composite pipeline...\n"
    model_path = _resolve_uploaded_path(head_model)

    if not model_path:
        return None, logs + "[Error] Upload a 3D head model first.\n"
    if target_image is None:
        return None, logs + "[Error] Upload a target image first.\n"

    supported_exts = (".glb", ".gltf", ".obj", ".fbx", ".blend")
    model_ext = os.path.splitext(str(model_path))[1].lower()
    if model_ext not in supported_exts:
        return None, logs + f"[Error] Unsupported 3D model format: {model_ext}. Use GLB, GLTF, OBJ, FBX, or BLEND.\n"

    if render_backend not in ("CPU Preview (Working)", "CPU Preview"):
        logs += f"[Warning] {render_backend} is reserved for the full renderer path. Running CPU Preview backend now.\n"

    try:
        import mediapipe as mp

        frame = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.35) as pose:
            result = pose.process(target_image)

        anchor = _detect_head_anchor(frame, anchor_mode, head_scale, vertical_offset, result)
        if anchor is None:
            return None, logs + "[Error] Could not detect neck/shoulders. Try Manual Center Anchor.\n"

        frame = _apply_proxy_head(frame, anchor, model_path=model_path)
        logs += f"[3D Head Image] Model: {model_path}\n"
        logs += f"[3D Head Image] Anchor mode: {anchor_mode}\n"
        logs += "[3D Head Image] CPU Preview rendered the uploaded model when GLB software rendering is available.\n"
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), logs
    except Exception as e:
        import traceback
        logs += f"[Error] 3D Head image composite failed: {str(e)}\n"
        logs += traceback.format_exc() + "\n"
        return None, logs

def perform_3d_head_composite(head_model, target_video, anchor_mode, render_backend, head_scale, vertical_offset, save_path=""):
    logs = "[3D Head] Initializing separate 3D head composite pipeline...\n"

    model_path = _resolve_uploaded_path(head_model)
    video_path = target_video

    if not model_path:
        return None, logs + "[Error] Upload a 3D head model first.\n"
    if not video_path:
        return None, logs + "[Error] Upload a target video first.\n"

    supported_exts = (".glb", ".gltf", ".obj", ".fbx", ".blend")
    model_ext = os.path.splitext(str(model_path))[1].lower()
    if model_ext not in supported_exts:
        return None, logs + f"[Error] Unsupported 3D model format: {model_ext}. Use GLB, GLTF, OBJ, FBX, or BLEND.\n"

    if render_backend not in ("CPU Preview (Working)", "CPU Preview"):
        logs += f"[Warning] {render_backend} is reserved for the full renderer path. Running CPU Preview backend now.\n"

    logs += f"[3D Head] Model: {model_path}\n"
    logs += f"[3D Head] Target video: {video_path}\n"
    logs += f"[3D Head] Anchor mode: {anchor_mode}\n"
    logs += "[3D Head] Render backend: CPU Preview (Working)\n"
    logs += f"[3D Head] Scale: {head_scale:.2f}, vertical offset: {vertical_offset:.2f}\n"

    try:
        import mediapipe as mp
        import imageio_ffmpeg
        import shutil
        import subprocess

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, logs + "[Error] Could not open target video.\n"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_dir = tempfile.mkdtemp(prefix="head3d_")
        silent_path = os.path.join(temp_dir, "head3d_silent.mp4")
        final_tmp = os.path.join(temp_dir, "head3d_final.mp4")
        writer = cv2.VideoWriter(silent_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, logs + "[Error] Could not create output video writer.\n"

        mp_pose = mp.solutions.pose
        last_anchor = None
        processed = 0
        detected = 0

        with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.35, min_tracking_confidence=0.35) as pose:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                anchor = None

                anchor = _detect_head_anchor(frame, anchor_mode, head_scale, vertical_offset, result)
                if anchor is not None and result.pose_landmarks:
                    detected += 1

                if anchor is not None:
                    last_anchor = anchor if last_anchor is None else (0.28 * anchor + 0.72 * last_anchor)

                if last_anchor is not None:
                    frame = _apply_proxy_head(frame, last_anchor, model_path=model_path)

                writer.write(frame)
                processed += 1

        cap.release()
        writer.release()

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        merge_cmd = [
            ffmpeg_bin, "-y", "-i", silent_path, "-i", video_path,
            "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-c:a", "aac", "-shortest", final_tmp,
        ]
        merge = subprocess.run(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output_source = final_tmp if merge.returncode == 0 and os.path.exists(final_tmp) else silent_path

        dest = save_path.strip() if (save_path and save_path.strip()) else ""
        if not dest:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("output", exist_ok=True)
            dest = os.path.join("output", f"head3d_preview_{timestamp}.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        shutil.copy(output_source, dest)

        logs += f"[3D Head] Processed {processed}/{total_frames or processed} frames.\n"
        logs += f"[3D Head] Pose anchors detected on {detected} frames.\n"
        logs += "[3D Head] CPU Preview rendered the uploaded model when GLB software rendering is available.\n"
        logs += f"[3D Head] Output saved to: {os.path.abspath(dest)}\n"
        shutil.rmtree(temp_dir, ignore_errors=True)
        return dest, logs
    except Exception as e:
        import traceback
        logs += f"[Error] 3D Head composite failed: {str(e)}\n"
        logs += traceback.format_exc() + "\n"
        return None, logs

# Custom CSS for Premium Dark UI theme
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Global Font Settings */
body, .gradio-container, .gradio-container * {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
}

/* Beautiful Animated Gradient Background */
body {
    background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(217, 70, 239, 0.12) 0%, transparent 40%),
                #030712 !important;
    background-attachment: fixed !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Glassmorphism Header Container */
.header-container {
    text-align: center;
    margin-bottom: 2.5rem;
    padding: 2.5rem 1.5rem;
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(16px) saturate(180%);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5),
                0 1px 1px 0 rgba(255, 255, 255, 0.1) inset;
}

/* Header Titles with Fuchsia-Violet-Indigo Gradient */
.header-title {
    background: linear-gradient(90deg, #c084fc 0%, #6366f1 50%, #38bdf8 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.05em !important;
    margin-bottom: 0.75rem !important;
    display: inline-block !important;
}

.header-subtitle {
    color: #9ca3af !important;
    font-size: 1.15rem !important;
    font-weight: 400 !important;
}

/* Hardware Status Badge */
.status-badge {
    display: inline-block;
    padding: 0.5rem 1.25rem;
    border-radius: 9999px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 1.25rem;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.status-gpu {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
    box-shadow: 0 0 15px rgba(52, 211, 153, 0.15);
}

.status-cpu {
    background: rgba(249, 115, 22, 0.12);
    color: #fb923c;
    border: 1px solid rgba(251, 146, 60, 0.3);
    box-shadow: 0 0 15px rgba(251, 146, 60, 0.15);
}

/* Modern Tab Panels */
.tabs {
    border: none !important;
    background: transparent !important;
}

.tab-nav {
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-bottom: 4px !important;
    margin-bottom: 1.5rem !important;
}

.tab-nav button {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #9ca3af !important;
    border: none !important;
    background: transparent !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 12px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.tab-nav button.selected {
    color: #ffffff !important;
    background: rgba(99, 102, 241, 0.15) !important;
    box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.3) inset !important;
}

.tab-nav button:hover:not(.selected) {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

/* Panels, Boxes, and Containers */
.block {
    background: rgba(17, 24, 39, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
    transition: all 0.3s ease !important;
    overflow: visible !important;
}

.block:hover {
    border-color: rgba(255, 255, 255, 0.1) !important;
}

.block:focus-within {
    z-index: 9999 !important;
}

/* Forms & Input Elements */
.gradio-dropdown, .gradio-slider, .gradio-checkbox, .gradio-radio {
    border: none !important;
    background: transparent !important;
    overflow: visible !important;
}

.gradio-dropdown div.wrap, .gradio-dropdown .wrap {
    overflow: visible !important;
}

.gradio-textbox input, textarea, select {
    background-color: rgba(3, 7, 18, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #f3f4f6 !important;
    transition: all 0.2s ease !important;
}

.gradio-textbox input:focus, textarea:focus, select:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

/* Premium Buttons Styling */
button.primary {
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #d946ef 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
    filter: brightness(1.1) !important;
}

button.primary:active {
    transform: translateY(0) !important;
}

button.secondary {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #f3f4f6 !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    transition: all 0.25s ease !important;
}

button.secondary:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    transform: translateY(-1px) !important;
}

/* Logs and estimated boxes */
.log-box textarea {
    font-family: 'Fira Code', 'Courier New', Courier, monospace !important;
    background-color: #030712 !important;
    color: #38bdf8 !important;
    font-size: 0.9rem !important;
    border-color: rgba(255, 255, 255, 0.06) !important;
}

.estimate-box {
    background: rgba(30, 41, 59, 0.5) !important;
    padding: 1rem !important;
    border-radius: 14px !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.08) !important;
}

.enhancement-settings {
    background: rgba(30, 41, 59, 0.4) !important;
    padding: 1.25rem !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    margin-top: 1rem !important;
    overflow: visible !important;
}


/* Footer layout */
.footer-text {
    text-align: center;
    color: #4b5563 !important;
    margin-top: 3rem;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}
"""

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            gr.Image(value="logo.png", show_label=False, interactive=False, height=180, container=False)
        with gr.Column(scale=1):
            pass

    with gr.Tabs():
        # --- TAB 1: Image Swapping ---
        with gr.TabItem("🖼️ Image Swap"):
            with gr.Row():
                # Column 1: Uploads (Source & Target)
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Upload Images")
                    source_image = gr.Image(type="numpy", label="Source Face (Will be taken from this image)", height=280)
                    target_image = gr.Image(type="numpy", label="Target Image (Where the face will be placed)", height=280)
                    batch_target_images = gr.File(
                        label="Batch Target Images / Folder Upload",
                        file_count="directory",
                        type="filepath"
                    )
                    batch_target_folder_path = gr.Textbox(label="Batch Target Local Folder Path (Optional)", placeholder="e.g. C:\\Users\\fds\\Pictures\\targets", value="")
                
                # Column 2: Swapped Result & Controls
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Output & Swap Action")
                    output_image = gr.Image(type="numpy", label="Swapped Result", height=280)
                    batch_output_gallery = gr.Gallery(label="Batch Results", columns=3, height=280, show_label=True)
                    batch_output_zip = gr.File(label="Batch Results Zip")
                    btn_swap_img = gr.Button("🚀 Start Face Swap", variant="primary")
                    btn_clear_vram_img = gr.Button("🧹 Clear VRAM Memory", variant="secondary")
                    btn_batch_swap_img = gr.Button("Start Batch Folder Swap", variant="secondary")
                    image_status = gr.Textbox(label="System Process Logs", interactive=False, lines=8, max_lines=12, elem_classes="log-box")
                    
                # Column 3: Settings & Models
                with gr.Column(scale=1):
                    gr.Markdown("### 3. Model & Settings Selection")
                    
                    with gr.Group():
                        gr.Markdown("🤖 **Model Selection**")
                        swapper_model_img = gr.Dropdown(choices=available_swappers, value=default_swapper, label="Face Swapper Model")
                        restorer_model_img = gr.Dropdown(choices=available_restorers, value=default_restorer, label="Face Restorer (Enhancer) Model")
                        target_detector_img = gr.Dropdown(choices=["SCRFD (Default)", "YOLOv11-Face"], value="SCRFD (Default)", label="Target Face Detector Model")
                        face_mask_type_img = gr.Dropdown(choices=["InsightFace 106-Point", "MediaPipe FaceMesh (468-Point)", "MediaPipe FaceMesh 3D Pose (Best)"], value="MediaPipe FaceMesh 3D Pose (Best)", label="Face Masking/Blending Method")
                    
                    with gr.Group():
                        gr.Markdown("⚙️ **Hardware Device**")
                        device_mode_img = gr.Radio(choices=["GPU Mode", "CPU Only"], value="GPU Mode" if available_gpus else "CPU Only", label="Processing Device Mode")
                        gpu_choice_img = gr.Dropdown(choices=available_gpus, value=available_gpus[0] if available_gpus else None, label="Select GPU Device", visible=(len(available_gpus) > 0))
                    
                    def update_gpu_visibility_img(mode):
                        return gr.update(visible=(mode == "GPU Mode" and len(available_gpus) > 0))
                        
                    device_mode_img.change(
                        fn=update_gpu_visibility_img,
                        inputs=[device_mode_img],
                        outputs=[gpu_choice_img]
                    )
                    
                    custom_save_path_img = gr.Textbox(label="Save Output Path / Batch Folder (Optional)", placeholder="single: outputs/result.png | batch: outputs/batch_folder", value="")
                    
                    # Collapsible Accordion for Advanced Parameters
                    with gr.Accordion("🔧 Advanced Settings & Realism Parameters", open=False):
                        enhance_img = gr.Checkbox(label="Enhance Face Details (GFPGAN Restorer)", value=False)
                        enhance_strength_img = gr.Slider(label="Enhance Strength", minimum=0.0, maximum=1.0, value=0.8, step=0.05)
                        handle_occlusions_img = gr.Checkbox(label="Enable Occlusion Masking (Keep hands/hair/arms in foreground)", value=True)
                        swap_blend_strength_img = gr.Slider(label="Face Swap Blend Strength (Identity Likeness)", minimum=0.5, maximum=2.0, value=1.0, step=0.05)
                        match_color_img = gr.Checkbox(label="Match Lighting & Skin Tone (Color Transfer)", value=False)
                        match_face_shape_img = gr.Checkbox(label="Match Face Shape Aspect Ratio (Slim/Wide Jawline to Source)", value=True)
                        match_scale_img = gr.Checkbox(label="Match Source Face Size (Scale to Source)", value=False, visible=False)
                        custom_scale_img = gr.Slider(label="Custom Face Scale Ratio", minimum=0.5, maximum=2.0, value=1.0, step=0.05, visible=False)
                        det_thresh_img = gr.Slider(label="Face Detection Confidence Threshold (Auto fallback)", minimum=0.1, maximum=0.9, value=0.5, step=0.05)
                        face_upscale_res_img = gr.Dropdown(choices=["128", "256", "512", "1024", "2048"], value="512", label="Face Resolution Upscaling Size")
            
            btn_swap_img.click(
                fn=perform_image_swap,
                inputs=[
                    source_image, target_image, 
                    enhance_img, enhance_strength_img, match_color_img, match_scale_img, custom_scale_img, det_thresh_img, face_upscale_res_img, handle_occlusions_img,
                    swapper_model_img, restorer_model_img, device_mode_img, gpu_choice_img,
                    custom_save_path_img, swap_blend_strength_img, match_face_shape_img, target_detector_img, face_mask_type_img
                ],
                outputs=[output_image, image_status]
            )

            btn_batch_swap_img.click(
                fn=perform_batch_image_swap,
                inputs=[
                    source_image, batch_target_images, batch_target_folder_path,
                    enhance_img, enhance_strength_img, match_color_img, match_scale_img, custom_scale_img, det_thresh_img, face_upscale_res_img, handle_occlusions_img,
                    swapper_model_img, restorer_model_img, device_mode_img, gpu_choice_img,
                    custom_save_path_img, swap_blend_strength_img, match_face_shape_img, target_detector_img, face_mask_type_img
                ],
                outputs=[batch_output_gallery, batch_output_zip, image_status]
            )
            
            btn_clear_vram_img.click(
                fn=clear_vram_callback,
                outputs=[image_status]
            )

        # --- TAB 2: Video Swapping ---
        with gr.TabItem("🎥 Video Swap"):
            with gr.Row():
                # Column 1: Uploads (Source Image & Target Video)
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Upload Source & Target")
                    source_video_image = gr.Image(type="filepath", label="Source Face Image (Face to use)", height=280)
                    target_video = gr.Video(label="Target Video (To swap face onto)", height=280)
                    
                    gr.Markdown("🔍 **Frame Preview Selector**")
                    frame_slider = gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Select Frame to Preview", visible=True)
                    btn_preview_frame = gr.Button("🔍 Generate Frame Swap Preview", variant="secondary")
                    
                    # Estimate box
                    estimate_output = gr.Markdown("### Estimated Processing Time: No video uploaded.", elem_classes="estimate-box")
                
                # Column 2: Results & Monitor
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Output & Swap Action")
                    preview_img = gr.Image(type="numpy", label="Live Frame Preview", height=280)
                    output_video = gr.Video(label="Final Swapped Video Result", height=280)
                    
                    btn_swap_vid = gr.Button("🚀 Start Face Swap Video", variant="primary")
                    btn_clear_vram_vid = gr.Button("🧹 Clear VRAM Memory", variant="secondary")
                    
                    realtime_status = gr.HTML(
                        value="""
                        <div class="kpi-container" style="display: flex; gap: 10px; margin-bottom: 10px;">
                            <div class="kpi-card" style="flex: 1; background: rgba(128,128,128,0.1); padding: 8px; border-radius: 6px; text-align: center; border: 1px solid rgba(128,128,128,0.2);">
                                <div style="font-size: 9px; color: #8a99ad; font-weight: 600;">⚡ SPEED</div>
                                <div style="font-size: 15px; font-weight: bold; color: #3b82f6; margin-top: 2px;">- FPS</div>
                            </div>
                            <div class="kpi-card" style="flex: 1; background: rgba(128,128,128,0.1); padding: 8px; border-radius: 6px; text-align: center; border: 1px solid rgba(128,128,128,0.2);">
                                <div style="font-size: 9px; color: #8a99ad; font-weight: 600;">⏱️ ETA</div>
                                <div style="font-size: 15px; font-weight: bold; color: #eab308; margin-top: 2px;">--:--</div>
                            </div>
                            <div class="kpi-card" style="flex: 1; background: rgba(128,128,128,0.1); padding: 8px; border-radius: 6px; text-align: center; border: 1px solid rgba(128,128,128,0.2);">
                                <div style="font-size: 9px; color: #8a99ad; font-weight: 600;">📊 PROGRESS</div>
                                <div style="font-size: 15px; font-weight: bold; color: #10b981; margin-top: 2px;">0%</div>
                            </div>
                        </div>
                        """,
                        label="Real-time Speed Monitor"
                    )
                    video_status = gr.Textbox(label="System Process Logs", interactive=False, lines=6, max_lines=10, elem_classes="log-box")
                    
                # Column 3: Settings & Advanced
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("🤖 **Model Selection**")
                        swapper_model_vid = gr.Dropdown(choices=available_swappers, value=default_swapper, label="Face Swapper Model")
                        restorer_model_vid = gr.Dropdown(choices=available_restorers, value=default_restorer, label="Face Restorer (Enhancer) Model")
                        target_detector_vid = gr.Dropdown(choices=["SCRFD (Default)", "YOLOv11-Face"], value="SCRFD (Default)", label="Target Face Detector Model")
                        face_mask_type_vid = gr.Dropdown(choices=["InsightFace 106-Point", "MediaPipe FaceMesh (468-Point)", "MediaPipe FaceMesh 3D Pose (Best)"], value="MediaPipe FaceMesh 3D Pose (Best)", label="Face Masking/Blending Method")
                    
                    with gr.Group():
                        gr.Markdown("⚙️ **Hardware Device**")
                        device_mode_vid = gr.Radio(choices=["GPU Mode", "CPU Only"], value="GPU Mode" if available_gpus else "CPU Only", label="Processing Device Mode")
                        gpu_choices_vid = gr.Dropdown(choices=available_gpus, value=available_gpus[0] if available_gpus else None, label="Select GPU Device", visible=(len(available_gpus) > 0))
                    
                    def update_gpu_visibility_vid(mode):
                        return gr.update(visible=(mode == "GPU Mode" and len(available_gpus) > 0))
                        
                    device_mode_vid.change(
                        fn=update_gpu_visibility_vid,
                        inputs=[device_mode_vid],
                        outputs=[gpu_choices_vid]
                    )
                    
                    custom_save_path_vid = gr.Textbox(label="Save Output Video to Path (Optional)", placeholder="e.g. outputs/result.mp4", value="")
                    
                    # Collapsible Accordion for Advanced Parameters
                    with gr.Accordion("🔧 Advanced Settings & Realism Parameters", open=False):
                        gr.Markdown("⚡ **Speed Optimization**")
                        frame_step_vid = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Frame Step / Skip (1 = Process all frames, 2 = Skip every other frame, etc.)")
                        det_size_vid = gr.Dropdown(choices=["320", "480", "640"], value="640", label="Face Detection Scan Size (Lower = Faster)")
                        batch_size_vid = gr.Dropdown(choices=["1", "2", "4", "6", "8", "12", "16", "32"], value="4", label="Execution Thread Count (Concurrent Threads)")
                        
                        gr.Markdown("🔧 **Enhancement & Realism**")
                        enhance_vid = gr.Checkbox(label="Enhance Face Details (GFPGAN)", value=False)
                        enhance_strength_vid = gr.Slider(label="Enhance Strength", minimum=0.0, maximum=1.0, value=0.8, step=0.05)
                        handle_occlusions_vid = gr.Checkbox(label="Enable Occlusion Masking (Keep hands/hair/arms in foreground)", value=True)
                        swap_blend_strength_vid = gr.Slider(label="Face Swap Blend Strength (Identity Likeness)", minimum=0.5, maximum=2.0, value=1.0, step=0.05)
                        match_color_vid = gr.Checkbox(label="Match Lighting & Skin Tone (Color Transfer)", value=False)
                        match_face_shape_vid = gr.Checkbox(label="Match Face Shape Aspect Ratio (Slim/Wide Jawline to Source)", value=True)
                        match_scale_vid = gr.Checkbox(label="Match Source Face Size (Scale to Source)", value=False, visible=False)
                        custom_scale_vid = gr.Slider(label="Custom Face Scale Ratio", minimum=0.5, maximum=2.0, value=1.0, step=0.05, visible=False)
                        det_thresh_vid = gr.Slider(label="Face Detection Confidence Threshold (Auto fallback)", minimum=0.1, maximum=0.9, value=0.5, step=0.05)
                        face_upscale_res_vid = gr.Dropdown(choices=["128", "256", "512", "1024", "2048"], value="512", label="Face Resolution Upscaling Size")
            
            # Trigger to load frame count & estimate time when target video changes
            target_video.upload(
                fn=load_video_info,
                inputs=[target_video, device_mode_vid],
                outputs=[frame_slider, video_status, estimate_output]
            )
            
            # Recalculate estimate if user swaps hardware device
            device_mode_vid.change(
                fn=calculate_estimated_time,
                inputs=[target_video, device_mode_vid],
                outputs=[estimate_output]
            )
            
            # Click trigger for single frame preview
            btn_preview_frame.click(
                fn=preview_selected_frame,
                inputs=[
                    source_video_image, target_video, frame_slider,
                    enhance_vid, enhance_strength_vid, match_color_vid, match_scale_vid, custom_scale_vid, det_thresh_vid, face_upscale_res_vid, handle_occlusions_vid,
                    swapper_model_vid, restorer_model_vid, device_mode_vid, gpu_choices_vid, swap_blend_strength_vid, match_face_shape_vid, target_detector_vid, face_mask_type_vid
                ],
                outputs=[preview_img, video_status]
            )
            
            # Auto-preview when slider is released (real-time frame preview)
            frame_slider.release(
                fn=preview_selected_frame,
                inputs=[
                    source_video_image, target_video, frame_slider,
                    enhance_vid, enhance_strength_vid, match_color_vid, match_scale_vid, custom_scale_vid, det_thresh_vid, face_upscale_res_vid, handle_occlusions_vid,
                    swapper_model_vid, restorer_model_vid, device_mode_vid, gpu_choices_vid, swap_blend_strength_vid, match_face_shape_vid, target_detector_vid, face_mask_type_vid
                ],
                outputs=[preview_img, video_status]
            )
 
            # Run full video swap
            btn_swap_vid.click(
                fn=perform_video_swap,
                inputs=[
                    source_video_image, target_video, 
                    enhance_vid, enhance_strength_vid, match_color_vid, match_scale_vid, custom_scale_vid, det_thresh_vid, face_upscale_res_vid, handle_occlusions_vid,
                    swapper_model_vid, restorer_model_vid, device_mode_vid, gpu_choices_vid,
                    frame_step_vid, det_size_vid, batch_size_vid,
                    custom_save_path_vid, swap_blend_strength_vid, match_face_shape_vid, target_detector_vid, face_mask_type_vid
                ],
                outputs=[output_video, video_status, realtime_status]
            )
            
            btn_clear_vram_vid.click(
                fn=clear_vram_callback,
                outputs=[video_status]
            )

    gr.Markdown("Built with Python, Gradio, ONNX Runtime, and OpenCV.", elem_classes="footer-text")

if __name__ == "__main__":
    print("[Gradio] Launching local-only server: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
