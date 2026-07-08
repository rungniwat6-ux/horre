import os
import sys

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
import onnxruntime as ort

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

def preview_selected_frame(source_img_path, video_path, frame_index, enhance, enhance_strength, match_color, match_scale, custom_scale, det_thresh, face_upscale_resolution, handle_occlusions, swapper_model, restorer_model, device_mode, selected_gpus, swap_blend_strength=0.85, match_face_shape=True, target_detector="SCRFD (Default)", face_mask_type="InsightFace 106-Point"):
    if not source_img_path or not video_path:
        return None, "[System] Upload both Source Image and Target Video to preview."
        
    active_engine = engine
        
    # Determine the execution provider based on device_mode and selected_gpus
    if device_mode == "GPU Mode" and selected_gpus:
        execution_device = selected_gpus[0]
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

def perform_image_swap(source_img, target_img, enhance, enhance_strength, match_color, match_scale, custom_scale, det_thresh, face_upscale_resolution, handle_occlusions, swapper_model, restorer_model, device_mode, selected_gpu, save_path="", swap_blend_strength=0.85, match_face_shape=True, target_detector="SCRFD (Default)", face_mask_type="InsightFace 106-Point"):
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

def perform_video_swap(source_img, target_video, enhance, enhance_strength, match_color, match_scale, custom_scale, det_thresh, face_upscale_resolution, handle_occlusions, swapper_model, restorer_model, device_mode, selected_gpus, frame_step, det_size_val, batch_size, save_path="", swap_blend_strength=0.85, match_face_shape=True, progress=gr.Progress(track_tqdm=False), target_detector="SCRFD (Default)", face_mask_type="InsightFace 106-Point"):
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
        gpus_to_use = selected_gpus
        provider_log = f"GPU Mode (Selected GPUs: {gpus_to_use})"
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
                
                # Column 2: Swapped Result & Controls
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Output & Swap Action")
                    output_image = gr.Image(type="numpy", label="Swapped Result", height=280)
                    btn_swap_img = gr.Button("🚀 Start Face Swap", variant="primary")
                    btn_clear_vram_img = gr.Button("🧹 Clear VRAM Memory", variant="secondary")
                    image_status = gr.Textbox(label="System Process Logs", interactive=False, lines=8, max_lines=12, elem_classes="log-box")
                    
                # Column 3: Settings & Models
                with gr.Column(scale=1):
                    gr.Markdown("### 3. Model & Settings Selection")
                    
                    with gr.Group():
                        gr.Markdown("🤖 **Model Selection**")
                        swapper_model_img = gr.Dropdown(choices=available_swappers, value=default_swapper, label="Face Swapper Model")
                        restorer_model_img = gr.Dropdown(choices=available_restorers, value=default_restorer, label="Face Restorer (Enhancer) Model")
                        target_detector_img = gr.Dropdown(choices=["SCRFD (Default)", "YOLOv11-Face"], value="SCRFD (Default)", label="Target Face Detector Model")
                        face_mask_type_img = gr.Dropdown(choices=["InsightFace 106-Point", "MediaPipe FaceMesh (468-Point)"], value="MediaPipe FaceMesh (468-Point)", label="Face Masking/Blending Method")
                    
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
                    
                    custom_save_path_img = gr.Textbox(label="Save Output Image to Path (Optional)", placeholder="e.g. outputs/result.png", value="")
                    
                    # Collapsible Accordion for Advanced Parameters
                    with gr.Accordion("🔧 Advanced Settings & Realism Parameters", open=False):
                        enhance_img = gr.Checkbox(label="Enhance Face Details (GFPGAN Restorer)", value=False)
                        enhance_strength_img = gr.Slider(label="Enhance Strength", minimum=0.0, maximum=1.0, value=0.8, step=0.05)
                        handle_occlusions_img = gr.Checkbox(label="Enable Occlusion Masking (Keep hands/hair/arms in foreground)", value=True)
                        swap_blend_strength_img = gr.Slider(label="Face Swap Blend Strength (Identity Likeness)", minimum=0.5, maximum=1.0, value=1.0, step=0.05)
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
                        face_mask_type_vid = gr.Dropdown(choices=["InsightFace 106-Point", "MediaPipe FaceMesh (468-Point)"], value="MediaPipe FaceMesh (468-Point)", label="Face Masking/Blending Method")
                    
                    with gr.Group():
                        gr.Markdown("⚙️ **Hardware Device**")
                        device_mode_vid = gr.Radio(choices=["GPU Mode", "CPU Only"], value="GPU Mode" if available_gpus else "CPU Only", label="Processing Device Mode")
                        gpu_choices_vid = gr.CheckboxGroup(choices=available_gpus, value=available_gpus, label="Select GPUs to use", visible=(len(available_gpus) > 0))
                    
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
                        batch_size_vid = gr.Dropdown(choices=["1", "2", "4", "6", "8", "12", "16"], value="4", label="Execution Thread Count (Concurrent Threads)")
                        
                        gr.Markdown("🔧 **Enhancement & Realism**")
                        enhance_vid = gr.Checkbox(label="Enhance Face Details (GFPGAN)", value=False)
                        enhance_strength_vid = gr.Slider(label="Enhance Strength", minimum=0.0, maximum=1.0, value=0.8, step=0.05)
                        handle_occlusions_vid = gr.Checkbox(label="Enable Occlusion Masking (Keep hands/hair/arms in foreground)", value=True)
                        swap_blend_strength_vid = gr.Slider(label="Face Swap Blend Strength (Identity Likeness)", minimum=0.5, maximum=1.0, value=1.0, step=0.05)
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
    # Detect proxy paths for OVHcloud / Jupyter proxy environments
    root_path = ""
    if os.environ.get("JUPYTERHUB_SERVICE_PREFIX"):
        root_path = os.environ.get("JUPYTERHUB_SERVICE_PREFIX").rstrip("/")
    elif os.path.exists("/workspace"):
        root_path = "/proxy/7860"
        
    print(f"[Gradio] Launching server on port 7860 with root_path='{root_path}'")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, root_path=root_path)
