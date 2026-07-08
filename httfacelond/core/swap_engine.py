import os
import cv2
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

# Import modular helper features
from httfacelond.core.pipeline import GPUPipeline, CPUPipeline
from httfacelond.core.detection import detect_faces_with_auto_rotation, rotate_back
from httfacelond.core.color import match_brightness_contrast
from httfacelond.core.enhancer import restore_face_gfpgan
from httfacelond.core.occluder import generate_occlusion_mask
from httfacelond.core.tracker import FaceTracker
from httfacelond.core.fallback_detectors import FallbackDetectors

class SwapEngine:
    def __init__(self):
        self.providers = self._get_execution_providers()
        self.cached_pipelines = {}
        print(f"[Info] SwapEngine initialized with ONNX providers: {self.providers}")
        
        # Initialize FaceAnalysis (buffalo_l model pack with all modules enabled for source face analysis)
        self.app = FaceAnalysis(name='buffalo_l', providers=self.providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize target_app (only detection and landmark_2d_106 allowed to maximize video FPS)
        self.target_app = FaceAnalysis(name='buffalo_l', providers=self.providers, allowed_modules=['detection', 'landmark_2d_106'])
        self.target_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load Swapper model (FP16 version by default)
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        self.inswapper_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
        
        # Fallback to fp32 if fp16 is not present
        if not os.path.exists(self.inswapper_path):
            self.inswapper_path = os.path.join(models_dir, "inswapper_128.onnx")
            
        if not os.path.exists(self.inswapper_path):
            raise FileNotFoundError("No inswapper model found. Please run downloader first.")
            
        print(f"[Info] Loading swapper model: {self.inswapper_path}")
        self.swapper = INSwapper(model_file=self.inswapper_path, session=ort.InferenceSession(self.inswapper_path, providers=self.providers))
        
        # Load GFPGAN model
        self.gfpgan_path = os.path.join(models_dir, "GFPGANv1.4.onnx")
        if os.path.exists(self.gfpgan_path):
            print(f"[Info] Loading GFPGAN model: {self.gfpgan_path}")
            self.gfpgan_session = ort.InferenceSession(self.gfpgan_path, providers=self.providers)
        else:
            print("[Warning] GFPGANv1.4.onnx not found. Detail enhancement will fallback to OpenCV filters.")
            self.gfpgan_session = None

        # Load Face Occluder model
        self.occluder_path = os.path.join(models_dir, "face_occluder.onnx")
        if os.path.exists(self.occluder_path):
            print(f"[Info] Loading Face Occluder model: {self.occluder_path}")
            self.occluder_session = ort.InferenceSession(self.occluder_path, providers=self.providers)
        else:
            print("[Warning] face_occluder.onnx not found. Occlusion handling will be disabled.")
            self.occluder_session = None

        # Load Fallback Detectors (YOLOv11-Face)
        self.yolo_path = os.path.join(models_dir, "yolov11n-face.onnx")
        self.fallback_detectors = FallbackDetectors(yolo_path=self.yolo_path, retina_path=None, providers=self.providers)
        
        # Instantiate a tracker for each video run
        self.tracker = FaceTracker(max_lost_frames=5, smoothing_factor=0.6)

    def set_execution_provider(self, provider_type):
        import re
        m = re.search(r'\d+', str(provider_type))
        gpu_id = int(m.group(0)) if m else 0
        
        if "cpu" in str(provider_type).lower():
            self.providers = ['CPUExecutionProvider']
        else:
            # GPU Mode
            available = self._get_execution_providers()
            self.providers = []
            for p in available:
                p_name = p[0] if isinstance(p, tuple) else p
                p_opts = p[1] if isinstance(p, tuple) and len(p) >= 2 else {}
                
                if p_name.lower() == 'cudaexecutionprovider':
                    opts = dict(p_opts)
                    opts['device_id'] = gpu_id
                    self.providers.append((p_name, opts))
                else:
                    self.providers.append(p)
            
        print(f"[Info] Switching execution providers to: {self.providers}")
        
        # Re-initialize sessions with the new providers (all modules enabled)
        self.app = FaceAnalysis(name='buffalo_l', providers=self.providers)
        self.app.prepare(ctx_id=gpu_id, det_size=(640, 640))
        
        # Also re-initialize target_app
        self.target_app = FaceAnalysis(name='buffalo_l', providers=self.providers, allowed_modules=['detection', 'landmark_2d_106'])
        self.target_app.prepare(ctx_id=gpu_id, det_size=(640, 640))
        
        msg = f"Switched to {provider_type} (Providers prioritized: {self.providers})\n"
        
        if os.path.exists(self.inswapper_path):
            self.swapper = INSwapper(model_file=self.inswapper_path, session=ort.InferenceSession(self.inswapper_path, providers=self.providers))
            active_p = "Unknown"
            if hasattr(self.swapper, 'session') and hasattr(self.swapper.session, 'get_providers'):
                # Check active provider on the model session
                active_p = self.swapper.session.get_providers()[0] if self.swapper.session.get_providers() else "None"
            msg += f"      • Swapper Model Active Provider: {active_p}\n"
            
        if self.gfpgan_path and os.path.exists(self.gfpgan_path):
            self.gfpgan_session = ort.InferenceSession(self.gfpgan_path, providers=self.providers)
            active_p = self.gfpgan_session.get_providers()[0] if self.gfpgan_session.get_providers() else "None"
            msg += f"      • GFPGAN Model Active Provider: {active_p}\n"
            
        if self.occluder_path and os.path.exists(self.occluder_path):
            self.occluder_session = ort.InferenceSession(self.occluder_path, providers=self.providers)
            active_p = self.occluder_session.get_providers()[0] if self.occluder_session.get_providers() else "None"
            msg += f"      • Occluder Model Active Provider: {active_p}\n"
            
        return msg

    def load_swapper(self, filename):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            if self.inswapper_path != path:
                print(f"[Info] Switching swapper model to: {path}")
                self.swapper = INSwapper(model_file=path, session=ort.InferenceSession(path, providers=self.providers))
                self.inswapper_path = path
            return f"Swapper model loaded: {filename}"
        else:
            return f"Error: Swapper model not found at {path}"

    def load_restorer(self, filename):
        if not filename or filename == "None":
            self.gfpgan_session = None
            self.gfpgan_path = None
            return "Restorer disabled."
            
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            if self.gfpgan_path != path:
                print(f"[Info] Switching restorer model to: {path}")
                self.gfpgan_session = ort.InferenceSession(path, providers=self.providers)
                self.gfpgan_path = path
            return f"Restorer model loaded: {filename}"
        else:
            self.gfpgan_session = None
            self.gfpgan_path = None
            return f"Error: Restorer model not found at {path}"

    def get_available_gpus(self):
        gpus = []
        # Method 1: Check torch (most reliable on PyTorch/CUDA environments)
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                for i in range(num_gpus):
                    name = torch.cuda.get_device_name(i)
                    gpus.append(f"GPU {i}: {name}")
                return gpus
        except ImportError:
            pass

        # Method 2: Check nvidia-smi command line (reliable on Linux/cloud containers)
        try:
            import subprocess
            output = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL).decode("utf-8")
            lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
            for line in lines:
                if line.startswith("GPU "):
                    gpus.append(line)
            if gpus:
                return gpus
        except Exception:
            pass

        # Method 3: Fallback check on ONNX Runtime active execution providers
        avail_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in avail_providers:
            # If CUDA is active but torch/nvidia-smi check was blocked, return a default GPU 0
            gpus.append("GPU 0: CUDA Device")
        return gpus

    def get_available_swappers(self):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        if not os.path.exists(models_dir):
            return []
        return [f for f in os.listdir(models_dir) if ("swapper" in f.lower()) and f.endswith(".onnx")]

    def get_available_restorers(self):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        if not os.path.exists(models_dir):
            return ["None"]
        restorers = ["None"]
        for f in os.listdir(models_dir):
            f_lower = f.lower()
            if (f.startswith("GFPGAN") or "gfpgan" in f_lower or f.startswith("codeformer") or "codeformer" in f_lower or f.startswith("GPEN") or "gpen" in f_lower) and f.endswith(".onnx"):
                restorers.append(f)
        return restorers

    def _get_execution_providers(self):
        available = ort.get_available_providers()
        priorities = ['CUDAExecutionProvider', 'ROCMExecutionProvider', 'DirectMLExecutionProvider']
        
        providers = []
        for p in priorities:
            if p in available:
                if p == 'CUDAExecutionProvider':
                    providers.append((p, {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 0,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True
                    }))
                else:
                    providers.append(p)
                    
        providers.append('CPUExecutionProvider')
        return providers

    def get_hardware_status(self):
        is_gpu = len(self.providers) > 1 or self.providers[0] != 'CPUExecutionProvider'
        active = self.providers[0][0] if isinstance(self.providers[0], tuple) else self.providers[0]
        return {
            "is_gpu": is_gpu,
            "active_provider": active,
            "all_providers": [p[0] if isinstance(p, tuple) else p for p in self.providers]
        }

    # Backward compatibility wrappers redirecting to external modules
    def _detect_faces_with_auto_rotation(self, img, det_thresh=0.5, log_callback=None, app_instance=None):
        if app_instance is None:
            app_instance = self.app
        return detect_faces_with_auto_rotation(img, det_thresh, log_callback, app_instance)

    def _rotate_back(self, img, angle):
        return rotate_back(img, angle)

    def _match_brightness_contrast(self, source, target):
        return match_brightness_contrast(source, target)

    def _restore_face_gfpgan(self, face_crop, gfpgan_session=None):
        if gfpgan_session is None:
            gfpgan_session = self.gfpgan_session
        return restore_face_gfpgan(face_crop, gfpgan_session)

    def _generate_occlusion_mask(self, face_crop, occluder_session=None):
        if occluder_session is None:
            occluder_session = self.occluder_session
        return generate_occlusion_mask(face_crop, occluder_session)

    def face_swap(self, source_img, target_img, enhance=True, enhance_strength=0.8, match_color=True, match_scale=False, custom_scale=1.0, det_thresh=0.5, face_upscale_resolution="512", handle_occlusions=True, target_rotation=None, log_callback=None, swap_blend_strength=0.85, match_face_shape=True, pipeline=None, target_faces=None, target_detector="SCRFD (Default)", face_mask_type="InsightFace 106-Point"):
        """
        Swaps face from source_img onto target_img with auto-rotation, upscaling, and occlusion handling.
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
                print(msg)

        # Resolve pipeline-specific sessions or fallback to self
        target_app = pipeline.target_app if pipeline is not None else self.target_app
        swapper = pipeline.swapper if pipeline is not None else self.swapper
        gfpgan_session = pipeline.gfpgan_session if pipeline is not None else self.gfpgan_session
        occluder_session = pipeline.occluder_session if pipeline is not None else self.occluder_session

        # Check if source_img is already a detected Face object or if we need to detect it
        if not hasattr(source_img, 'det_score'):
            # Detect faces in source image (with auto rotation check)
            log("[Process] Running face detection on Source image...")
            source_faces, source_oriented, src_rot = self._detect_faces_with_auto_rotation(source_img, det_thresh=det_thresh, log_callback=log_callback)
            if not source_faces:
                log("[Warning] No face detected in source image (even with rotation).")
                return target_img
            
            # Use the first face detected in the source image
            source_face = source_faces[0]
            log(f"[Process] Detected source face with confidence: {source_face.det_score:.2f}")
        else:
            source_face = source_img
            source_oriented = None # Not needed since face object is pre-analyzed
        
        # Detect faces in target image (with auto rotation check bypassed if target_rotation is provided)
        tgt_rot = 0
        target_oriented = target_img.copy() if hasattr(target_img, 'copy') else target_img
        
        if target_faces is None:
            if target_detector == "YOLOv11-Face" and hasattr(self, 'fallback_detectors') and self.fallback_detectors.yolo_session is not None:
                log("[Process] Running YOLOv11-Face detector with auto-rotation...")
                if target_rotation is not None:
                    cached_angle = target_rotation.get('angle', 0) if isinstance(target_rotation, dict) else target_rotation
                    
                    if cached_angle == 90:
                        target_oriented = cv2.rotate(target_img, cv2.ROTATE_90_CLOCKWISE)
                    elif cached_angle == 180:
                        target_oriented = cv2.rotate(target_img, cv2.ROTATE_180)
                    elif cached_angle == 270:
                        target_oriented = cv2.rotate(target_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        target_oriented = target_img.copy() if hasattr(target_img, 'copy') else target_img
                        
                    target_faces = self.fallback_detectors.detect_yolo(target_oriented, thresh=max(0.15, det_thresh - 0.2))
                    tgt_rot = cached_angle
                    
                    if not target_faces:
                        log(f"[YOLO Process] No face found at cached angle {cached_angle}°. Sweeping other angles...")
                        target_faces, target_oriented, fallback_rot = self.fallback_detectors.detect_yolo_with_auto_rotation(target_img, thresh=max(0.15, det_thresh - 0.2), log_callback=log_callback)
                        tgt_rot = fallback_rot
                        if isinstance(target_rotation, dict):
                            target_rotation['angle'] = fallback_rot
                else:
                    target_faces, target_oriented, tgt_rot = self.fallback_detectors.detect_yolo_with_auto_rotation(target_img, thresh=max(0.15, det_thresh - 0.2), log_callback=log_callback)
            else:
                # SCRFD (Default) pipeline
                if target_rotation is not None:
                    cached_angle = target_rotation.get('angle', 0) if isinstance(target_rotation, dict) else target_rotation
                    
                    # Try cached angle first
                    if cached_angle == 90:
                        target_oriented = cv2.rotate(target_img, cv2.ROTATE_90_CLOCKWISE)
                    elif cached_angle == 180:
                        target_oriented = cv2.rotate(target_img, cv2.ROTATE_180)
                    elif cached_angle == 270:
                        target_oriented = cv2.rotate(target_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        target_oriented = target_img.copy() if hasattr(target_img, 'copy') else target_img
                        
                    if hasattr(target_app, 'det_model'):
                        target_app.det_model.det_thresh = det_thresh
                    target_faces = target_app.get(target_oriented)
                    tgt_rot = cached_angle
                    
                    # Smart fallback: if no face detected at cached angle, try a lower threshold first on the same angle to prevent flickering
                    if not target_faces:
                        for temp_thresh in [max(0.15, det_thresh - 0.15), 0.15]:
                            if hasattr(target_app, 'det_model'):
                                target_app.det_model.det_thresh = temp_thresh
                            target_faces = target_app.get(target_oriented)
                            if target_faces:
                                break
                    
                    # If still no face, do a full rotation sweep
                    if not target_faces:
                        log(f"[Process] No face found at cached angle {cached_angle}°. Scanning other orientations...")
                        target_faces, target_oriented, fallback_rot = self._detect_faces_with_auto_rotation(target_img, det_thresh=det_thresh, log_callback=log_callback, app_instance=target_app)
                        tgt_rot = fallback_rot
                        if isinstance(target_rotation, dict):
                            target_rotation['angle'] = fallback_rot
                            log(f"[Process] Cache updated: new angle is {fallback_rot}°")
                else:
                    log("[Process] Running face detection on Target image...")
                    target_faces, target_oriented, tgt_rot = self._detect_faces_with_auto_rotation(target_img, det_thresh=det_thresh, log_callback=log_callback, app_instance=target_app)
            
            # Global Fallback if chosen detector still finds nothing
            if not target_faces and target_detector != "YOLOv11-Face" and hasattr(self, 'fallback_detectors') and self.fallback_detectors.yolo_session is not None:
                log("[Process] SCRFD failed. Triggering YOLOv11-Face fallback detector...")
                target_faces, target_oriented, fallback_rot = self.fallback_detectors.detect_yolo_with_auto_rotation(target_img, thresh=max(0.15, det_thresh - 0.2), log_callback=log_callback)
                tgt_rot = fallback_rot
                if target_faces:
                    log(f"[Process] YOLOv11-Face successfully recovered {len(target_faces)} face(s)!")

        if not target_faces:
            log(f"[Warning] No face detected in target image above threshold fallback.")
            return target_img
            
        # Ensure all faces have 106 landmarks extracted if using standard InsightFace mask
        if face_mask_type == "InsightFace 106-Point":
            for f in target_faces:
                if getattr(f, 'landmark_2d_106', None) is None:
                    # Align local crop and extract landmarks using target_app
                    try:
                        if hasattr(target_app, 'models') and 'landmark_2d_106' in target_app.models:
                            lm_model = target_app.models['landmark_2d_106']
                            f.landmark_2d_106 = lm_model.get(target_oriented, f)
                    except Exception as e:
                        log(f"[Warning] Failed to generate 106 landmarks for fallback detection: {e}")
        
        log(f"[Process] Found {len(target_faces)} valid target face(s) for swap.")
        result = target_oriented.copy() if hasattr(target_oriented, 'copy') else target_oriented
        for idx, face in enumerate(target_faces):
            log(f"[Face {idx+1}] Swapping with score: {face.det_score:.2f}...")
            # Calculate scale factor
            scale_factor = custom_scale
            if match_scale:
                src_dist = np.linalg.norm(source_face.kps[0] - source_face.kps[1])
                tgt_dist = np.linalg.norm(face.kps[0] - face.kps[1])
                if tgt_dist > 0:
                    scale_factor = (src_dist / tgt_dist) * custom_scale
            
            # Clip scale factor to [0.5, 2.0] for safety
            scale_factor = np.clip(scale_factor, 0.5, 2.0)
            
            # Perform face swapping without built-in paste-back to allow custom scaling
            bgr_fake, M = swapper.get(target_oriented.copy(), face, source_face, paste_back=False)
            
            # Get original target face aligned crop
            aimg, _ = insightface.utils.face_align.norm_crop2(target_oriented, face.kps, 128)
            
            # Scale aligned face crops around their center (64, 64) based on calculated scale_factor
            if abs(scale_factor - 1.0) > 1e-5:
                T_scale = cv2.getRotationMatrix2D((64, 64), 0, scale_factor)
                bgr_fake = cv2.warpAffine(bgr_fake, T_scale, (128, 128), borderMode=cv2.BORDER_REPLICATE)
                aimg = cv2.warpAffine(aimg, T_scale, (128, 128), borderMode=cv2.BORDER_REPLICATE)
            
            # Create a difference-based mask to isolate the swapped face features
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2, :] = 0
            fake_diff[-2:, :] = 0
            fake_diff[:, :2] = 0
            fake_diff[:, -2:] = 0
            
            # Warp the fake crop and difference mask back to target coordinates
            IM = cv2.invertAffineTransform(M)
            target_h, target_w = target_oriented.shape[:2]
            
            bgr_fake_warped = cv2.warpAffine(bgr_fake, IM, (target_w, target_h), borderValue=0.0)
            img_white = np.full((128, 128), 255, dtype=np.float32)
            img_white = cv2.warpAffine(img_white, IM, (target_w, target_h), borderValue=0.0)
            fake_diff = cv2.warpAffine(fake_diff, IM, (target_w, target_h), borderValue=0.0)
            
            img_white[img_white > 20] = 255
            fthresh = 10
            fake_diff[fake_diff < fthresh] = 0
            fake_diff[fake_diff >= fthresh] = 255
            img_mask = img_white
            
            # Erode and blur mask to make borders seamless
            mask_h_inds, mask_w_inds = np.where(img_mask == 255)
            if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:
                mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
                mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
                mask_size = int(np.sqrt(mask_h * mask_w))
                k = max(mask_size // 10, 10)
                kernel = np.ones((k, k), np.uint8)
                img_mask = cv2.erode(img_mask, kernel, iterations=1)
                
                k_blur = max(mask_size // 20, 5)
                img_mask = cv2.GaussianBlur(img_mask, (2 * k_blur + 1, 2 * k_blur + 1), 0)
                
            img_mask /= 255.0
            img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
            
            swapped = img_mask * bgr_fake_warped + (1.0 - img_mask) * target_oriented.astype(np.float32)
            swapped = np.clip(swapped, 0, 255).astype(np.uint8)
            
            # Bounding box bounds check
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            h, w, _ = target_oriented.shape
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 > x1 and y2 > y1:
                swapped_face_crop = swapped[y1:y2, x1:x2].copy()
                target_face_crop = target_oriented[y1:y2, x1:x2].copy()
                
                processed_crop = swapped_face_crop
                
                # Apply Dynamic Resolution Upscaling for enhancement (only if enhance is enabled)
                target_res = int(face_upscale_resolution)
                if enhance and target_res > 128:
                    processed_crop = cv2.resize(processed_crop, (target_res, target_res), interpolation=cv2.INTER_CUBIC)
                    target_face_crop_scaled = cv2.resize(target_face_crop, (target_res, target_res), interpolation=cv2.INTER_CUBIC)
                else:
                    target_face_crop_scaled = target_face_crop
                
                # 1. Match lighting/color to target face
                if match_color:
                    log(f"[Face {idx+1}] Matching lighting & skin tone (Reinhard Color Transfer)...")
                    processed_crop = self._match_brightness_contrast(processed_crop, target_face_crop_scaled)
                    
                # 2. Restoring face using GFPGAN for maximum sharpness
                if enhance:
                    log(f"[Face {idx+1}] Restoring facial details using GFPGAN (Strength: {enhance_strength})...")
                    restored_crop = self._restore_face_gfpgan(processed_crop, gfpgan_session=gfpgan_session)
                    # Blend based on enhance strength
                    processed_crop = cv2.addWeighted(restored_crop, enhance_strength, processed_crop, 1.0 - enhance_strength, 0)
                
                # 3. Blend swapped face with original target face to maximize identity similarity
                if swap_blend_strength < 1.0:
                    if target_face_crop_scaled.shape != processed_crop.shape:
                        target_face_crop_scaled = cv2.resize(target_face_crop_scaled, (processed_crop.shape[1], processed_crop.shape[0]), interpolation=cv2.INTER_CUBIC)
                    processed_crop = cv2.addWeighted(processed_crop, swap_blend_strength, target_face_crop_scaled, 1.0 - swap_blend_strength, 0)

                # Resize back to bounding box dimensions for blending (only if shape differs)
                if processed_crop.shape[0] != (y2 - y1) or processed_crop.shape[1] != (x2 - x1):
                    processed_crop = cv2.resize(processed_crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
                
                # 3. Handle Scaling & Blending
                # We use the mean of the 106 landmarks for a rock-solid center coordinates, 
                # completely avoiding bounding box jitter shaking!
                if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    cx = int(np.mean(face.landmark_2d_106[:, 0]))
                    cy = int(np.mean(face.landmark_2d_106[:, 1]))
                else:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                
                # Generate dynamic landmark convex hull mask for target face
                mask = np.zeros(swapped_face_crop.shape[:2], dtype=np.float32)
                
                # Option A: Google MediaPipe FaceMesh (468 points)
                if face_mask_type == "MediaPipe FaceMesh (468-Point)":
                    try:
                        import mediapipe as mp
                        mp_face_mesh = mp.solutions.face_mesh
                        # Use aligned face crop for reliable detection (same approach as V2)
                        # Generate a clean aligned crop at a good resolution for MediaPipe
                        aimg_for_mesh, _ = insightface.utils.face_align.norm_crop2(target_oriented, face.kps, 256)
                        with mp_face_mesh.FaceMesh(
                            static_image_mode=True,
                            max_num_faces=1,
                            refine_landmarks=False,
                            min_detection_confidence=0.3
                        ) as face_mesh:
                            rgb_crop = cv2.cvtColor(aimg_for_mesh, cv2.COLOR_BGR2RGB)
                            results_mesh = face_mesh.process(rgb_crop)
                            if results_mesh.multi_face_landmarks:
                                log(f"[Face {idx+1}] Generating Google MediaPipe 468-point face mesh mask...")
                                # Get the alignment matrix used for the crop
                                M_mesh = insightface.utils.face_align.estimate_norm(face.kps, 256)
                                IM_mesh = cv2.invertAffineTransform(M_mesh)
                                
                                # Convert mesh landmarks back to target image coordinates
                                mesh_pts_aligned = []
                                for lm in results_mesh.multi_face_landmarks[0].landmark:
                                    mesh_pts_aligned.append([lm.x * 256, lm.y * 256])
                                mesh_pts_aligned = np.array(mesh_pts_aligned, dtype=np.float32)
                                
                                # Warp back to target image coordinates
                                mesh_pts_target = cv2.transform(mesh_pts_aligned[np.newaxis], IM_mesh)[0]
                                
                                # Offset to local bounding box crop coordinates
                                mesh_pts_local = mesh_pts_target.copy()
                                mesh_pts_local[:, 0] -= x1
                                mesh_pts_local[:, 1] -= y1
                                
                                hull = cv2.convexHull(mesh_pts_local.astype(np.int32))
                                cv2.fillConvexPoly(mask, hull, 255)
                            else:
                                raise ValueError("No mesh detected in local crop")
                    except Exception as e:
                        log(f"[Warning] MediaPipe FaceMesh failed: {e}. Falling back to 106-point landmark mask...")
                        face_mask_type = "InsightFace 106-Point"
                
                # Option B: InsightFace 106-Point Mask (Convex Hull + forehead vertical gradient)
                if face_mask_type == "InsightFace 106-Point" and hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    log(f"[Face {idx+1}] Generating high-precision 106-point landmark mask...")
                    # Offset the landmarks to fit the local face crop
                    local_landmarks = face.landmark_2d_106.copy()
                    local_landmarks[:, 0] -= x1
                    local_landmarks[:, 1] -= y1
                    
                    # Calculate landmark height to estimate forehead height
                    lm_x1 = np.min(local_landmarks[:, 0])
                    lm_x2 = np.max(local_landmarks[:, 0])
                    lm_y1 = np.min(local_landmarks[:, 1])
                    lm_y2 = np.max(local_landmarks[:, 1])
                    lm_h = lm_y2 - lm_y1
                    
                    # Estimate top of forehead (approx 50% of face height above eyebrows to cover original eyebrows)
                    y_forehead = max(0, lm_y1 - 0.50 * lm_h)
                    
                    # Add 3 forehead points to extend the convex hull to the forehead
                    forehead_pts = np.array([
                        [lm_x1, y_forehead],
                        [(lm_x1 + lm_x2) / 2.0, y_forehead],
                        [lm_x2, y_forehead]
                    ], dtype=np.float32)
                    local_landmarks_extended = np.vstack([local_landmarks, forehead_pts])
                    
                    hull = cv2.convexHull(local_landmarks_extended.astype(np.int32))
                    cv2.fillConvexPoly(mask, hull, 255)
                    
                    # Apply a linear vertical gradient to fade out the forehead region smoothly
                    # from lm_y1 (eyebrows) to y_forehead (top of forehead). This prevents any sharp cutoffs/seams!
                    y_start = int(y_forehead)
                    y_end = int(lm_y1)
                    if y_end > y_start:
                        for y_idx in range(y_start, y_end):
                            weight = (y_idx - y_start) / (y_end - y_start)
                            mask[y_idx, :] = mask[y_idx, :] * weight
                
                # Option C: Fallback Bounding Box Mask (if neither is available)
                if np.sum(mask) == 0:
                    log(f"[Face {idx+1}] Generating fallback bounding box mask...")
                    center = (swapped_face_crop.shape[1] // 2, swapped_face_crop.shape[0] // 2)
                    axes = (swapped_face_crop.shape[1] // 2, swapped_face_crop.shape[0] // 2)
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                
                # Dynamic feathering based on face size to ensure 100% invisible seams at any scale
                face_h, face_w = swapped_face_crop.shape[:2]
                blur_size = int(max(face_h, face_w) * 0.15)
                blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size
                blur_size = max(5, blur_size)
                blur_sigma = blur_size / 2.0
                
                # Feather mask boundaries dynamically
                mask = cv2.GaussianBlur(mask, (blur_size, blur_size), blur_sigma) / 255.0
                mask = np.expand_dims(mask, axis=2)
                
                # 4. Handle Occlusions (hands, hair, objects blocking the face)
                if handle_occlusions and occluder_session is not None:
                    log(f"[Face {idx+1}] Running occlusion detection...")
                    try:
                        # Align the target face to 256x256 using its 5 keypoints (required by the occluder model)
                        M_align = insightface.utils.face_align.estimate_norm(face.kps, image_size=256)
                        aligned_face = cv2.warpAffine(target_oriented, M_align, (256, 256))
                        
                        # Generate occlusion mask on the aligned face
                        occlusion_mask_256 = self._generate_occlusion_mask(aligned_face, occluder_session=occluder_session)
                        
                        if occlusion_mask_256 is not None:
                            # Warp the 256x256 mask back to the target image coordinate space
                            M_inv = cv2.invertAffineTransform(M_align)
                            h_tgt, w_tgt = target_oriented.shape[:2]
                            occlusion_mask_target = cv2.warpAffine(
                                occlusion_mask_256[:, :, 0], 
                                M_inv, 
                                (w_tgt, h_tgt), 
                                flags=cv2.INTER_LINEAR, 
                                borderValue=0.0
                            )
                            # Crop the global warped mask to the local bounding box of the face
                            local_occlusion_mask = occlusion_mask_target[y1:y2, x1:x2]
                            local_occlusion_mask = np.expand_dims(local_occlusion_mask, axis=2)
                            
                            log(f"[Face {idx+1}] Masking out occluded elements (hands/hair/arms)...")
                            mask = mask * local_occlusion_mask
                    except Exception as e:
                        log(f"[Warning] Occlusion handling alignment or warp failed: {e}")
                
                # Calculate rotation-invariant face width/height ratios using eye-mouth keypoints.
                # This prevents tilted head orientations from inflating the width ratio.
                aspect_warp = 1.0
                if match_face_shape:
                    src_eye_dist = np.linalg.norm(source_face.kps[0] - source_face.kps[1])
                    src_eye_mid = (source_face.kps[0] + source_face.kps[1]) / 2.0
                    src_mouth_mid = (source_face.kps[3] + source_face.kps[4]) / 2.0
                    src_height = np.linalg.norm(src_eye_mid - src_mouth_mid)
                    
                    tgt_eye_dist = np.linalg.norm(face.kps[0] - face.kps[1])
                    tgt_eye_mid = (face.kps[0] + face.kps[1]) / 2.0
                    tgt_mouth_mid = (face.kps[3] + face.kps[4]) / 2.0
                    tgt_height = np.linalg.norm(tgt_eye_mid - tgt_mouth_mid)
                    
                    if src_height > 0 and tgt_height > 0 and tgt_eye_dist > 0:
                        src_ratio = src_eye_dist / src_height
                        tgt_ratio = tgt_eye_dist / tgt_height
                        aspect_warp = src_ratio / tgt_ratio
                        # Clip to safe boundaries [0.8, 1.2]
                        aspect_warp = np.clip(aspect_warp, 0.8, 1.2)

                # Always keep scale_w and scale_h at 1.0. 
                # This prevents background stretching (sharp vertical edges) and hairline misalignment (long foreheads).
                scale_w = 1.0
                scale_h = 1.0

                # Paste directly onto result (pristine target_oriented copy) to completely avoid double-pasting or ghosting.
                if abs(scale_w - 1.0) > 0.01 or abs(scale_h - 1.0) > 0.01:
                    new_w = int((x2 - x1) * scale_w)
                    new_h = int((y2 - y1) * scale_h)
                    if new_w > 5 and new_h > 5:
                        processed_crop = cv2.resize(processed_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        
                        # Resize the high-precision landmark mask instead of falling back to a crude ellipse!
                        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        if len(resized_mask.shape) == 2:
                            resized_mask = np.expand_dims(resized_mask, axis=2)
                        
                        # Coordinates on target image
                        tx1 = cx - new_w // 2
                        ty1 = cy - new_h // 2
                        tx2 = tx1 + new_w
                        ty2 = ty1 + new_h
                        
                        rx1 = max(0, -tx1)
                        ry1 = max(0, -ty1)
                        rx2 = new_w - max(0, tx2 - w)
                        ry2 = new_h - max(0, ty2 - h)
                        
                        bx1 = max(0, tx1)
                        by1 = max(0, ty1)
                        bx2 = min(w, tx2)
                        by2 = min(h, ty2)
                        
                        if (bx2 > bx1) and (by2 > by1) and (rx2 > rx1) and (ry2 > ry1):
                            cropped_resized = processed_crop[ry1:ry2, rx1:rx2]
                            cropped_mask = resized_mask[ry1:ry2, rx1:rx2]
                            
                            # Blend with original target frame pixels to restore hair/occlusion perfectly
                            target_crop_resized = target_oriented[by1:by2, bx1:bx2]
                            result[by1:by2, bx1:bx2] = (cropped_resized * cropped_mask + target_crop_resized * (1.0 - cropped_mask)).astype(np.uint8)
                else:
                    # Blend back processed face crop in original bbox using the convex hull mask.
                    # We blend with target_face_crop (original face) to correctly restore occluded elements (like hair/hands).
                    result[y1:y2, x1:x2] = (processed_crop * mask + target_face_crop * (1.0 - mask)).astype(np.uint8)
            
        # Rotate result back to original orientation if target was auto-rotated
        if tgt_rot != 0:
            log(f"[Process] Rotating output back by {-tgt_rot}° to original orientation...")
            result = self._rotate_back(result, tgt_rot)
            
        return result

    def process_video(self, source_img_path, target_video_path, output_path, enhance=True, enhance_strength=0.8, match_color=True, match_scale=False, custom_scale=1.0, det_thresh=0.5, face_upscale_resolution="512", handle_occlusions=True, frame_step=1, det_size=640, batch_size=1, progress_callback=None, log_callback=None, swap_blend_strength=0.85, match_face_shape=True, selected_gpus=None, target_detector="SCRFD (Default)", face_mask_type="InsightFace 106-Point"):
        import time
        import tempfile
        import shutil
        import subprocess
        import os
        import queue
        import threading
        
        # Programmatically detect and use static, self-contained FFmpeg binary from imageio_ffmpeg if available
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"[Info] Using static FFmpeg binary from imageio-ffmpeg: {ffmpeg_bin}")
        except ImportError:
            ffmpeg_bin = 'ffmpeg'
            print("[Warning] imageio-ffmpeg not found. Falling back to system 'ffmpeg' command.")
        
        def log(msg):
            if log_callback:
                log_callback(msg)
            print(msg)
            
        def run_ffmpeg_checked(cmd_args, desc="FFmpeg"):
            log(f"[Video Process] Executing {desc} command...")
            result = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                error_msg = f"{desc} failed (Code: {result.returncode}).\nFFmpeg Stderr Output:\n{result.stderr}"
                log(f"[Error] {error_msg}")
                raise RuntimeError(error_msg)
            log(f"[Video Process] {desc} executed successfully.")

        # Parse selected_gpus to get numeric GPU IDs
        gpu_ids = []
        if selected_gpus:
            for g in selected_gpus:
                import re
                m = re.search(r'\d+', str(g))
                if m:
                    gpu_ids.append(int(m.group(0)))
                    
        pipelines = []
        if not gpu_ids:
            # CPU Only
            cache_key = "cpu"
            if cache_key not in self.cached_pipelines:
                log("[Video Process] Initializing CPU Pipeline session...")
                self.cached_pipelines[cache_key] = CPUPipeline(self.inswapper_path, self.gfpgan_path, self.occluder_path)
            pipelines.append(self.cached_pipelines[cache_key])
        else:
            # GPU Mode
            for gpu_id in gpu_ids:
                cache_key = f"gpu_{gpu_id}"
                need_init = True
                if cache_key in self.cached_pipelines:
                    pipeline = self.cached_pipelines[cache_key]
                    if getattr(pipeline, 'inswapper_path', '') == self.inswapper_path:
                        need_init = False
                
                if need_init:
                    log(f"[Video Process] Initializing GPU Pipeline session for GPU {gpu_id}...")
                    pipeline = GPUPipeline(gpu_id, self.providers, self.inswapper_path, self.gfpgan_path, self.occluder_path)
                    pipeline.inswapper_path = self.inswapper_path
                    self.cached_pipelines[cache_key] = pipeline
                pipelines.append(self.cached_pipelines[cache_key])

        log("[Video Process] Loading source face image...")
        source_img = cv2.imread(source_img_path)
        if source_img is None:
            raise ValueError("Could not read source image.")
            
        log("[Video Process] Pre-detecting source face (only once)...")
        source_faces, _, _ = self._detect_faces_with_auto_rotation(source_img, det_thresh=det_thresh, log_callback=log_callback)
        if not source_faces:
            raise ValueError("No face detected in source image.")
        source_face = source_faces[0]
        log(f"[Video Process] Source face successfully extracted (score: {source_face.det_score:.2f})")
            
        # Configure target detector scan size on all pipelines
        for pipeline in pipelines:
            log(f"[Video Process] Setting face detector scan size on GPU {pipeline.gpu_id} to: {det_size}x{det_size}")
            pipeline.target_app.prepare(ctx_id=max(0, pipeline.gpu_id), det_size=(det_size, det_size))

        # Open video to get metadata
        cap = cv2.VideoCapture(target_video_path)
        if not cap.isOpened():
            raise ValueError("Could not open target video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure dimensions are even to prevent encoding failures with codecs like H.264
        width = (width // 2) * 2
        height = (height // 2) * 2
        
        log(f"[Video Process] Video metadata (adjusted to even dimensions): {width}x{height} @ {fps:.2f} fps ({total_frames} total frames)")

        # Detect target video face rotation once on the first frame using the first pipeline
        target_rotation = {'angle': 0}
        ret, first_frame = cap.read()
        if ret and first_frame is not None:
            log("[Video Process] Auto-detecting rotation on the first frame...")
            faces, _, rot_angle = self._detect_faces_with_auto_rotation(first_frame, det_thresh=det_thresh, log_callback=log_callback, app_instance=pipelines[0].target_app)
            if faces:
                target_rotation['angle'] = rot_angle
                log(f"[Video Process] Locked initial target video face rotation: {target_rotation['angle']}°")
        cap.release()

        start_time = time.time()
        
        # Determine number of concurrent threads (matching FaceFusion's thread execution model)
        num_workers = max(batch_size, len(pipelines))
        log(f"[Video Process] Initializing Threaded Queue Pipeline with {num_workers} parallel workers across {len(pipelines)} pipelines...")

        # Setup bounded queues to keep memory extremely low (constant ~100MB-300MB RAM)
        input_queue = queue.Queue(maxsize=32)
        output_queue = queue.Queue(maxsize=32)

        # Worker thread loop
        # Reset tracker state before starting the video run
        if hasattr(self, 'tracker'):
            self.tracker.reset()

        # Worker thread loop
        def thread_worker(pipeline):
            while True:
                try:
                    task = input_queue.get()
                except Exception:
                    break
                if task is None:
                    break
                    
                f_idx, frame_img, target_faces = task
                if frame_img is None or source_face is None or not target_faces:
                    output_queue.put((f_idx, frame_img))
                    continue
                    
                try:
                    # Perform face swap on the frame (runs concurrently on C++ ONNX threads, releasing Python GIL!)
                    swapped = self.face_swap(
                        source_face,
                        frame_img,
                        enhance=enhance,
                        enhance_strength=enhance_strength,
                        match_color=match_color,
                        match_scale=match_scale,
                        custom_scale=custom_scale,
                        det_thresh=det_thresh,
                        face_upscale_resolution=face_upscale_resolution,
                        handle_occlusions=handle_occlusions,
                        target_rotation=target_rotation,
                        swap_blend_strength=swap_blend_strength,
                        match_face_shape=match_face_shape,
                        pipeline=pipeline,
                        target_faces=target_faces,
                        target_detector=target_detector,
                        face_mask_type=face_mask_type
                    )
                    output_queue.put((f_idx, swapped))
                except Exception as e:
                    # Retry with a 2-pixel border shift to force a different face crop dimension, bypassing cuDNN graph compiler bugs!
                    print(f"[Warning] Frame {f_idx} failed on GPU: {e}. Retrying with shape shift...")
                    try:
                        padded_img = cv2.copyMakeBorder(frame_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0,0,0])
                        swapped_padded = self.face_swap(
                            source_face,
                            padded_img,
                            enhance=enhance,
                            enhance_strength=enhance_strength,
                            match_color=match_color,
                            match_scale=match_scale,
                            custom_scale=custom_scale,
                            det_thresh=det_thresh,
                            face_upscale_resolution=face_upscale_resolution,
                            handle_occlusions=handle_occlusions,
                            target_rotation=target_rotation,
                            swap_blend_strength=swap_blend_strength,
                            match_face_shape=match_face_shape,
                            pipeline=pipeline,
                            target_faces=target_faces,
                            target_detector=target_detector,
                            face_mask_type=face_mask_type
                        )
                        swapped = swapped_padded[2:-2, 2:-2]
                        output_queue.put((f_idx, swapped))
                        print(f"[Success] Frame {f_idx} successfully recovered on retry!")
                    except Exception as retry_err:
                        print(f"[Worker Error] Frame {f_idx} failed retry: {retry_err}")
                        output_queue.put((f_idx, frame_img))

        # Start worker threads
        workers = []
        for i in range(num_workers):
            pipeline = pipelines[i % len(pipelines)]
            t = threading.Thread(target=thread_worker, args=(pipeline,))
            t.start()
            workers.append(t)

        # Start background thread to read video frames and feed input_queue
        def video_reader_thread():
            cap_read = cv2.VideoCapture(target_video_path)
            f_idx = 0
            
            # Setup a local detection app instance for reader-side detection
            reader_detector = pipelines[0].target_app
            
            while True:
                ret, frame = cap_read.read()
                if not ret or frame is None:
                    break
                
                # Direct bypass to output for skipped frames (frame_step optimization)
                if f_idx > 0 and frame_step > 1 and (f_idx % frame_step != 0):
                    output_queue.put((f_idx, None))
                else:
                    # Run face detection on the reader thread (thread-safe and sequential)
                    target_oriented = frame.copy()
                    cached_angle = target_rotation.get('angle', 0) if isinstance(target_rotation, dict) else target_rotation
                    
                    if cached_angle == 90:
                        target_oriented = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif cached_angle == 180:
                        target_oriented = cv2.rotate(frame, cv2.ROTATE_180)
                    elif cached_angle == 270:
                        target_oriented = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        
                    if hasattr(reader_detector, 'det_model'):
                        reader_detector.det_model.det_thresh = det_thresh
                        
                    faces = reader_detector.get(target_oriented)
                    
                    if target_detector == "YOLOv11-Face" and hasattr(self, 'fallback_detectors') and self.fallback_detectors.yolo_session is not None:
                        faces = self.fallback_detectors.detect_yolo(target_oriented, thresh=max(0.15, det_thresh - 0.2))
                    else:
                        if hasattr(reader_detector, 'det_model'):
                            reader_detector.det_model.det_thresh = det_thresh
                            
                        faces = reader_detector.get(target_oriented)
                        
                        # Fallback to lower thresholds
                        if not faces:
                            for temp_thresh in [max(0.15, det_thresh - 0.15), 0.15]:
                                if hasattr(reader_detector, 'det_model'):
                                    reader_detector.det_model.det_thresh = temp_thresh
                                faces = reader_detector.get(target_oriented)
                                if faces:
                                    break
                                    
                        # Fallback to YOLOv11-Face if SCRFD completely fails
                        if not faces and hasattr(self, 'fallback_detectors') and self.fallback_detectors.yolo_session is not None:
                            faces = self.fallback_detectors.detect_yolo(target_oriented, thresh=max(0.15, det_thresh - 0.2))
                    
                    # Update face tracker
                    tracked_faces = []
                    if faces:
                        # Feed the first detected face to the tracker
                        tracked_face = self.tracker.update(faces[0])
                        if tracked_face is not None:
                            tracked_faces.append(tracked_face)
                            # Keep other detected faces if multi-face (currently single tracking optimized)
                            if len(faces) > 1:
                                tracked_faces.extend(faces[1:])
                    else:
                        # Try temporal prediction
                        predicted_face = self.tracker.update(None)
                        if predicted_face is not None:
                            tracked_faces.append(predicted_face)
                    
                    input_queue.put((f_idx, frame, tracked_faces))
                f_idx += 1
                
            cap_read.release()
            # Send sentinel value to terminate workers
            for _ in range(num_workers):
                input_queue.put(None)

        # Start the video reader thread
        reader = threading.Thread(target=video_reader_thread)
        reader.start()

        # Step 3: Reassemble video on-the-fly (Producer-Consumer using FFmpeg pipe to preserve pristine quality!)
        temp_dir = tempfile.mkdtemp(prefix="face_swap_pc_")
        temp_video_path = os.path.join(temp_dir, "temp_output.mp4")
        
        # Start FFmpeg subprocess to receive raw video frames directly from Python memory.
        cmd_ffmpeg = [
            ffmpeg_bin, '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f"{width}x{height}",
            '-pix_fmt', 'bgr24',
            '-r', f"{fps}",
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '16',
            '-preset', 'veryfast',
            temp_video_path
        ]
        
        ffmpeg_process = subprocess.Popen(cmd_ffmpeg, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        try:
            next_frame_to_write = 0
            completed_frames = {}
            last_swapped_frame = None
            
            while next_frame_to_write < total_frames:
                f_idx, swapped_frame = output_queue.get()
                completed_frames[f_idx] = swapped_frame
                
                # Write all consecutive frames that are ready
                while next_frame_to_write in completed_frames:
                    frame_data = completed_frames.pop(next_frame_to_write)
                    
                    if frame_data is None: # Skipped frame
                        if last_swapped_frame is not None:
                            ffmpeg_process.stdin.write(last_swapped_frame.tobytes())
                    else: # Swapped frame
                        if frame_data.shape[1] != width or frame_data.shape[0] != height:
                            frame_data = cv2.resize(frame_data, (width, height))
                        ffmpeg_process.stdin.write(frame_data.tobytes())
                        last_swapped_frame = frame_data
                        
                    next_frame_to_write += 1
                    
                    # Yield progress
                    elapsed = time.time() - start_time
                    speed_fps = next_frame_to_write / elapsed if elapsed > 0 else 0
                    remaining = total_frames - next_frame_to_write
                    eta = remaining / speed_fps if speed_fps > 0 else 0
                    
                    yield next_frame_to_write, total_frames, speed_fps, elapsed, eta
                    if progress_callback:
                        progress_callback(min(next_frame_to_write / total_frames, 1.0))
                        
            # Close FFmpeg pipe to finish the temporary video file
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            
            # Wait for workers and reader thread to join
            reader.join()
            for t in workers:
                t.join()
                
            # Step 4: Merge audio
            log("[Video Process] Step 3/3: Merging audio track with FFmpeg...")
            cmd_merge = [
                ffmpeg_bin, '-y',
                '-i', temp_video_path,
                '-i', target_video_path,
                '-map', '0:v',
                '-map', '1:a?',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-shortest',
                output_path
            ]
            run_ffmpeg_checked(cmd_merge, "Audio merge")
            yield "COMPLETED", output_path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            log("[Video Process] Temporary files cleaned successfully.")

    def extract_frame(self, video_path, frame_index):
        """
        Extracts a single frame from the video at frame_index (0-indexed).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0 or frame_index >= total_frames:
            cap.release()
            return None
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def unload_models(self):
        """
        Unloads all models from memory and triggers garbage collection to free GPU VRAM.
        """
        print("[Info] Unloading all AI models to free GPU VRAM...")
        self.swapper = None
        self.gfpgan_session = None
        self.occluder_session = None
        self.app = None
        
        import gc
        gc.collect()
        
        # Free libc memory cache on Linux
        import sys
        if not sys.platform.startswith('win'):
            try:
                import ctypes
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)
            except Exception:
                pass
        return "🧹 [System] GPU VRAM successfully released and cleared!"
