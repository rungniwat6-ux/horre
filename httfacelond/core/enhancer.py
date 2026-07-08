import cv2
import numpy as np

def restore_face_gfpgan(face_crop, gfpgan_session=None):
    """
    Uses GFPGAN ONNX model to restore and sharpen facial details.
    """
    if gfpgan_session is None:
        # Fallback to OpenCV sharpening if GFPGAN is missing
        gaussian = cv2.GaussianBlur(face_crop, (0, 0), 3.0)
        return cv2.addWeighted(face_crop, 1.6, gaussian, -0.6, 0)
        
    h, w, _ = face_crop.shape
    if h < 10 or w < 10:
        return face_crop
        
    # 1. Preprocess: Resize to 512x512, BGR to RGB, normalize to [-1, 1]
    img_512 = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img_512, cv2.COLOR_BGR2RGB)
    
    img_normalized = (img_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    
    # HWC to CHW and add batch dimension
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # 2. Run ONNX session
    try:
        inputs = {gfpgan_session.get_inputs()[0].name: img_tensor}
        outputs = gfpgan_session.run(None, inputs)
        output_tensor = outputs[0][0] # Get first batch output
        
        # 3. Postprocess: CHW to HWC, scale back to [0, 255]
        output_img = np.transpose(output_tensor, (1, 2, 0))
        output_img = (output_img * 0.5 + 0.5) * 255.0
        output_img = np.clip(output_img, 0, 255).astype(np.uint8)
        
        # Convert back to BGR and resize to original crop size
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        output_resized = cv2.resize(output_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
        return output_resized
    except Exception as e:
        print(f"[Warning] GFPGAN restoration failed: {e}")
        # Fallback
        gaussian = cv2.GaussianBlur(face_crop, (0, 0), 3.0)
        return cv2.addWeighted(face_crop, 1.6, gaussian, -0.6, 0)
