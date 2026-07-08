import cv2
import numpy as np

def generate_occlusion_mask(face_crop, occluder_session=None):
    """
    Uses face_occluder.onnx model to generate a mask of occluded areas (e.g. hands, hair).
    Returns a mask of shape (h, w, 1) with values between 0.0 (occluded) and 1.0 (face).
    """
    if occluder_session is None:
        return None
        
    h, w, _ = face_crop.shape
    if h < 10 or w < 10:
        return None
        
    try:
        # Preprocess: Resize to 256x256, BGR to RGB, normalize to [0, 1]
        img_256 = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_256, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension: shape (1, 256, 256, 3)
        img_tensor = np.expand_dims(img_normalized, axis=0)
        
        # Run inference
        inputs = {occluder_session.get_inputs()[0].name: img_tensor}
        outputs = occluder_session.run(None, inputs)
        
        # Output shape is (1, 256, 256, 1), values 0.0 to 1.0
        # Threshold to get a strict binary mask where 0 is occluded (hands/shoulder) and 1 is face skin
        mask_256 = (outputs[0][0, :, :, 0] >= 0.35).astype(np.float32)
        
        # Resize back to original crop size
        mask_resized = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Ensure shape is (h, w, 1)
        return np.expand_dims(mask_resized, axis=2)
    except Exception as e:
        print(f"[Warning] Face occluder inference failed: {e}")
        return None
