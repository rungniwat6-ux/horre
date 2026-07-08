import cv2

def detect_faces_with_auto_rotation(img, det_thresh=0.5, log_callback=None, app_instance=None):
    """
    Tries to detect faces in original image. If none found, tries rotating
    by 90, 180, 270 degrees to locate faces. Dynamic auto-thresholding drops
    thresh incrementally if no face is found.
    """
    if app_instance is None:
        return [], img, 0
        
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # Dynamic thresholds to try if default high threshold fails
    thresholds = [det_thresh]
    if det_thresh > 0.4:
        thresholds.append(det_thresh - 0.1)
    if det_thresh > 0.3:
        thresholds.append(det_thresh - 0.2)
    thresholds.append(0.25) # Absolute minimum fallback

    # Deduplicate and sort descending
    thresholds = sorted(list(set(thresholds)), reverse=True)

    for thresh in thresholds:
        # Set detection threshold on the detector model
        if hasattr(app_instance, 'det_model'):
            app_instance.det_model.det_thresh = thresh

        # 0 degrees
        faces = app_instance.get(img)
        if faces:
            return faces, img, 0
            
        # 90 degrees clockwise
        img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        faces = app_instance.get(img_90)
        if faces:
            log(f"[Process] Face detected after 90° clockwise rotation (Thresh: {thresh:.2f}).")
            return faces, img_90, 90
            
        # 180 degrees
        img_180 = cv2.rotate(img, cv2.ROTATE_180)
        faces = app_instance.get(img_180)
        if faces:
            log(f"[Process] Face detected after 180° rotation (Thresh: {thresh:.2f}).")
            return faces, img_180, 180
            
        # 270 degrees clockwise (90 counter-clockwise)
        img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        faces = app_instance.get(img_270)
        if faces:
            log(f"[Process] Face detected after 270° clockwise rotation (Thresh: {thresh:.2f}).")
            return faces, img_270, 270
            
    return [], img, 0

def rotate_back(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img
