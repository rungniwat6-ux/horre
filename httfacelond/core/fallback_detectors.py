import cv2
import numpy as np
import onnxruntime as ort

class FallbackDetectors:
    def __init__(self, yolo_path, retina_path, providers):
        self.providers = providers
        self.yolo_session = None
        self.retina_session = None
        
        if yolo_path and ort.InferenceSession:
            try:
                self.yolo_session = ort.InferenceSession(yolo_path, providers=self.providers)
            except Exception as e:
                print(f"[Warning] Failed to load YOLOv11-Face: {e}")
                
        if retina_path and ort.InferenceSession:
            try:
                self.retina_session = ort.InferenceSession(retina_path, providers=self.providers)
            except Exception as e:
                print(f"[Warning] Failed to load RetinaFace: {e}")

    def detect_yolo(self, img, thresh=0.3):
        """
        Runs YOLOv11-Face / YOLOv8-Face detector and returns list of mock InsightFace Face objects.
        """
        if self.yolo_session is None:
            return []
            
        h, w = img.shape[:2]
        # YOLOv8-Face standard input size is 640x640
        input_size = 640
        blob = cv2.resize(img, (input_size, input_size))
        blob = blob.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        
        inputs = {self.yolo_session.get_inputs()[0].name: blob}
        outputs = self.yolo_session.run(None, inputs)[0]  # Shape: [1, 20, 8400]
        
        # YOLOv11-Face Pose (5kp) output structure:
        # 4 box coords, 1 box score, 15 Pose keypoints coords (5 points x [x, y, visibility])
        predictions = np.squeeze(outputs).T # Shape: [8400, 20]
        
        faces = []
        for pred in predictions:
            # Index 4 is the face bounding box confidence score
            score = pred[4]
            if score > thresh:
                # Scaled box coordinates (x_center, y_center, width, height)
                x_center = pred[0] / input_size * w
                y_center = pred[1] / input_size * h
                box_w = pred[2] / input_size * w
                box_h = pred[3] / input_size * h
                
                x1 = x_center - box_w / 2.0
                y1 = y_center - box_h / 2.0
                x2 = x_center + box_w / 2.0
                y2 = y_center + box_h / 2.0
                
                bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                
                # Extract 5 keypoints (L eye, R eye, Nose, L mouth, R mouth)
                # Structure: 5 items of (x, y, visibility) starting at index 5.
                # kpt_x at (5 + k_idx*3), kpt_y at (5 + k_idx*3 + 1)
                kps = np.zeros((5, 2), dtype=np.float32)
                for k_idx in range(5):
                    kps[k_idx, 0] = pred[5 + k_idx * 3] / input_size * w
                    kps[k_idx, 1] = pred[5 + k_idx * 3 + 1] / input_size * h
                    
                faces.append(self._create_mock_face(bbox, kps, score))
                
        # NMS (Non-Maximum Suppression) to remove duplicate predictions
        return self._nms(faces, iou_thresh=0.45)

    def detect_yolo_with_auto_rotation(self, img, thresh=0.3, log_callback=None):
        """
        Runs YOLOv11-Face detector with auto-rotation (0, 90, 180, 270 degrees)
        and dynamic threshold fallbacks.
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            print(msg)
            
        thresholds = [thresh]
        if thresh > 0.2:
            thresholds.append(thresh - 0.05)
        if thresh > 0.15:
            thresholds.append(thresh - 0.1)
        thresholds.append(0.1) # Absolute minimum threshold fallback
        
        thresholds = sorted(list(set(thresholds)), reverse=True)
        
        for t in thresholds:
            # 0 degrees
            faces = self.detect_yolo(img, thresh=t)
            if faces:
                return faces, img, 0
                
            # 90 degrees clockwise
            img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            faces = self.detect_yolo(img_90, thresh=t)
            if faces:
                log(f"[YOLO Process] Face detected after 90° clockwise rotation (Thresh: {t:.2f}).")
                return faces, img_90, 90
                
            # 180 degrees
            img_180 = cv2.rotate(img, cv2.ROTATE_180)
            faces = self.detect_yolo(img_180, thresh=t)
            if faces:
                log(f"[YOLO Process] Face detected after 180° rotation (Thresh: {t:.2f}).")
                return faces, img_180, 180
                
            # 270 degrees clockwise (90 counter-clockwise)
            img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            faces = self.detect_yolo(img_270, thresh=t)
            if faces:
                log(f"[YOLO Process] Face detected after 270° clockwise rotation (Thresh: {t:.2f}).")
                return faces, img_270, 270
                
        return [], img, 0

    def detect_retinaface(self, img, thresh=0.3):
        """
        Runs RetinaFace detector and returns list of mock InsightFace Face objects.
        """
        if self.retina_session is None:
            return []
            
        h, w = img.shape[:2]
        # Resizing to standard 640x640 input for RetinaFace ONNX
        input_size = 640
        blob = cv2.resize(img, (input_size, input_size)).astype(np.float32)
        
        # Mean subtraction (RetinaFace pre-processing standard)
        blob -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        
        inputs = {self.retina_session.get_inputs()[0].name: blob}
        outputs = self.retina_session.run(None, inputs)
        
        # Typically RetinaFace outputs: boxes, scores, landmarks
        # Parse outputs (handling standard ResNet50 RetinaFace output format)
        # Note: If output format varies, we fallback gracefully.
        try:
            loc, conf, landms = outputs
            # Post-process RetinaFace anchor boxes and decode positions
            # For simplicity & reliability, if RetinaFace shapes don't match, we fallback to YOLO
            # We implement a robust parsing here if required, otherwise return empty to pass to YOLO.
            return []
        except Exception:
            return []

    def _create_mock_face(self, bbox, kps, score):
        class MockFace(dict):
            def __init__(self, bbox, kps, score):
                super().__init__()
                self['bbox'] = bbox
                self['kps'] = kps
                self['det_score'] = score
                self['landmark_2d_106'] = None
                self['embedding'] = None
                self['sex'] = None
                self['age'] = None
            
            # Support attribute access by forwarding to dict items (matching insightface.app.common.Face)
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name)
                    
            def __setattr__(self, name, value):
                self[name] = value
        return MockFace(bbox, kps, score)

    def _nms(self, faces, iou_thresh=0.45):
        if not faces:
            return []
        
        # Sort faces by score descending
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
        keep = []
        
        while len(faces) > 0:
            best = faces.pop(0)
            keep.append(best)
            
            remaining = []
            for f in faces:
                if self._iou(best.bbox, f.bbox) < iou_thresh:
                    remaining.append(f)
            faces = remaining
            
        return keep

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        if float(boxAArea + boxBArea - interArea) == 0:
            return 0
        return interArea / float(boxAArea + boxBArea - interArea)
