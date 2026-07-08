import numpy as np

class FaceTracker:
    def __init__(self, max_lost_frames=5, smoothing_factor=0.6):
        """
        Lightweight temporal face tracker.
        max_lost_frames: Number of frames to predict before giving up.
        smoothing_factor: Weight of the new detection (0.0 to 1.0). Lower means smoother but slower tracking response.
        """
        self.max_lost_frames = max_lost_frames
        self.smoothing_factor = smoothing_factor
        
        self.last_bbox = None
        self.last_kps = None
        self.last_landmarks = None
        self.last_det_score = None
        self.lost_counter = 0
        self.is_active = False

    def update(self, detected_face):
        """
        Updates tracker state with a new successful detection.
        """
        if detected_face is None:
            self.lost_counter += 1
            if self.lost_counter > self.max_lost_frames:
                self.is_active = False
                self.reset()
            return self.predict()

        self.lost_counter = 0
        self.is_active = True

        bbox = detected_face.bbox
        kps = detected_face.kps
        landmarks = getattr(detected_face, 'landmark_2d_106', None)
        score = detected_face.det_score

        # Apply exponential moving average (EMA) smoothing to eliminate coordinates jitter
        if self.last_bbox is None:
            self.last_bbox = bbox.copy()
            self.last_kps = kps.copy()
            if landmarks is not None:
                self.last_landmarks = landmarks.copy()
            self.last_det_score = score
        else:
            self.last_bbox = self.smoothing_factor * bbox + (1.0 - self.smoothing_factor) * self.last_bbox
            self.last_kps = self.smoothing_factor * kps + (1.0 - self.smoothing_factor) * self.last_kps
            if landmarks is not None and self.last_landmarks is not None:
                self.last_landmarks = self.smoothing_factor * landmarks + (1.0 - self.smoothing_factor) * self.last_landmarks
            self.last_det_score = self.smoothing_factor * score + (1.0 - self.smoothing_factor) * self.last_det_score

        # Return a copy of the tracked/smoothed face properties
        return self._build_mock_face(detected_face)

    def predict(self):
        """
        Predicts face location based on last known state when detector fails.
        """
        if not self.is_active or self.last_bbox is None:
            return None
        
        # In a simple tracker, prediction just holds the last known smooth state
        return self._build_mock_face(None)

    def reset(self):
        self.last_bbox = None
        self.last_kps = None
        self.last_landmarks = None
        self.last_det_score = None
        self.lost_counter = 0
        self.is_active = False

    def _build_mock_face(self, original_face):
        # Create a lightweight wrapper object mimicking InsightFace Face object
        class TrackedFace:
            def __init__(self, bbox, kps, landmarks, score, embedding=None, sex=None, age=None):
                self.bbox = bbox
                self.kps = kps
                self.landmark_2d_106 = landmarks
                self.det_score = score
                self.embedding = embedding
                self.sex = sex
                self.age = age
                
        emb = getattr(original_face, 'embedding', None)
        sex = getattr(original_face, 'sex', None)
        age = getattr(original_face, 'age', None)

        return TrackedFace(
            bbox=self.last_bbox.copy(),
            kps=self.last_kps.copy(),
            landmarks=self.last_landmarks.copy() if self.last_landmarks is not None else None,
            score=self.last_det_score,
            embedding=emb,
            sex=sex,
            age=age
        )
