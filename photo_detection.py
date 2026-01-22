from typing import Tuple, List
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image

PADDING_HORIZONTAL = 0.2
PADDING_VERTICAL = 0.7

def load_detectors():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

def detect_face_mediapipe(image: Image.Image, face_detector) -> Tuple[List[Tuple[int,int,int,int]], str | None]:
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_array if img_array.shape[2] == 3 else cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    if not results.multi_face_landmarks:
        return [], None
    h, w = img_array.shape[:2]
    detections = []
    for face_landmarks in results.multi_face_landmarks:
        x_coords = [lm.x * w for lm in face_landmarks.landmark]
        y_coords = [lm.y * h for lm in face_landmarks.landmark]
        x_min, y_min = int(min(x_coords)), int(min(y_coords))
        x_max, y_max = int(max(x_coords)), int(max(y_coords))
        x_min, y_min = max(0, x_min), max(0, y_min)
        detections.append((x_min, y_min, x_max - x_min, y_max - y_min))
    return detections, 'facemesh'

def smart_crop(image: Image.Image, face_detector, padding_horizontal=PADDING_HORIZONTAL, padding_vertical=PADDING_VERTICAL):
    detections, detection_type = detect_face_mediapipe(image, face_detector)
    if len(detections) == 0:
        return None, False, None
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    x, y, w, h = max(detections, key=lambda f: f[2] * f[3])
    pad_w, pad_h = int(w * padding_horizontal), int(h * padding_vertical)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)
    cropped = img_array[y1:y2, x1:x2]
    return Image.fromarray(cropped), True, detection_type