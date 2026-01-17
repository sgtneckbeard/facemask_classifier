"""
run one of these commands in terminal to run the app

streamlit run app.py
python -m streamlit run app.py
"""

from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('50ep_xtraContrast_cnn_facemask_model.h5')
    except Exception as e:
        st.error(f"Failed to load model. Please ensure model file exists. Details: {e}")
        return None

@st.cache_resource
def load_detectors():
    """Load MediaPipe FaceMesh for better masked face detection"""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

def detect_face_mediapipe(image, face_detector):
    """Detect face using MediaPipe FaceMesh (works better with masks)"""
    img_array = np.array(image)
    
    # MediaPipe expects RGB
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
        # Get all landmark points
        x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
        
        # Calculate bounding box from landmarks
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        x_max = int(max(x_coords))
        y_max = int(max(y_coords))
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        
        detections.append((x_min, y_min, width, height))
    
    return detections, 'facemesh'

def smart_crop(image, face_detector, padding_horizontal=0.2, padding_vertical=0.3):
    """
    Smart crop using MediaPipe face detection.
    """
    detections, detection_type = detect_face_mediapipe(image, face_detector)
    
    if len(detections) == 0:
        return None, False, None
    
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    
    # Use largest face
    x, y, w, h = max(detections, key=lambda f: f[2] * f[3])
    
    # Apply padding
    pad_w = int(w * padding_horizontal)
    pad_h = int(h * padding_vertical)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)
    
    cropped = img_array[y1:y2, x1:x2]
    return Image.fromarray(cropped), True, detection_type

def preprocess_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Increase brightness and contrast for low-light conditions
    # Brightness: add value to all pixels
    brightness_gain = 20  # Increase this for more brightness (0-100)
    img_array = np.clip(img_array + brightness_gain, 0, 255)
    
    # Contrast: stretch the range of pixel values
    contrast_factor = 1.3  # Increase this for more contrast (1.0 = no change)
    img_array = cv2.convertScaleAbs(img_array, alpha=contrast_factor, beta=0)
    
    # Optional: CLAHE (Contrast Limited Adaptive Histogram Equalization) for better local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_array = clahe.apply(np.uint8(img_array))
    
    # Direct resize to match training (no letterboxing/padding)
    img_array = cv2.resize(img_array, (300, 224))  # width=300, height=224
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 224, 300, 1)
    return img_array

def main():
    st.set_page_config(page_title="Face Mask Classifier (Auto-crop Branch)", layout="wide")
    
    model = load_model()
    face_detector = load_detectors()
    
    # Padding configuration (MediaPipe uses relative padding)
    PADDING_HORIZONTAL = 0.2  # 20% of face width
    PADDING_VERTICAL = 0.7    # 70% of face height
    
    # Class labels to match four types
    class_labels = ['type_1_mask_on', 'type_2_nose_exposed', 'type_3_below_chin', 'type_4_no_mask']
    
    # Display labels for UI
    display_labels = {
        'type_1_mask_on': '✅ Mask Fully On',
        'type_2_nose_exposed': '⚠️ Nose Exposed',
        'type_3_below_chin': '❌ Mask Below Chin',
        'type_4_no_mask': '🚫 No Mask'
    }
    
    st.title("Face Mask Detection")
    st.write("Upload an image or use webcam to detect face mask usage")

    # Map each class to an image in ./images
    base_dir = Path(__file__).parent
    label_images = {
        'type_1_mask_on': base_dir / 'images' / '1_mask_on.png',
        'type_2_nose_exposed': base_dir / 'images' / '2_nose_exposed.png',
        'type_3_below_chin': base_dir / 'images' / '3_below_chin.png',
        'type_4_no_mask': base_dir / 'images' / '4_no_mask.png',
    }

    col1, col2 = st.columns(2)
    
    with col1:
        input_option = st.radio("Select Input:", ["Upload Image", "Use Webcam"])
        
        auto_crop = st.checkbox("Auto-crop (recommended)", value=True, 
                               help="Automatically crops to face region using MediaPipe")
        
        if input_option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                
                if auto_crop:
                    cropped_image, found, det_type = smart_crop(image, face_detector, PADDING_HORIZONTAL, PADDING_VERTICAL)
                    if found:
                        st.success(f"✓ Face detected")
                        st.image(cropped_image, caption='Face Detected & Cropped', width='stretch')
                        image = cropped_image
                    else:
                        st.warning("⚠️ No face detected - using full image")
                        st.image(image, caption='Uploaded Image', width='stretch')
                else:
                    st.image(image, caption='Uploaded Image', width='stretch')
        else:
            st.info("📸 Position your face in the frame")
            img_file_buffer = st.camera_input("Take a photo")
            if img_file_buffer:
                image = Image.open(img_file_buffer)
                
                if auto_crop:
                    cropped_image, found, det_type = smart_crop(image, face_detector, PADDING_HORIZONTAL, PADDING_VERTICAL)
                    if found:
                        st.success(f"✓ Face detected!")
                        st.image(cropped_image, caption='Face Detected & Cropped', width='stretch')
                        image = cropped_image
                    else:
                        st.warning("⚠️ No face detected - using full image. Please retake.")
                        st.image(image, caption='Captured Image', width='stretch')
                else:
                    st.image(image, caption='Captured Image', width='stretch')

    with col2:
        if st.button('CLASSIFY IMAGE', type='primary', icon="😷", width='stretch'):
            if 'image' in locals():
                with st.spinner('Processing...'):
                    if model is None:
                        st.error("Model is not loaded.")
                    else:
                        processed_image = preprocess_image(image)

                        # Preview the model input (grayscale, 224x300)
                        st.subheader("Model Input Preview")
                        preview = (processed_image[0, :, :, 0] * 255).astype(np.uint8)
                        st.image(preview, caption="Resized 224x300 grayscale", width='stretch')

                        prediction = model.predict(processed_image)
                        predicted_class = class_labels[np.argmax(prediction)]
                        display_class = display_labels[predicted_class]
                        confidence = float(np.max(prediction))
                        
                        st.success("Classification complete!")
                        col_img, col_info = st.columns([1, 2])
                        with col_img:
                            img_path = label_images.get(predicted_class)
                            if img_path and img_path.is_file():
                                st.image(str(img_path), width=160)
                            else:
                                st.write("Image not found for this label.")
                        with col_info:
                            st.metric("Prediction", display_class)
                            st.progress(confidence)
                            st.write(f"Confidence: {confidence:.2%}")
            else:
                st.warning("Please upload an image or take a photo first")

if __name__ == '__main__':
    main()