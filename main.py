"""
Run one of these commands in terminal to launch:
streamlit run main.py
python -m streamlit run main.py
"""
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np

from models import load_model
from photo_detection import load_detectors, smart_crop, PADDING_HORIZONTAL, PADDING_VERTICAL
from preprocess import preprocess_image

def main():
    st.set_page_config(page_title="Face Mask Classifier (Auto-crop Branch)", layout="wide")

    model = load_model()
    face_detector = load_detectors()

    class_labels = ['type_1_mask_on', 'type_2_nose_exposed', 'type_3_below_chin', 'type_4_no_mask']
    display_labels = {
        'type_1_mask_on': '✅ Mask Fully On',
        'type_2_nose_exposed': '⚠️ Nose Exposed',
        'type_3_below_chin': '❌ Mask Below Chin',
        'type_4_no_mask': '🚫 No Mask'
    }

    st.title("Face Mask Detection")
    st.write("Upload an image or use webcam to detect face mask usage")

    base_dir = Path(__file__).parent
    label_images = {
        'type_1_mask_on': base_dir / 'images' / '1_mask_on.png',
        'type_2_nose_exposed': base_dir / 'images' / '2_nose_exposed.png',
        'type_3_below_chin': base_dir / 'images' / '3_below_chin.png',
        'type_4_no_mask': base_dir / 'images' / '4_no_mask.png',
    }

    image = None
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
                    cropped_image, found, _ = smart_crop(image, face_detector, PADDING_HORIZONTAL, PADDING_VERTICAL)
                    if found:
                        st.success("✓ Face detected")
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
                    cropped_image, found, _ = smart_crop(image, face_detector, PADDING_HORIZONTAL, PADDING_VERTICAL)
                    if found:
                        st.success("✓ Face detected!")
                        st.image(cropped_image, caption='Face Detected & Cropped', width='stretch')
                        image = cropped_image
                    else:
                        st.warning("⚠️ No face detected - using full image. Please retake.")
                        st.image(image, caption='Captured Image', width='stretch')
                else:
                    st.image(image, caption='Captured Image', width='stretch')

    with col2:
        if st.button('CLASSIFY IMAGE', type='primary', icon="😷", use_container_width=True):
            if image is not None:
                with st.spinner('Processing...'):
                    if model is None:
                        st.error("Model is not loaded.")
                    else:
                        processed_image = preprocess_image(image)
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