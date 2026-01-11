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

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('50ep_xtraContrast_cnn_facemask_model.h5')
    except:
        st.error("Failed to load model. Please ensure model file exists.")
        return None

def preprocess_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_array = cv2.resize(img_array, (300, 224))
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 224, 300, 1)
    return img_array

def main():
    st.set_page_config(page_title="Face Mask Classifier", layout="wide")
    
    model = load_model()
    # Updated class labels to match four types
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

        # Map each class to an image in ./images (adjust filenames as needed)
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
        
        if input_option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_container_width=True)
        else:
            img_file_buffer = st.camera_input("Take a photo")
            if img_file_buffer:
                image = Image.open(img_file_buffer)
                st.image(image, caption='Captured Image', use_container_width=True)

    with col2:
        if st.button('CLASSIFY IMAGE', type='primary', icon="😷", use_container_width=True):
            if 'image' in locals():
                with st.spinner('Processing...'):
                    processed_image = preprocess_image(image)
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