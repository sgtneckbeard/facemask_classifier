"""
Run one of these commands in terminal to run the app

streamlit run app.py
python -m streamlit run app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import streamlit.components.v1 as components
import base64
from pathlib import Path

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('50ep_xtraContrast_cnn_facemask_model.h5')
    except:
        st.error("Failed to load model. Please ensure model file exists.")
        return None

def preprocess_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3: # if image is colored
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) # convert to grayscale
    img_array = cv2.resize(img_array, (300, 224)) # resize to model input shape
    img_array = img_array / 255.0 # normalize pixel values to 0-1 range
    img_array = img_array.reshape(1, 224, 300, 1) # reshape to model input shape
    return img_array

def scroll_to_bottom():
    scroll_script = """
    <script>
        function scrollToBottom() {
            window.setTimeout(function() {
                window.scrollTo({
                    top: document.documentElement.scrollHeight,
                    behavior: 'smooth'
                });
            }, 500);
        }
        scrollToBottom();
    </script>
    """
    try:
        components.html(scroll_script, height=0)
    except Exception as e:
        st.error(f"Failed to scroll: {e}")

def get_base64_overlay():
    overlay_path = Path(__file__).parent / "outline.png"
    with open(overlay_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    

def add_camera_overlay():
    st.markdown("""
        <style>
            .camera-container {
                position: relative;
                width: 100%;
            }
            .overlay-image {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0.5;
                pointer-events: none;
                z-index: 1;
            }
            .stCamera {
                z-index: 0;
            }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="camera-container">
            <img class="overlay-image" src="data:image/png;base64,{get_base64_overlay()}">
        </div>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Face Mask Classifier", layout="wide")
    
    model = load_model()
    class_labels = ['type_1_mask_on', 'type_2_nose_exposed', 'type_3_below_chin', 'type_4_no_mask']
    
    display_labels = {
        'type_1_mask_on': 'Mask Fully On',
        'type_2_nose_exposed': 'Nose Exposed',
        'type_3_below_chin': 'Mask Below Chin',
        'type_4_no_mask': 'No Mask'
    }
    st.title("Face Mask Detection (experimental branch)")
    st.write("Upload an image or use webcam to detect face mask usage")

    col1, col2 = st.columns(2)
    
    with col1:
        input_option = st.radio("Select Input:", ["Upload Image", "Use Webcam"]) 
        
        if input_option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                image = Image.open(uploaded_file) 
                st.image(image, caption='Uploaded Image', use_container_width=True) 
        else:
            add_camera_overlay()  # Add overlay before camera input
            img_file_buffer = st.camera_input("Take a photo")
            if img_file_buffer:
                image = Image.open(img_file_buffer)
                st.image(image, caption='Captured Image', use_container_width=True)
                scroll_to_bottom()

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
                    st.metric("Prediction", display_class)
                    st.progress(confidence)
                    st.write(f"Confidence: {confidence:.2%}")
            else:
                st.warning("Please upload an image or take a photo first")

if __name__ == '__main__':
    main()