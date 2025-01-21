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

def capture_image():
    for index in range(5):  # Try different camera indices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            break
    else:
        st.error("Error: Could not open webcam.")
        return None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    captured_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_color = frame[y:y+h, x:x+w]
            if w > 200 and h > 200:  # Check if the face is properly aligned and large enough
                captured_image = frame
                break

        cv2.imshow('Align your face', frame)
        if captured_image is not None:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_image

def main():
    st.set_page_config(page_title="Face Mask Classifier", layout="wide") # set page title and layout
    
    model = load_model()
    # Define class labels
    class_labels = ['type_1_mask_on', 'type_2_nose_exposed', 'type_3_below_chin', 'type_4_no_mask']
    
    # Display labels for UI
    display_labels = {
        'type_1_mask_on': 'Mask Fully On',
        'type_2_nose_exposed': 'Nose Exposed',
        'type_3_below_chin': 'Mask Below Chin',
        'type_4_no_mask': 'No Mask'
    }
    st.title("Face Mask Detection (experimental branch)")
    st.write("Upload an image or use webcam to detect face mask usage")

    col1, col2 = st.columns(2) # create columns for image input and classification
    
    with col1: # column for image input
        input_option = st.radio("Select Input:", ["Upload Image", "Use Webcam"]) 
        
        if input_option == "Upload Image": # if user chooses to upload an image
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file: # if user has uploaded an image
                image = Image.open(uploaded_file) 
                st.image(image, caption='Uploaded Image', use_container_width=True) 
        else: # if user chooses to use webcam
            if st.button('Start Webcam'):
                image = capture_image()
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    st.image(image, caption='Captured Image', use_container_width=True) # display image for user to see

    with col2: # display classification results
        if st.button('CLASSIFY IMAGE', type='primary', icon="😷", use_container_width=True): 
            if 'image' in locals():
                with st.spinner('Processing...'): # display spinner while processing image
                    processed_image = preprocess_image(image) # preprocess image for model input
                    prediction = model.predict(processed_image) # make prediction
                    predicted_class = class_labels[np.argmax(prediction)] # get predicted class
                    display_class = display_labels[predicted_class] # get display label for predicted class
                    confidence = float(np.max(prediction)) # get confidence score for prediction
                    
                    st.success("Classification complete!")
                    st.metric("Prediction", display_class)
                    st.progress(confidence)
                    st.write(f"Confidence: {confidence:.2%}")
            else:
                st.warning("Please upload an image or take a photo first")

if __name__ == '__main__':
    main()