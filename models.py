import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model(path: str = '50ep_xtraContrast_cnn_facemask_model.h5'):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Failed to load model. Please ensure model file exists. Details: {e}")
        return None