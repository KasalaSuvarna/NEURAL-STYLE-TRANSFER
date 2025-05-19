# app.py

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

# Load model once (outside function)
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

model = load_model()

# Preprocessing
def preprocess(image_data, target_size=(256, 256)):
    img = Image.open(image_data).convert("RGB")
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img, dtype=tf.float32)

# Main Interface
st.title("🎨 Neural Style Transfer App")
st.write("Upload a content image and a style image, and this app will apply artistic style transfer!")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    content_img = preprocess(content_file)
    style_img = preprocess(style_file)

    st.subheader("Content Image")
    st.image(content_file, use_column_width=True)

    st.subheader("Style Image")
    st.image(style_file, use_column_width=True)

    if st.button("Apply Style"):
        with st.spinner("Stylizing..."):
            output = model(content_img, style_img)[0]
            output_image = tf.squeeze(output).numpy()
            output_image = (output_image * 255).astype(np.uint8)
            output_pil = Image.fromarray(output_image)

            # Display result
            st.subheader("Stylized Image")
            st.image(output_pil, use_column_width=True)

            # Download option
            buf = io.BytesIO()
            output_pil.save(buf, format="JPEG")
            st.download_button("Download Stylized Image", data=buf.getvalue(),
                               file_name="stylized_output.jpg", mime="image/jpeg")
