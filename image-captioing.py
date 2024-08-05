import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Hugging Face pipeline
pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Streamlit app
st.title("Image Captioning with Hugging Face")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the image in Streamlit
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    with st.spinner("Generating caption..."):
        captions = pipe(image)

    # Display the caption
    st.subheader("Generated Caption:")
    st.write(captions[0]['generated_text'])

