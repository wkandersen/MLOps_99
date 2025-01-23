import streamlit as st
import requests
from PIL import Image
import io

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict/"

# Streamlit App
st.title("Image Prediction App")
st.write("Upload an image, and the model will predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Predict button
    if st.button("Predict"):
        st.write("Sending the image to the model for prediction...")

        # Prepare the image for the API
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="JPEG")
        image_buffer.seek(0)
        
        # Send the image to the FastAPI backend
        response = requests.post(
            API_URL,
            files={"file": ("uploaded_image.jpg", image_buffer, "image/jpeg")}
        )
        
        # Handle the response
        if response.status_code == 200:
            data = response.json()
            st.write(f"**Filename:** {data['filename']}")
            st.write(f"**Predicted Class:** {data['predicted_class']}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
