import requests
from io import BytesIO
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from image_utils import numpy_to_base64, base64_to_numpy

st.title("YOLO Model Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    img_base64 = numpy_to_base64(image_cv2)

    response = requests.post("http://localhost:8000/predict", json={"image": img_base64})

    if response.status_code == 200:
        result_img_base64 = response.json()["image"]
        result_image = base64_to_numpy(result_img_base64)
        st.image(result_image, caption="Prediction Result")
    else:
        st.error("Failed to get a response from the server.")
