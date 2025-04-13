import requests
import base64
from io import BytesIO
import streamlit as st
from PIL import Image

response = requests.get("http://localhost:8000/predict", params={"image_path": "images/bus.jpg"})
img_base64 = response.json()["image_base64"]

img_bytes = base64.b64decode(img_base64)
image = Image.open(BytesIO(img_bytes))

st.image(image, caption="Prediction Result")