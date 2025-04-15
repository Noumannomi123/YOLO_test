import requests
from io import BytesIO
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from image_utils import numpy_to_base64, base64_to_numpy
import asyncio
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🤖", # Optional: Add an emoji or path to a favicon
    layout="wide" # Use wide layout for better side-by-side view
)

# --- UI Enhancements ---
st.title("✨ YOLO Object Detection Service ✨")
st.markdown(
    "Upload an image, and the YOLO model will detect objects within it. "
    "Results will be displayed side-by-side."
)
st.divider() # Adds a visual separator line (subtle grey)

# --- File Uploader ---
# Use columns to potentially constrain width or add info alongside
col_uploader, col_info = st.columns([2, 1]) # Give uploader more space

with col_uploader:
    uploaded_file = st.file_uploader(
        "**1. Choose an image**",
        type=["jpg", "jpeg", "png"],
        help="Select an image file (JPG, JPEG, PNG) for object detection."
    )

with col_info:
    st.markdown(
        """
        <div style="padding-top: 35px; text-align: center; color: #555;">
            <i>Your image will be processed by our backend model.</i>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Image Processing and Prediction Display ---
if uploaded_file is not None:
    st.divider()
    st.subheader("Processing & Results")

    # Use columns for side-by-side display
    col1, col2 = st.columns(2)

    try:
        # Load and display the original image
        image_pil = Image.open(uploaded_file)
        with col1:
            st.markdown("**Original Image**")
            st.image(image_pil, caption="Your Uploaded Image", use_container_width=True)

        # Convert PIL Image to OpenCV format (NumPy array BGR)
        image_cv2 = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

        # --- API Interaction with User Feedback ---
        with col2:
            st.markdown("**Prediction Result**")
            # Show a spinner while waiting for the API response
            with st.spinner('🔄 Sending image and awaiting prediction...'):
                try:
                    # 1. Convert image to base64
                    img_base64 = asyncio.run(numpy_to_base64(image_cv2))

                    # 2. Send request to the backend API
                    api_url = "http://localhost:8000/predict" # Make sure this is correct
                    response = requests.post(api_url, json={"image": img_base64}, timeout=60) # Added timeout
                    response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

                    # 3. Process the response
                    result_json = response.json()
                    if "image" in result_json:
                        result_img_base64 = result_json["image"]
                        result_image_cv2 = asyncio.run(base64_to_numpy(result_img_base64))
                        # Convert BGR (OpenCV default) back to RGB for PIL/Streamlit display
                        result_image_rgb = cv2.cvtColor(result_image_cv2, cv2.COLOR_BGR2RGB)
                        st.image(result_image_rgb, caption="Image with Detections", use_container_width=True)
                    else:
                        st.error("🚫 Prediction successful, but the response format is unexpected (missing 'image' key).")
                        st.json(result_json) # Show the raw JSON response for debugging

                except requests.exceptions.ConnectionError:
                    st.error(f"🚫 Connection Error: Could not connect to the prediction server at {api_url}. Is it running?")
                except requests.exceptions.Timeout:
                    st.error("🚫 Timeout Error: The request to the prediction server timed out.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"🚫 HTTP Error: {e}. Check the server logs for details.")
                    try:
                        # Try to display server error message if available
                        st.error(f"Server response: {response.text}")
                    except Exception:
                        pass # Ignore if response text is not available
                except Exception as e:
                    st.error(f"🚫 An unexpected error occurred: {e}")

    except Exception as e:
        st.error(f"An error occurred while processing the uploaded file: {e}")

else:
    # Optionally, display a placeholder or instruction when no file is uploaded
    st.info("☝️ Upload an image to start the detection process.")

# --- Footer (Optional) ---
st.divider()
st.markdown(
    "<div style='text-align: center; color: grey;'>Powered by Streamlit & YOLO</div>",
    unsafe_allow_html=True
)