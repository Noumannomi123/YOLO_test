import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
def numpy_to_base64(arr: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", arr)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_numpy(b64_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(img)