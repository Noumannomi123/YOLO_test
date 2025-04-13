import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
def numpy_to_base64(arr: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", arr)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_numpy(b64_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

def plot_image(image, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")
    return ax

def plot_results(results, image, model):
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            label = f"{model.names[int(cls)]} {conf:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    plot_image(image, ax)
    plt.savefig('results.jpg', bbox_inches='tight', pad_inches=0)
    return np.array(Image.open('results.jpg').convert("RGB"))