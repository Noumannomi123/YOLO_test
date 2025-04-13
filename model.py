from contextlib import asynccontextmanager
from fastapi import FastAPI
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import torch
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from image_utils import numpy_to_base64, base64_to_numpy
def plot_image(image, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
    plt.savefig('results.jpg')
    return np.array(Image.open('results.jpg').convert("RGB"))
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    app.state.model = model
    print("Model loaded.")
    yield
    print("Unloading YOLO model...")
    del app.state.model
    print("Model unloaded.")

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    image: str
@app.get("/")
def home():
    return {"message": "Welcome to YOLO model API."}


@app.post("/predict")
def predict(request: PredictRequest):
    image = base64_to_numpy(request.image)
    results = app.state.model(image)
    result_img = plot_results(results, image, app.state.model)
    img_base64 = numpy_to_base64(result_img)
    return JSONResponse(content = {'image': img_base64})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model:app", host='0.0.0.0', port=8000, reload=True)