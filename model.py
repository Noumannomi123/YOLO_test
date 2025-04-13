from contextlib import asynccontextmanager
from fastapi import FastAPI
from ultralytics import YOLO
import torch
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from image_utils import numpy_to_base64, base64_to_numpy, plot_results

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Loading YOLO model...")
        model = YOLO("yolov8n.pt")
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        app.state.model = model
        print("Model loaded.")
        yield
    except Exception as e:
        print(f"Error during model loading/unloading: {e}")
        yield
    finally:
        try:
            print("Unloading YOLO model...")
            del app.state.model
            print("Model unloaded.")
        except Exception as e:
            print(f"Error during model deletion: {e}")

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    image: str

@app.get("/")
def home():
    try:
        return {"message": "Welcome to YOLO model API."}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        image = base64_to_numpy(request.image)
        results = app.state.model(image)
        result_img = plot_results(results, image, app.state.model)
        img_base64 = numpy_to_base64(result_img)
        return JSONResponse(content={'image': img_base64})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("model:app", host='0.0.0.0', port=8000, reload=True)
    except Exception as e:
        print(f"Error running server: {e}")
