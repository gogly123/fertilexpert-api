from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import joblib
import io

app = FastAPI()

# Load your trained model
model = joblib.load("trained_rf_model.joblib")

def extract_rgb_center(image):
    image = image.convert("RGB")
    width, height = image.size
    center_pixel = image.getpixel((width // 2, height // 2))
    return center_pixel

def create_features(r, g, b):
    return np.array([[
        r, g, b,
        r / g if g != 0 else 0,
        g / b if b != 0 else 0,
        (r + g + b) / 3,
        r / b if b != 0 else 0,
        r + g,
        g + b
    ]])

@app.post("/predict-ph/")
async def predict_ph(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    r, g, b = extract_rgb_center(image)
    features = create_features(r, g, b)
    predicted_ph = model.predict(features)[0]
    return JSONResponse(content={"predicted_ph": round(predicted_ph, 2)})
