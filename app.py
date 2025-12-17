import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = FastAPI()

# --- Folder Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Load Model ---
MODEL_PATH = os.path.join(BASE_DIR, "food_classifier_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

LABELS = [
    "Nasi Goreng", "Ayam Goreng", "Burger", "Sate Ayam",
    "Kentang Goreng", "Bakso", "Mie Goreng"
]

INFO_MAKANAN = {
    "Nasi Goreng": {"kalori": 350, "vitamin": "B1, B2"},
    "Ayam Goreng": {"kalori": 250, "vitamin": "B3, B6"},
    "Burger": {"kalori": 500, "vitamin": "A, B12"},
    "Sate Ayam": {"kalori": 400, "vitamin": "B6"},
    "Kentang Goreng": {"kalori": 312, "vitamin": "C"},
    "Bakso": {"kalori": 280, "vitamin": "B12"},
    "Mie Goreng": {"kalori": 420, "vitamin": "B1, B2"},
}

def analyze_food(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    idx = np.argmax(prediction)

    nama_makanan = LABELS[idx]
    info = INFO_MAKANAN.get(nama_makanan, {"kalori": "-", "vitamin": "-"})

    return {
        "nama_makanan": nama_makanan,
        "kalori": info["kalori"],
        "vitamin": info["vitamin"]
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/scan", response_class=HTMLResponse)
async def scan_food(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    result = analyze_food(file_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image": f"/static/uploads/{file.filename}",
        "result": result
    })

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)