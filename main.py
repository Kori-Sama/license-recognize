import base64
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import uvicorn
import infer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "web")

os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/web", StaticFiles(directory=STATIC_DIR), name="web")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/run")
async def run(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    temp_img_path = "temp_image.jpg"
    cv2.imwrite(temp_img_path, img)

    output_img, label, score = infer.run(temp_img_path)

    _, img_encoded = cv2.imencode('.jpg', output_img)
    img_bytes = img_encoded.tobytes()

    return JSONResponse(content={
        "label": label,
        "score": float(score),
        "image": base64.b64encode(img_bytes).decode('utf-8')
    })


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
