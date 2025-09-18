# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from typing import List
from io import BytesIO
from PIL import Image
import base64
import cv2
import numpy as np
import torch
from torchvision import transforms

app = FastAPI()

# --- CORS config ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("yolo11n.pt")  # hoặc yolov11s.pt, yolov11m.pt

# PyTorch transform
to_tensor = transforms.ToTensor()

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_base64(cv2_image):
    _, buffer = cv2.imencode(".jpg", cv2_image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    results_all = []

    for file in files:
        # Đọc ảnh từ upload
        image_bytes = await file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Chuyển sang PyTorch Tensor
        tensor_image = to_tensor(pil_image).unsqueeze(0)  # [1, C, H, W]

        # YOLO detect trực tiếp từ PIL image (YOLO ultralytics vẫn nhận PIL)
        results = model.predict(source=pil_image, save=False)

        # Vẽ ảnh dùng OpenCV
        cv_image = pil_to_cv2(pil_image)

        objects_list = []
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy)
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    objects_list.append({
                        "class_id": class_id,
                        "xyxy": [x1, y1, x2, y2],
                        "confidence": confidence
                    })

                    # Vẽ bbox
                    color = (0, 255, 0)
                    cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(cv_image, f"{class_id} {confidence:.2f}", 
                                (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, color, 1, cv2.LINE_AA)

        img_base64 = cv2_to_base64(cv_image)

        results_all.append({
            "filename": file.filename,
            "objects": objects_list,
            "num_errors": len(objects_list),
            "image_base64": img_base64
        })

    return JSONResponse(content={"results": results_all})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
