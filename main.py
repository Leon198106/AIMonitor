from ultralytics import YOLO
from fastapi import FastAPI, Query
import cv2
import pandas as pd
import json

app = FastAPI()

model = YOLO("scene.pt")
model.to("cuda:0")

imgDir = 'images/'

@app.get("/")
async def read_root():
 return {"message": "Hello, This Is An Multimodal Large-Language Model-Based Monitoring System!"}

@app.get("/api")
async def detect_scene(file: str, draw: int = Query(0)):
    # Process the uploaded image for object detection
    imgFile = imgDir + file + ".jpg"
 
    image = cv2.imread(imgFile)

    # Perform object detection with YOLOv8
    predictions = model.predict(image)
    datas = predictions[0].boxes.data

    json_serializable = [arr.tolist() for arr in datas]
    
    return {"results": json_serializable}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=80)
