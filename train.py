import os
from ultralytics import YOLO
import torch

def train():
    # Load a model
    model = YOLO(backbone="yolov8s.pt", task="custom")

    model.train(
        data="data.yaml",
        epochs=1,
        imgsz=640,
        batch=24,
        project="runs/yolov8_custom",
        name="exp1",
        exist_ok=True,
        optimizer="AdamW",
        lr0=1e-3,
        device="cuda:0",
        prefix_path="/home/s/tuyenld/DATA",
    )

train()
