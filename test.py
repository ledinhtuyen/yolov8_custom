import os
from ultralytics import YOLO
import torch

def train():
    # Load a model
    model = YOLO("/home/s/man/Batch_Train/test/weights/best.pt", task="custom", branch="cls_vtgp")
    # model = YOLO(backbone="yolov8s.pt", task="custom", branch="detect")
    # model = YOLO("yolov8s.pt")
    # model.train(
    #     data="/home/s/man/Batch_Train/data.yaml",
    #     epochs=1,
    #     imgsz=640,
    #     batch=32,
    #     project="/home/s/man/Batch_Train/",
    #     name="test",
    #     exist_ok=True,
    #     optimizer="AdamW",
    #     lr0=1e-3,
    #     device="cuda:1",
    # )
    model.train(
        data="/mnt/tuyenld/endoscopy/vi_tri_giai_phau.json",
        epochs=1,
        imgsz=640,
        batch=16,
        project="/home/s/man/Batch_Train/",
        name="test3",
        exist_ok=True,
        optimizer="AdamW",
        lr0=1e-3,
        device="cuda:0",
        prefix_path="/home/s/tuyenld/DATA"
    )

train()
