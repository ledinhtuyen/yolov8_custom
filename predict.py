from ultralytics import YOLO

model = YOLO('/mnt/tuyenld/runs/yolov8l_custom/custom_300_epoch_dung/weights/best.pt', task="custom").to("cuda:0")

results = model(['/home/s/DATA/Vi_tri_giai_phau/5_Phinh_vi/LCI/PKHL_12 201009_200915_BN051_008.jpg'], data_type=1)
print(results)
