# INSTALL
```bash
git clone https://github.com/tuyenldhust/yolov8_custom.git
cd yolov8_custom/ultralytics
pip install -v -e .
```

# Config train.txt for train positive and negative images
```bash
train: 
- train.txt
train_negative:
- train_negative.txt
positive_ratio: 0.75 # postive_ratio = num_positive / (num_positive + num_negative) in 1 batch
val: 
- /u01/man/test.txt
# Classes
names:
  {0: polyp}
```
