from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.classify import ClassificationValidator

class CustomValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.det_validator = DetectionValidator(dataloader, save_dir, pbar, args, _callbacks)
        self.vtgp_validator = ClassificationValidator(dataloader, save_dir, pbar, args, _callbacks)
      
    def __call__(self, trainer=None, model=None):
        if self.args.task == "custom":
            self.det_validator(model=model)
            self.vtgp_validator(model=model)
        else:
            super().__call__(model)