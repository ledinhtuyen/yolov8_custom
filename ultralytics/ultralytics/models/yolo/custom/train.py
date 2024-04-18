from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import CustomModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.classify.train import ClassificationTrainer

class CustomTrainer(BaseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True, *args, **kwargs):
        """Return a YOLO custom model."""
        self.kwargs = kwargs
        model = CustomModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1, *args, **kwargs)
        if weights:
            model.load(weights)
        return model
    
    def build_dataset(self, img_path, mode="train", batch=None):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.build_dataset(self, img_path, mode, batch)
        
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.get_dataloader(self, dataset_path, batch_size, rank, mode)

    def get_validator(self):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.get_validator(self)
