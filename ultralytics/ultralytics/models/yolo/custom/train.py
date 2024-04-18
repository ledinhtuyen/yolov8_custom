from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import CustomModel
from ultralytics.utils import RANK
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models import yolo

class CustomTrainer(BaseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True, *args, **kwargs):
        """Return a YOLO custom model."""
        self.kwargs = kwargs
        model = CustomModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1, *args, **kwargs)
        if weights:
            model.load(weights)
            
        if kwargs["branch"].startswith("cls"):
            for m in model.modules():
                if not self.args.pretrained and hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            for p in model.parameters():
                p.requires_grad = True

        return model
    
    def build_dataset(self, img_path, mode="train", batch=None):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.build_dataset(self, img_path, mode, batch)
        elif self.kwargs['branch'].startswith("cls"):
            return ClassificationTrainer.build_dataset(self, img_path, mode, batch)
        
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.get_dataloader(self, dataset_path, batch_size, rank, mode)
        elif self.kwargs['branch'].startswith("cls"):
            return ClassificationTrainer.get_dataloader(self, dataset_path, batch_size, rank, mode)

    def get_validator(self):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.get_validator(self)
        elif self.kwargs['branch'].startswith("cls"):
            self.loss_names = ["loss"]
            args = {"task": "custom"}
            return yolo.classify.ClassificationValidator(self.test_loader, self.save_dir, args=args, _callbacks=self.callbacks)
        
    def set_model_attributes(self):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.set_model_attributes(self)
        elif self.kwargs['branch'].startswith("cls"):
            return ClassificationTrainer.set_model_attributes(self)

    def preprocess_batch(self, batch):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.preprocess_batch(self, batch)
        elif self.kwargs['branch'].startswith("cls"):
            return ClassificationTrainer.preprocess_batch(self, batch)

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.plot_training_samples(self, batch, ni)
        elif self.kwargs['branch'].startswith("cls"):
            return ClassificationTrainer.plot_training_samples(self, batch, ni)

    def plot_metrics(self):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.plot_metrics(self)
        elif self.kwargs['branch'].startswith("cls"):
            return ClassificationTrainer.plot_metrics(self)
        
    def label_loss_items(self, loss_items=None, prefix="train"):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.label_loss_items(self, loss_items, prefix)
        elif self.kwargs['branch'].startswith("cls"):
            return ClassificationTrainer.label_loss_items(self, loss_items, prefix)

    def plot_training_labels(self):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.plot_training_labels(self)
    
    def final_eval(self):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.final_eval(self)
        elif self.kwargs['branch'].startswith("cls"):
            return ClassificationTrainer.final_eval(self)
