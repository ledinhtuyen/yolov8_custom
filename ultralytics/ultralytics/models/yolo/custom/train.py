import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import v2 as T

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import CustomModel
from ultralytics.utils import RANK
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models import yolo
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.data import build_dataloader

from ultralytics.data.dataset import CustomClsDataset, YOLOConcatDataset

DEFAULT_MEAN = torch.Tensor([0.485, 0.456, 0.406])
DEFAULT_STD = torch.Tensor([0.229, 0.224, 0.225])

class CustomTrainer(BaseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True, *args, **kwargs):
        """Return a YOLO custom model."""
        self.kwargs = kwargs
        model = CustomModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1, *args, **kwargs)
        if weights:
            model.load(weights)
            
        # if kwargs["branch"].startswith("cls"):
        #     for m in model.modules():
        #         if not self.args.pretrained and hasattr(m, "reset_parameters"):
        #             m.reset_parameters()
        #     for p in model.parameters():
        #         p.requires_grad = True

        return model
    
    def build_dataset(self, img_path, mode="train", batch=None):
        det_dataset = DetectionTrainer.build_dataset(self, img_path[f"{mode}_det"], mode, batch)
        vtgp_dataset = CustomClsDataset(self.args, data=img_path[f"{mode}_vtgp"], augment = mode == "train", data_name="cls_vtgp", prefix=mode)

        if mode == "train":
            return YOLOConcatDataset([det_dataset, vtgp_dataset])
        else:
            return {"val_det": det_dataset, "val_vtgp": vtgp_dataset}

        
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        
        if mode == "train":
            dataloader = build_dataloader(dataset, batch_size, self.args.workers, mode == "train", rank)
        else:
            dataloader = {"val_det": build_dataloader(dataset["val_det"], batch_size, self.args.workers, mode == "train", rank),
                          "val_vtgp": build_dataloader(dataset["val_vtgp"], batch_size, self.args.workers, mode == "train", rank)}
        return dataloader


    def get_validator(self):
        self.loss_names = "box_loss", "det_cls_loss", "dfl_loss", "vtgp_loss"
        det_validator = yolo.detect.DetectionValidator(
            self.test_loader["val_det"], save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks
        )
        vtgp_validator = yolo.classify.ClassificationValidator(
            self.test_loader["val_vtgp"], save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks
        )
        return {"val_det": det_validator, "val_vtgp": vtgp_validator}
        
    def set_model_attributes(self):
        self.model.args = self.args
        
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names_det"]
        
        self.model.nc_vtgp = self.data["nc_vtgp"]
        self.model.names_vtgp = self.data["names_vtgp"]
        

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].float()
        
        idx_det = batch["data_type"] == 0
        batch["img"][idx_det] = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ])(batch["img"][idx_det])
        
        batch["img"] = batch["img"].to(torch.float32)
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["cls_img"] = batch["cls_img"].to(self.device)
        return batch

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        return ("\n" + "%14s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        # if self.kwargs['branch'].startswith("detect"):
        #     return DetectionTrainer.plot_training_samples(self, batch, ni)
        # elif self.kwargs['branch'].startswith("cls"):
        #     return ClassificationTrainer.plot_training_samples(self, batch, ni)
        pass

    def plot_metrics(self):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.plot_metrics(self)
        elif self.kwargs['branch'].startswith("cls"):
            return ClassificationTrainer.plot_metrics(self)
        
    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        import numpy as np
        from ultralytics.utils.plotting import plot_labels
        
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names_det"], save_dir=self.save_dir, on_plot=self.on_plot)
    
    def final_eval(self):
        if self.kwargs['branch'].startswith("detect"):
            return DetectionTrainer.final_eval(self)
        elif self.kwargs['branch'].startswith("cls"):
            return ClassificationTrainer.final_eval(self)
