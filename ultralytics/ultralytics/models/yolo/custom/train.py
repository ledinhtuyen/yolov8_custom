import torch
import torch.utils
import torch.utils.data

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import CustomModel
from ultralytics.utils import RANK, LOGGER, colorstr
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models import yolo
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import CustomClsDataset, YOLOConcatDataset
from ultralytics.utils.torch_utils import strip_optimizer
from ultralytics.utils.plotting import plot_results

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
        batch["img"][idx_det] = batch["img"][idx_det].float() / 255.0
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
        batch_det = {}
        batch_vtgp = {}
        
        idx_det = batch["cls"].squeeze() >= 0
        idx_vtgp = batch["cls_img"].squeeze() >= 0
        
        batch_det["img"] = batch["img"][~idx_vtgp]
        batch_det["batch_idx"] = batch["batch_idx"][idx_det]
        batch_det["cls"] = batch["cls"][idx_det]
        batch_det["bboxes"] = batch["bboxes"][idx_det]
        batch_det["im_file"] = []
        for i in idx_det:
            batch_det["im_file"].append(batch["im_file"][i])
        
        batch_vtgp["img"] = batch["img"][idx_vtgp]
        batch_vtgp["cls"] = batch["cls_img"][idx_vtgp]
        
        DetectionTrainer.plot_training_samples(self, batch_det, ni)
        ClassificationTrainer.plot_training_samples(self, batch_vtgp, ni)

    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot) 
        
    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        import numpy as np
        from ultralytics.utils.plotting import plot_labels
        
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names_det"], save_dir=self.save_dir, on_plot=self.on_plot)
    
    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.args.plots
                    self.validator["val_det"].args.plots = self.args.plots
                    self.metrics = self.validator["val_det"](model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")
                    
        """Evaluate trained model and save validation results."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator["val_vtgp"].args.data = self.args.data
                    self.validator["val_vtgp"].args.plots = self.args.plots
                    self.metrics = self.validator["val_vtgp"](model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
