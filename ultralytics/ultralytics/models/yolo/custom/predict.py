# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
from PIL import Image
import cv2
import argparse
from pathlib import Path

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops

class CustomPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a custom model that performs both detection and classification.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.custom import CustomPredictor

        args = dict(model='yolov8_custom.pt', source=ASSETS)
        predictor = CustomPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes CustomPredictor setting the task to 'custom'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "custom"

    def preprocess(self, img, data_type):
        """Converts input image to model-compatible data type."""
        if data_type == 0:
            img = super().preprocess(img)
            return img
        elif data_type == 1:
            _legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"
            if not isinstance(img, torch.Tensor):
                is_legacy_transform = any(
                    _legacy_transform_name in str(transform) for transform in self.transforms.transforms
                )
                if is_legacy_transform:  # to handle legacy transforms
                    img = torch.stack([self.transforms(im) for im in img], dim=0)
                else:
                    img = torch.stack(
                        [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
                    )
            img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
            return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs, data_type=None):
        if data_type == 0:
            """Post-processes predictions and returns a list of Results objects."""
            names_det = {0: "UTDD", 1: "UTTQ", 2: "VDD", 3: "VTQ", 4: "VLHTT"}
            preds[data_type] = ops.non_max_suppression(
                preds[data_type],
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes,
            )

            if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

            results = []
            for i, pred in enumerate(preds[data_type]):
                orig_img = orig_imgs[i]
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                img_path = self.batch[0][i]
                results.append(Results(orig_img, path=img_path, names=names_det, boxes=pred))
            return results

        if data_type == 1:
            names_vtgp = {0: "Ta_trang", 1: "Hau_hong", 2: "Thuc_quan", 3: "Tam_vi", 4: "Than_vi", 5: "Phinh_vi", 6: "Hang_vi", 7: "Bo_cong_lon", 8: "Bo_cong_nho", 9: "Hanh_ta_trang"}
            """Post-processes predictions to return Results objects."""
            if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

            results = []
            for i, pred in enumerate(preds[data_type]):
                orig_img = orig_imgs[i]
                img_path = self.batch[0][i]
                results.append(Results(orig_img, path=img_path, names=names_vtgp, probs=pred))
            return results
