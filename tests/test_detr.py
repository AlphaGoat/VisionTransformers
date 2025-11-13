"""
Run unittests for DETR model.

Author: Peter Thomas
Date: 28 October 2025
"""
import math
import torch
import json
import unittest
import numpy as np
from PIL import Image

from vision_transformers import build_model
from vision_transformers.utils import get_trainable_parameters
from vision_transformers.loss import DETRLoss
from utils.hooks import get_layer_statistics
from utils.logger import Logger


class TestDETRModel(unittest.TestCase):
    def test_detr_initialization(self):
        model, loss = build_model(name='detr', backbone='resnet50')
        self.assertIsNotNone(model)

    def test_detr_forward_pass(self):
        model, loss = build_model(name='detr', backbone='resnet50')
        dummy_input = torch.randn(2, 3, 256, 256)  # Batch of 2 images
        outputs = model(dummy_input)
        for i in range(6):
            self.assertIn(f'layer_{i:02d}', outputs)

        self.assertIn('pred_logits', outputs["layer_05"])
        self.assertIn('pred_boxes', outputs["layer_05"])
        self.assertEqual(outputs["layer_05"]['pred_logits'].shape[0], 2)  # Batch size
        self.assertEqual(outputs["layer_05"]['pred_boxes'].shape[0], 2)   # Batch size

    def test_detr_loss_computation(self):

        model, criterion = build_model(name='detr', backbone='resnet50', 
                                       batch_size=2, num_classes=92, num_queries=100)

        dummy_input = torch.randn(2, 3, 256, 256)  # Batch of 2 images
        outputs = model(dummy_input)["layer_05"]

        dummy_targets = [
            {
                'labels': torch.randint(1, 91, (5,)),  # 5 objects. 2 pads
                'boxes': torch.rand(5, 4),     # 5 bounding boxes, 2 pads
            },
            {
                'labels': torch.randint(1, 91, (3,)), # 3 objects, 4 pads
                'boxes': torch.rand(3, 4) # 3 bounding boxes, 4 pads
            }
        ]

        loss_tensor = criterion(outputs, dummy_targets)
        loss = (loss_tensor.sum() / 2).cpu().item()  # Average over batch size
        self.assertGreater(loss, 0.0)
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(math.isnan(loss) == False)

    def test_hungarian_matching(self):
        batch_size = 2
        criterion = DETRLoss(num_classes=3, batch_size=batch_size)
        outputs = {
            'pred_logits': torch.tensor([[[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
                                         [[0.4, 0.3, 0.3], [0.1, 0.7, 0.2]]]),  # (B, num_queries, num_classes)
            'pred_boxes': torch.tensor([[[0.3, 0.3, 0.4, 0.4],
                                         [0.1, 0.1, 0.2, 0.2]],
                                         [[0.2, 0.2, 0.8, 0.8],   # (B, num_queries, 4)
                                          [0.5, 0.5, 0.6, 0.6]]]),
        }
        targets = [
            {
                'labels': torch.tensor([2, 1]),
                'boxes': torch.tensor([[0.1, 0.1, 0.2, 0.2],
                                       [0.3, 0.3, 0.4, 0.4]])
            },
            {
                'labels': torch.tensor([1]),
                'boxes': torch.tensor([[0.5, 0.5, 0.6, 0.6]])
            }
        ]
        indices = criterion.hungarian_matcher(outputs, targets)
        indices = [(i.cpu().numpy(), j.cpu().numpy()) for i, j in indices]
        self.assertEqual(len(indices), batch_size)
        self.assertEqual(len(indices[0][0]), 2)  # First batch has 2 matches
        self.assertEqual(indices[0][0].tolist(), [0, 1])
        self.assertEqual(indices[1][1].tolist(), [1, 0])  # Second batch has 1 match

    def test_overfit(self):
        """
        Test if the DETR model can overfit on a single image dataset.
        """
        test_image_path = "tests/overfit_data/cat_collects_rocks/cat_collects_rocks.jpg"
        test_annotation_path = "tests/overfit_data/cat_collects_rocks/cat_collects_rocks.json"
        with open(test_annotation_path, 'r') as f:
            data = json.load(f)["data"]
            labels = []
            boxes = []
            dummy_targets = []
            for obj in data["objects"]:
                labels.append(obj["class_id"])
                x_min = obj["x_min"]
                y_min = obj["y_min"]
                x_max = obj["x_max"]
                y_max = obj["y_max"]
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                width = x_max - x_min
                height = y_max - y_min
                bbox = [x_center, y_center, width, height]
                boxes.append(bbox)
            dummy_targets.append({"labels": torch.as_tensor(labels), "boxes": torch.as_tensor(boxes)})

        image = Image.open(test_image_path).convert("RGB")
        image = image.resize((256, 256))
        image = np.asarray(image).transpose(2, 0, 1)  # Convert to (C, H, W)
        image = image / 255.0  # Normalize to [0, 1]

        # Remove any existing logs
        import shutil, os
        if os.path.exists("tests/overfit_data/logs"):
            shutil.rmtree("tests/overfit_data/logs")

        # Initialize logger object to keep track of statistics
        logger = Logger(log_dir="tests/overfit_data/logs")

        model, criterion = build_model(name='detr', backbone='resnet50', 
                                       batch_size=1, num_classes=3,
                                       num_queries=100, train_backbone=True,
                                       class_weight=5.0)
        model.to(torch.device('cpu'))
        model_parameters = get_trainable_parameters(model)
        optimizer = torch.optim.AdamW(model_parameters, lr=1e-4)
        for epoch in range(20):  # Train for 20 epochs
            model.train()
            input = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            outputs = model(input)["layer_05"]

            loss_tensor = criterion(outputs, dummy_targets)
            loss = (loss_tensor.sum() / 1)  # Average over batch size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1} Loss: {loss.cpu().item()}")

            # Get layer statistics after each epoch of training
            layer_statistics = get_layer_statistics(model)
            logger.log_metrics(step=epoch, layer_stats=layer_statistics)

        # Check that bounding boxes are close
        with torch.no_grad():
            model.eval()
            input = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            outputs = model(input)["layer_05"]
            pred_boxes = outputs['pred_boxes'].detach().cpu()[0]
            pred_classes = outputs['pred_logits'].detach().cpu().argmax(-1)[0]
            for box, label in zip(dummy_targets[0]['boxes'], dummy_targets[0]['labels']):
                box = self.cxcywh_to_xyxy(box.numpy()[None, ...]).squeeze()
                pred_boxes = self.cxcywh_to_xyxy(pred_boxes.numpy()).squeeze()
                iou = self.compute_iou(pred_boxes, box)

                # Get index of box with highest iou
                max_iou_idx = np.argmax(iou)
                pred_box = pred_boxes[max_iou_idx]
                pred_label = pred_classes[max_iou_idx]
                max_iou = iou[max_iou_idx]

                self.assertGreater(max_iou, 0.75)
                self.assertEqual(pred_label, label)

    @staticmethod
    def compute_iou(boxes1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        min_x1 = np.maximum(boxes1[:, 0], box2[0])
        min_y1 = np.maximum(boxes1[:, 1], box2[1])
        max_x2 = np.minimum(boxes1[:, 2], box2[2])
        max_y2 = np.minimum(boxes1[:, 3], box2[3])

        inter_area = np.clip((max_x2 - min_x1), a_min=0, a_max=None) * np.clip((max_y2 - min_y1), a_min=0, a_max=None)
        box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou

    @staticmethod
    def cxcywh_to_xyxy(boxes: np.array) -> torch.Tensor:
        """ Convert bounding boxes from (cx, cy, w, h) to (x_min, y_min, x_max, y_max) format. """
        cx, cy, w, h = np.split(boxes, 4, axis=-1)
        x_min = cx - 0.5 * w
        y_min = cy - 0.5 * h
        x_max = cx + 0.5 * w
        y_max = cy + 0.5 * h
        return np.stack((x_min, y_min, x_max, y_max), axis=-1)


if __name__ == '__main__':
    unittest.main()