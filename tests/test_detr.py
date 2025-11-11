"""
Run unittests for DETR model.

Author: Peter Thomas
Date: 28 October 2025
"""
import torch
import json
import unittest
import numpy as np
from PIL import Image
from vision_transformers import build_model


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

        loss_dict = criterion(outputs, dummy_targets)
        self.assertIn('loss_labels', loss_dict)
        self.assertIn('loss_boxes', loss_dict)

#    def test_overfit(self):
#        """
#        Test if the DETR model can overfit on a single image dataset.
#        """
#        test_image_path = "tests/overfit_data/cat_collects_rocks/cat_collects_rocks.jpg"
#        test_annotation_path = "tests/overfit_data/cat_collects_rocks/cat_collects_rocks.json"
#        with open(test_annotation_path, 'r') as f:
#            data = json.load(f)["data"]
#            labels = []
#            boxes = []
#            dummy_targets = []
#            for obj in data["objects"]:
#                labels.append(obj["class_id"])
#                boxes.append(obj["bbox"])
#            dummy_targets.append({"labels": labels, "boxes": boxes})
#
#        image = Image.open(test_image_path).convert("RGB")
#        image = np.asarray(image).permute(2, 0, 1)  # Convert to (C, H, W)
#        image = image / 255.0  # Normalize to [0, 1]
#
#        model, criterion = build_model(name='detr', backbone='resnet50')
#        model.to(torch.device('cpu'))
#        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#        for _ in range(20):  # Train for 20 epochs
#            model.train()
#            outputs = model(image.unsqueeze(0))
#
#            loss_dict = criterion(outputs, dummy_targets)
#            losses = sum(loss for loss in loss_dict.values())
#
#            optimizer.zero_grad()
#            losses.backward()
#            optimizer.step()
#
#        # Check that bounding boxes are close
#        with torch.no_grad():
#            model.eval()
#            outputs = model(image.unsqueeze(0)).detach().cpu()
#            pred_boxes = outputs['pred_boxes']
#            pred_classes = outputs['pred_logits'].argmax(-1)
#            for box, label in zip(dummy_targets[0]['boxes'], dummy_targets[0]['labels']):
#                iou = self.compute_iou(pred_boxes, box)
#
#                # Get index of box with highest iou
#                max_iou_idx = np.argmax(iou)
#                pred_box = pred_boxes[0, max_iou_idx]
#                pred_label = pred_classes[0, max_iou_idx]
#                max_iou = iou[max_iou_idx]
#
#                self.assertGreater(max_iou, 0.75)
#                self.assertEqual(pred_label, label)

    @staticmethod
    def compute_iou(boxes1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        min_x1 = np.maximum(boxes1[:, 0], box2[0])
        min_y1 = np.maximum(boxes1[:, 1], box2[1])
        max_x2 = np.minimum(boxes1[:, 2], box2[2])
        max_y2 = np.minimum(boxes1[:, 3], box2[3])

        inter_area = np.clamp((max_x2 - min_x1), min=0) * np.clamp((max_y2 - min_y1), min=0)
        box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou


if __name__ == '__main__':
    unittest.main()