"""
Run unittests for DETR model.

Author: Peter Thomas
Date: 28 October 2025
"""
import unittest


class TestDETRModel(unittest.TestCase):
    def test_detr_initialization(self):
        from vision_transformers.models.detr import DETRModel
        model = DETRModel(num_classes=91, num_queries=100)
        self.assertIsNotNone(model)

    def test_detr_forward_pass(self):
        import torch
        from vision_transformers.models.detr import DETRModel

        model = DETRModel(num_classes=91, num_queries=100)
        dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        outputs = model(dummy_input)
        self.assertIn('pred_logits', outputs)
        self.assertIn('pred_boxes', outputs)
        self.assertEqual(outputs['pred_logits'].shape[0], 2)  # Batch size
        self.assertEqual(outputs['pred_boxes'].shape[0], 2)   # Batch size

    def test_detr_loss_computation(self):
        import torch
        from vision_transformers.models.detr import DETRModel
        from vision_transformers.loss import DETRLoss

        model = DETRModel(num_classes=91, num_queries=100)
        criterion = DETRLoss(num_classes=91, matcher=None, weight_dict=None, eos_coef=0.1, losses=['labels', 'boxes'])

        dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        outputs = model(dummy_input)

        dummy_targets = [
            {
                'labels': torch.randint(0, 91, (5,)),  # 5 objects
                'boxes': torch.rand(5, 4)               # 5 bounding boxes
            },
            {
                'labels': torch.randint(0, 91, (3,)),  # 3 objects
                'boxes': torch.rand(3, 4)               # 3 bounding boxes
            }
        ]

        loss_dict = criterion(outputs, dummy_targets)
        self.assertIn('loss_labels', loss_dict)
        self.assertIn('loss_boxes', loss_dict)

    def test_overfit(self):
        """
        Test if the DETR model can overfit on a single image dataset.
        """
        pass


if __name__ == '__main__':
    unittest.main()