"""
Test deformable attention classification model.

Author: Peter Thomas
Date: 28 October 2025
"""
import torch
import unittest
from vision_transformers import build_vision_transformer_backbone
from vision_transformers import DeformableAttentionTransformerClassifier


class TestDeformableAttentionTransformerClassifier(unittest.TestCase):
    def test_initialization(self):
        from vision_transformers.dat import DeformableAttentionTransformerClassifier

        backbone = build_vision_transformer_backbone(name='vgg16', pretrained=False, train_backbone=False)
        model = DeformableAttentionTransformerClassifier(backbone=backbone, image_shape=(3, 256, 256), nhead=8, num_classes=10)

        self.assertIsNotNone(model)
        self.assertEqual(model.nhead, 8)
        self.assertEqual(model.classifier.out_features, 10)

    def test_forward_pass(self):
        backbone = build_vision_transformer_backbone(name='vgg16', pretrained=False, train_backbone=False)
        model = DeformableAttentionTransformerClassifier(backbone=backbone, image_shape=(3, 256, 256), nhead=8, num_classes=10)

        dummy_input = torch.randn(2, 3, 256, 256)  # Batch of 2 images
        outputs = model(dummy_input)

        self.assertEqual(outputs.shape, (2, 10))  # Batch size x num_classes

    def test_overfit_single_batch(self):
        test_image_path = "tests/overfit_data/cat_collects_rocks/cat_collects_rocks.jpg"

        backbone = build_vision_transformer_backbone(name='vgg16', pretrained=False, train_backbone=True)
        model = DeformableAttentionTransformerClassifier(backbone=backbone, image_shape=(3, 256, 256), nhead=8, num_classes=2)
        model.to(torch.device('cpu'))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        for _ in range(20):
            image = torch.randn(1, 3, 256, 256)  # Single image
            label = torch.tensor([1])  # Assume class '1' is the correct class

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()
            self.assertEqual(predicted_class, 1)  # Check if the model predicts the correct class
