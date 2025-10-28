import torch

from vision_transformers.layers import MultiHeadAttention, SinusoidalPositionalEncoding


class DETRBase(torch.nn.Module):
    def __init__(self, backbone, num_classes, num_queries): 
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries

    def forward(self, x):
        """ 
        Forward pass through the DETR model. 

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            dict: Dictionary containing class logits and bounding box predictions.
        """
        features = self.backbone(x)

        features = torch.nn.Conv2d(features.size(1), 256, kernel_size=1)(features)
        features = torch.nn.BatchNorm2d(256)(features)
        features = torch.nn.ReLU()(features)
        
        # Flatten spatial features of output
        features = features.flatten(2).permute(2, 0, 1)

        # Positional embedding and transformer encoder
        for _ in range(6):  # Example: 6 layers of attention
            pos_embed = SinusoidalPositionalEncoding(features.size(-1))(features) # NOTE: addition already performed in forward pass, may remove addition in head input
            attn_output = MultiHeadAttention(d_model=features.size(-1), nhead=8)(features + pos_embed, features + pos_embed, features)
            attn_output = torch.nn.LayerNorm(features.size(-1))(attn_output + features)
            ffn_output = torch.nn.Sequential(
                torch.nn.Linear(features.size(-1), features.size(-1) * 4),
                torch.nn.ReLU(),
                torch.nn.Linear(features.size(-1) * 4, features.size(-1))
            )(attn_output)
            features = torch.nn.LayerNorm(features.size(-1))(ffn_output + attn_output)

        # Prepare for decoder
        feature_embeddings = torch.nn.Embedding(self.num_queries, features.size(-1))(features)
        for _ in range(6):  # Example: 6 layers of attention
            pos_embed = SinusoidalPositionalEncoding(feature_embeddings.size(-1))(feature_embeddings)
            attn_output = MultiHeadAttention(d_model=features.size(-1), nhead=8)(feature_embeddings + pos_embed, features + pos_embed, features)
            attn_output = torch.nn.LayerNorm(features.size(-1))(attn_output + feature_embeddings)
            ffn_output = torch.nn.Sequential(
                torch.nn.Linear(features.size(-1), features.size(-1) * 4),
                torch.nn.ReLU(),
                torch.nn.Linear(features.size(-1) * 4, features.size(-1))
            )(attn_output)
            feature_embeddings = torch.nn.LayerNorm(features.size(-1))(ffn_output + attn_output)

        # Detection head (classification and bounding box regression)
        class_logits = torch.nn.Linear(in_features=feature_embeddings.size(-1), out_features=self.num_classes)(feature_embeddings)
        bbox_preds = torch.nn.Linear(in_features=feature_embeddings.size(-1), out_features=4)(feature_embeddings)

        return {'pred_logits': class_logits, 'pred_boxes': bbox_preds}