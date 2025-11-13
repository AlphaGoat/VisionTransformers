import torch
from collections import OrderedDict

from .utils import get_output_shape, initialize_parameters
from .layers import MultiHeadAttention, get_positional_encoding


class DETREncoder(torch.nn.Module):
    def __init__(self, d_model=256, num_tokens=225,
                 nhead=8, num_layers=6, positional_encoding="sinusoidal"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.pos_encoding = get_positional_encoding(positional_encoding, num_tokens, d_model)

        self.self_attn_layers = torch.nn.ModuleList()
        self.feed_forward_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.self_attn_layers.append(torch.nn.ModuleDict({
                "self_attn": MultiHeadAttention(d_model, nhead),
                "self_attn_norm": torch.nn.LayerNorm(d_model),
            }))
            self.feed_forward_layers.append(torch.nn.ModuleDict({
                "ffn": torch.nn.Linear(d_model, d_model),
                "ffn_norm": torch.nn.LayerNorm(d_model),
            }))

    def forward(self, features):
        for i in range(self.num_layers):
            # Self attention on input image features
            x = self.self_attn_layers[i]["self_attn"](
                self.pos_encoding(features),
                self.pos_encoding(features),
                features
            )
            x = self.self_attn_layers[i]["self_attn_norm"](x + features)

            # Feed forward network
            ffn_out = self.feed_forward_layers[i]["ffn"](x)
            features = encoder_out = self.feed_forward_layers[i]["ffn_norm"](ffn_out + x)

        return encoder_out


class DETRDecoder(torch.nn.Module):
    def __init__(self, d_model=256, num_tokens=225, num_queries=100,
                 nhead=8, num_layers=6, positional_encoding="sinusoidal"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.self_attn_pos_encoding = get_positional_encoding(positional_encoding, num_queries, d_model)
        self.cross_attn_pos_encoding = get_positional_encoding(positional_encoding, num_tokens, d_model)
        
        self.self_attn_layers = torch.nn.ModuleList()
        self.cross_attn_layers = torch.nn.ModuleList()
        self.feed_forward_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.self_attn_layers.append(torch.nn.ModuleDict({
                "self_attn": MultiHeadAttention(d_model, nhead),
                "self_attn_norm": torch.nn.LayerNorm(d_model),
            }))
            self.cross_attn_layers.append(torch.nn.ModuleDict({
                "cross_attn": MultiHeadAttention(d_model, nhead),
                "cross_attn_norm": torch.nn.LayerNorm(d_model),
            }))
            self.feed_forward_layers.append(torch.nn.ModuleDict({
                "ffn": torch.nn.Linear(d_model, d_model),
                "ffn_norm": torch.nn.LayerNorm(d_model),
            }))

    def forward(self, object_queries, encoder_out):

        decoder_out = {}
        for i in range(self.num_layers):
            # Self attention on object queries
            x = self.self_attn_layers[i]["self_attn"](
                self.self_attn_pos_encoding(object_queries),
                self.self_attn_pos_encoding(object_queries),
                object_queries
            )
            x = self.self_attn_layers[i]["self_attn_norm"](x + object_queries)

            # Cross attention on object queries as well as encoder outputs
            cross_attn_out = self.cross_attn_layers[i]["cross_attn"](
                self.self_attn_pos_encoding(object_queries + x),
                self.cross_attn_pos_encoding(encoder_out),
                encoder_out
            )
            x = self.cross_attn_layers[i]["cross_attn_norm"](cross_attn_out + x)

            # Feed forward layer
            ffn_out = self.feed_forward_layers[i]["ffn"](x)
            object_queries = self.feed_forward_layers[i]["ffn_norm"](ffn_out + x)
            decoder_out["layer_{:02d}".format(i)] = object_queries

        return decoder_out


class DETRBase(torch.nn.Module):
    def __init__(self, backbone, num_classes, num_queries, image_shape=(3, 256, 256), 
                 d_model=256, num_heads=8, positional_encoding="sinusoidal"): 

        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.d_model = d_model

        # Get the shape of the feature map from the backbone, which will
        # determine the number of tokens fed into the encoder
        feature_map_shape = get_output_shape(backbone, input_shape=image_shape)
        self.num_tokens = feature_map_shape[2] * feature_map_shape[3]

        # Feature embedding network
        self.conv1x1 = torch.nn.Sequential(
            OrderedDict({
                "conv1x1": torch.nn.Conv2d(feature_map_shape[1], self.d_model, kernel_size=1, stride=1),
                "batch_norm": torch.nn.BatchNorm2d(self.d_model),
                "relu": torch.nn.ReLU()
            })
        )

        # Query embeddings
        self.queries = torch.nn.Parameter(
            (torch.rand(1, num_queries, d_model)),
            requires_grad=False
        )

        # Initialize DETR encoder and decoder
        self.encoder = DETREncoder(d_model, self.num_tokens, num_heads, num_layers=6,
                                   positional_encoding=positional_encoding)
        self.decoder = DETRDecoder(d_model, self.num_tokens, num_queries, num_heads, num_layers=6,
                                   positional_encoding=positional_encoding)

        # Detection and classification heads
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, num_classes),
            torch.nn.Softmax(dim=-1),
        )
        self.detection_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4),
            torch.nn.Sigmoid(),
        )

        initialize_parameters(self)

    def forward(self, x):
        """ 
        Forward pass through the DETR model. 

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            dict: Dictionary containing class logits and bounding box predictions.
        """
        features = self.backbone(x)["feature_map"]
        features = self.conv1x1(features)
        
        # Flatten spatial features of output
        features = features.flatten(2).transpose(2, 1)

        # Pass image features to encoder
        encoder_out = self.encoder(features)

        # Generate object queries
        object_queries = self.queries.repeat(len(encoder_out), 1, 1)

        # Pass object queries and encoder outputs to decoder
        decoder_out = self.decoder(object_queries, encoder_out)

        # Generate class logits and bbox predictions 
        outs = {}
        for name, dout in decoder_out.items():
            class_logits = self.classification_head(dout)
            bbox_preds = self.detection_head(dout)
            # Output bboxes in format (cx, cy, w, h) normalized between [0, 1] (batch_size, num_queries, 4)
            # Output class logits as probabilities over all classes (batch_size, num_queries, num_classes)
            outs[name] = {
                "pred_logits": class_logits,
                "pred_boxes": bbox_preds
            }

        return outs