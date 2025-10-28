import pyrallis
from dataclasses import field
from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = field(
        default="vit_base",
        metadata={"help": "Name of the vision transformer model architecture."}
    )
    num_classes: int = field(
        default=100,
        metadata={"help": "Number of output classes for classification."}
    )
    image_size: int = field(
        default=224,
        metadata={"help": "image size (height and width) for the model input."}
    )
    patch_size: int = field(
        default=16,
        metadata={"help": "Patch size for the vision transformer."}
    )


@dataclass
class TrainClassificationConfig:
    dataset_path: str = field(
        default="path/to/dataset",
        metadata={"help": "Path to the training dataset."}
    )
    model_save_path: str = field(
        default="path/to/save/model",
        metadata={"help": "Path to save the trained model."}
    )
    logging_path: str = field(
        default="logs/",
        metadata={"help": "Path for logging training metrics."}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training."}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for the optimizer."}
    )
    num_epochs: int = field(
        default=50,
        metadata={"help": "Number of epochs to train."}
    )
    weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Weight decay for the optimizer."}
    )
    lr_step_size: int = field(
        default=30,
        metadata={"help": "Step size for learning rate decay."}
    )
    lr_gamma: float = field(
        default=0.1,
        metadata={"help": "Gamma for learning rate decay."}
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device to use for training (e.g., 'cuda' or 'cpu')."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(),
        metadata={"help": "Model configuration settings."}
    )


class Config:
    @staticmethod
    def from_yaml(file_path: str) -> 'TrainClassificationConfig':
        """ Load configuration from a YAML file. """
        return pyrallis.parse(file_path, TrainClassificationConfig)