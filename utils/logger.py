"""
Logger object for logging metrics to tensorboard (and potentially console / other backends in the future).

Author: Peter Thomas
Date: 28 October 2025
"""
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, step=0, **kwds):
        for key, value in kwds.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, global_step=step)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.log_metrics(step=step, **{f"{key}/{sub_key}": sub_value})
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    self.log_metrics(step=step, **{f"{key}/{i}": item})
            # Add more types as needed