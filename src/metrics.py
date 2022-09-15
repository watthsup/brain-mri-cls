from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.utils import base
import torch.nn.functional as F
import torch
import numpy as np

def accuracy(preds, labels):
    total, correct = 0, 0
    scores, predictions = torch.max(preds, 1)
    total += labels.shape[0]
    correct += (predictions == labels).sum().item()
    accuracy = (correct / total) * 100
    return accuracy

class accuracy_metric(base.Metric):
    __name__ = 'accuracy'

    def forward(self, preds, labels):
        accuracy_score = accuracy(preds, labels)
        return torch.tensor(accuracy_score)