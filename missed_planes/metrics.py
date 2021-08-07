import re

import torch.nn as nn
from sklearn.metrics import accuracy_score


class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is not None:
            return self._name

        name = self.__class__.__name__
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class Accuracy(BaseObject):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, y_pr, y_gt):
        return accuracy_score(y_pr > self.threshold, y_gt)
