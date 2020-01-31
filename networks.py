import gin
import torch
import torch.nn as nn
from torchvision import models


@gin.configurable
class TransferNet(nn.Module):
    def __init__(self, num_outputs,
                 saved_model_path=None,
                 freeze_conv_layers=False):
        super(TransferNet, self).__init__()

        if not saved_model_path:
            self._model = models.resnet50(pretrained=True)
            self._model.fc = nn.Linear(2048, num_outputs)
        else:
            self._model = models.resnet50(pretrained=False)
            if freeze_conv_layers:
                for child in list(self._model.children())[:-3]:
                    for param in child.parameters():
                        param.requires_grad = False
            self._model.fc = nn.Linear(2048, num_outputs)
            self.load_state_dict(torch.load(saved_model_path))

    def forward(self, x):
        return self._model(x)
