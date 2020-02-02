import numpy as np
import gin
import torch
from torch.utils import data
from torchvision import transforms


# DataLoader
gin.config.external_configurable(data.DataLoader, module='torch.utils.data')

# Transforms
gin.config.external_configurable(transforms.CenterCrop,
                                 module='torchvision.transforms')
gin.config.external_configurable(transforms.ColorJitter,
                                 module='torchvision.transforms')
gin.config.external_configurable(transforms.FiveCrop,
                                 module='torchvision.transforms')
gin.config.external_configurable(transforms.Grayscale,
                                 module='torchvision.transforms')
gin.config.external_configurable(transforms.Normalize,
                                 module='torchvision.transforms')
gin.config.external_configurable(transforms.RandomGrayscale,
                                 module='torchvision.transforms')
gin.config.external_configurable(transforms.RandomHorizontalFlip,
                                 module='torchvision.transforms')
gin.config.external_configurable(transforms.RandomRotation,
                                 module='torchvision.transforms')
gin.config.external_configurable(transforms.RandomVerticalFlip,
                                 module='torchvision.transforms')
gin.config.external_configurable(transforms.Resize,
                                 module='torchvision.transforms')
gin.config.external_configurable(transforms.Scale,
                                 module='torchvision.transforms')

# Class weights
gin.constant('RARE_CLASS_WEIGHTS', torch.from_numpy(np.load('./rare_class_weights.npy')).to(torch.device('cuda')))
gin.constant('CLASS_WEIGHTS', torch.from_numpy(np.load('./class_weights.npy')).to(torch.device('cuda')))
