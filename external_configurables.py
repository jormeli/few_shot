import gin
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
