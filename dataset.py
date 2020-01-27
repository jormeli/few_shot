from os import path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform

class FashionDataset(Dataset):
    """Fashion dataset class."""

    def __init__(self, csv_file, classes_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): CSV containing the labels
            classes_file (str): Path to a file containing all class names
            root_dir (str): Root directory of the dataset
            transform: Transform that should be applied to the data
        """

        self._csv_file = csv_file
        self._root_dir = root_dir
        self._transform = transform
        self._to_pil_image = transforms.ToPILImage()

        self._data = np.genfromtxt(csv_file, delimiter=',', skip_header=1,
                                   usecols=(0,4), dtype=None)

        # Some images are not found in the dataset for some reason,
        # so remove corresponding rows
        for idx in reversed(range(self._data.shape[0])):
            image = self._data[idx][0]
            if not path.exists(path.join(self._root_dir,
                                         'images',
                                         '{}.jpg'.format(image))):
                self._data = np.delete(self._data, idx)

        with open(classes_file, 'r') as f:
            self._labels = f.read().splitlines()

    def __len__(self):
        return self._data.shape[0]

    @property
    def num_classes(self):
        return len(self._labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = path.join(self._root_dir,
                             'images',
                             '{}.jpg'.format(self._data[idx][0]))
        img = io.imread(img_file)

        # Some images are greyscale
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)

        label = self._labels.index(self._data[idx][1].decode('ascii'))
        label = np.array(label)

        img = self._to_pil_image(img)

        sample = {'image': img, 'label': label}

        if self._transform is not None:
            sample = self._transform(sample)

        return sample


class ToTensor(object):
    """Convert sample to tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.array(image)

        image = image.transpose((2,0,1))
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label)}


class Normalize(object):
    """Normalize image with predefined mean and std."""

    def __init__(self, mean, std):
        self._norm = transforms.Normalize(
            mean=mean,
            std=std
        )

    def __call__(self, sample):
        return {'image': self._norm(sample['image']),
                'label': sample['label']}


class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        self.transform = transforms.Resize(output_size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': self.transform(image), 'label': label}


class RandomAugment(object):
    def __init__(self, transforms, p=0.5):
        self.p = p
        self.transforms = transforms

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        for t in self.transforms:
            if random.random() < self.p:
                image = t(image)

        return {'image': image, 'label': label}
