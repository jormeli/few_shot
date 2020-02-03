from os import path
import random
import gin
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform

@gin.configurable
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
                                   usecols=(0,4), dtype=None, names=['image', 'label'],
                                   encoding=None)

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

    def load_image(self, image_id):
        img_file = path.join(self._root_dir,
                             'images',
                             '{}.jpg'.format(image_id))
        img = io.imread(img_file)

        # Some images are greyscale
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)

        img = self._to_pil_image(img)
        if self._transform is not None:
            img = self._transform(img)

        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.load_image(self._data['image'][idx])
        label = self._labels.index(self._data[idx][1])
        label = np.array(label)

        sample = {'image': img, 'label': torch.from_numpy(label)}

        return sample


@gin.configurable
class TripletFashionDataset(FashionDataset):
    def __init__(self, csv_file, classes_file, root_dir, transform=None):
        super().__init__(csv_file, classes_file, root_dir, transform)

        c = 0
        for label in self._labels:
            idxs = self._data['label'] == label
            if idxs.sum() < 4 and idxs.sum() > 0:
                c += 1
                self._data = self._data[~idxs]

        print(c)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self._data['label'][idx]
        pos_idxs = self._data['label'] == label
        pos_idxs[idx] = False
        pos_img_id = np.random.choice(self._data['image'][pos_idxs])
        neg_img_id = np.random.choice(self._data['image'][~pos_idxs])

        anchor_img = self.load_image(self._data['image'][idx])
        pos_img = self.load_image(pos_img_id)
        neg_img = self.load_image(neg_img_id)

        img = torch.cat([anchor_img, pos_img, neg_img], 0)

        label = np.array(self._labels.index(label))
        return {'image': img,
                'label': torch.from_numpy(label)}


class RandomAugment(object):
    def __init__(self, transforms, p=0.5):
        self.p = p
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            if random.random() < self.p:
                sample = t(sample)

        return sample
