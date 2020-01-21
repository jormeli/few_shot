import argparse
import math
import os
from os import path
import sys

import gin
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

from dataset import FashionDataset, Rescale, ToTensor, Normalize

@gin.configurable
def train(batch_size=64,
         num_epochs=10,
         lr=1e-4,
         save_every=1,
         val_split=0.2,
         data_root='./data',
         csv_file='./train_top20.csv'):
    transform = transforms.Compose([Rescale((224, 224)),
                                    ToTensor(),
                                    Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                    )])
    dataset = FashionDataset(csv_file='./train_top20.csv',
                             classes_file='./classes.txt',
                             root_dir=data_root,
                             transform=transform)

    val_set_len = math.floor(len(dataset) * val_split)
    train_set, val_set = torch.utils.data.random_split(
            dataset,
            [len(dataset) - val_set_len, val_set_len]
    )

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, dataset.num_classes)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}
    datasets = {'train': train_set, 'val': val_set}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        for phase in ['train', 'val']:
            avg_loss = 0.
            accuracy = 0


            if phase == 'train':
                model.train()
            else:
                model.eval()

            wrapper = tqdm if phase == 'train' else lambda x: x

            for batch in wrapper(dataloaders[phase]):
                imgs = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels)

                _, preds = torch.max(outputs, 1)
                avg_loss += loss.item() * imgs.size(0)
                accuracy += torch.sum(preds == labels.data)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            print('{} loss: {:.05f}'.format(phase, avg_loss /
                len(datasets[phase])))
            print('{} accuracy: {:.05f}'.format(phase,
                accuracy.double() / len(datasets[phase])))

        if epoch % save_every == 0:
            torch.save(model.state_dict(),
                       path.join('.', 'weights', 'model_{:04d}.mdl'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Configuration file')
    args = parser.parse_args()
    if args.config:
        gin.parse_config_file(args.config)

    os.makedirs('./weights', exist_ok=True)
    train()
