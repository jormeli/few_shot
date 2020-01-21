import argparse
from datetime import datetime
import math
import os
from os import path
from shutil import copyfile
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
def train(experiment_dir,
          batch_size=64,
          num_epochs=10,
          lr=1e-4,
          save_every=1,
          val_split=0.2,
          data_root='./data',
          csv_file='./train_top20.csv'):
    """Create model, run training and save checkpoints."""

    transform = transforms.Compose([Rescale((224, 224)),
                                    ToTensor(),
                                    Normalize(
                                        # Mean and STD for pretrained model
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                    )])
    dataset = FashionDataset(csv_file='./train_top20.csv',
                             classes_file='./classes.txt',
                             root_dir=data_root,
                             transform=transform)

    # Split data into train/val sets
    val_set_len = math.floor(len(dataset) * val_split)
    train_set, val_set = torch.utils.data.random_split(
            dataset,
            [len(dataset) - val_set_len, val_set_len]
    )

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, dataset.num_classes)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=12)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=True, num_workers=12)

    dataloaders = {'train': train_loader, 'val': val_loader}
    datasets = {'train': train_set, 'val': val_set}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    step_history = []
    acc_history = []
    loss_history = []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))

        # Run train and val for each epoch
        for phase in ['train', 'val']:
            avg_loss = 0.
            accuracy = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Don't show progress bar for validation
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

            epoch_loss = avg_loss / len(datasets[phase])
            epoch_acc = accuracy.double() / len(datasets[phase])
            print('{} loss: {:.05f}'.format(phase, epoch_loss))
            print('{} accuracy: {:.05f}'.format(phase, epoch_acc))

            if phase == 'val':
                n_steps = epoch * len(datasets['train'])
                step_history.append(n_steps)
                acc_history.append(epoch_acc)
                loss_history.append(epoch_loss)

                # Write accuracies to a log file
                with open(path.join(experiment_dir, 'log.csv'), 'w') as f:
                    f.write('step,val_loss,val_acc\n')
                    f.write('\n'.join(['{},{},{}'.format(s,l,a)
                        for s,l,a in zip(step_history, loss_history, acc_history)]))



        if epoch % save_every == 0:
            torch.save(model.state_dict(),
                       path.join(experiment_dir, 'weights',
                                 'model_{:04d}.mdl'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Configuration file')
    args = parser.parse_args()

    experiment_name = '{0:%Y-%m-%d-%H-%M-%S}'.format(
            datetime.now())

    experiment_dir = path.join('.', 'experiments', experiment_name)
    os.makedirs(path.join(experiment_dir, 'weights'))

    # Load configuration file if specified
    if args.config:
        copyfile(args.config, path.join(experiment_dir, 'config.gin'))
        gin.parse_config_file(args.config)

    train(experiment_dir)
