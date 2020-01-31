import argparse
import copy
from datetime import datetime
import math
import os
from os import path
from shutil import copyfile
import sys

import gin
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

from dataset import FashionDataset, Resize, ToTensor, Normalize, RandomAugment
from metrics import TopKAccuracy, ConfusionMatrix, PerClassAccuracy
from networks import TransferNet
from utils import log


@gin.configurable
def load_dataset(dataset_cls, val_split=0.2, augmentations=None,
                 augment_prob=0.5):
    assert 0. <= val_split <= 1.0

    if not augmentations:
        augmentations = []

    transform = transforms.Compose([
                                    RandomAugment(augmentations, p=augment_prob),
                                    Resize((224, 224)),
                                    ToTensor(),
                                    Normalize(
                                        # Mean and STD for pretrained model
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                    )])
    dataset = dataset_cls(transform=transform)

    # Split data into train/val sets
    if val_split == 1.0 or val_split == 0.0:
        return dataset, dataset

    val_set_len = math.floor(len(dataset) * val_split)
    train_set, val_set = torch.utils.data.random_split(
            dataset,
            [len(dataset) - val_set_len, val_set_len]
    )

    return train_set, val_set


def train_step(model, loss_fn, optimizer, x, y, update=False):
    optimizer.zero_grad()
    with torch.set_grad_enabled(update):
        outputs = model(x)
        loss = loss_fn(outputs, y)

    if update:
        loss.backward()
        optimizer.step()

    return loss, outputs


def train(experiment_dir,
          model,
          optimizer,
          train_loader,
          val_loader,
          loss_fn,
          metric_fns,
          num_epochs,
          device,
          save_every,
          phases,
          dump_predictions):
    dataloaders = {'train': train_loader, 'val': val_loader}

    metric_names = [m.name for m in metric_fns]
    # Initialize log files
    for phase in phases:
        with open(path.join(experiment_dir,
                            '{}_log.csv'.format(phase)), 'w') as f:
            header = ','.join(['step', 'loss'] + metric_names)
            f.write('{}\n'.format(header))

    # Training loop
    for epoch in range(1, num_epochs + 1):
        log('Epoch {}/{}'.format(epoch, num_epochs))
        log('-' * 20)

        avg_loss = 0.

        # Run train and val for each epoch
        for phase in phases:
            log(phase)
            log('-' * 20)

            if phase == 'train':
                model.train()
            else:
                model.eval()

            y_preds = []
            y_trues = []

            for batch in tqdm(dataloaders[phase]):
                imgs = batch['image'].to(device)
                labels = batch['label'].to(device)

                loss, outputs = train_step(model, loss_fn, optimizer,
                                           imgs, labels, update=phase == 'train')

                avg_loss += loss.item() * imgs.size(0)
                y_preds.append(outputs.cpu().detach().numpy())
                y_trues.append(labels.cpu().detach().numpy())

            y_preds = np.concatenate(y_preds, axis=0)
            y_trues = np.concatenate(y_trues, axis=0)

            epoch_loss = avg_loss / len(dataloaders[phase].dataset)
            epoch_metrics = {m.name: m(y_preds, y_trues) \
                    for m in metric_fns}

            log('Loss: {:.05f}'.format(epoch_loss))

            for k,v in epoch_metrics.items():
                print('{}: {}'.format(k, v))

            n_steps = epoch * len(train_loader)

            # Write accuracies to a log file
            with open(path.join(experiment_dir,
                                '{}_log.csv'.format(phase)), 'a') as f:
                m = [n_steps, epoch_loss] + \
                        [epoch_metrics[n] for n in metric_names]
                m = map(str, m)
                f.write('{}\n'.format(','.join(m)))

            if dump_predictions:
                if phase == 'val':
                    np.save(path.join(experiment_dir, 'y_preds.npy'),
                            y_preds)
                    np.save(path.join(experiment_dir, 'y_trues.npy'),
                            y_trues)

                if 'confusion_matrix' in epoch_metrics:
                    np.save(path.join(experiment_dir,
                                      '{}-confusion_matrix.npy'.format(phase)),
                            epoch_metrics['confusion_matrix'])

        log('-' * 20)
        # Save model weights every n steps
        if epoch % save_every == 0:
            torch.save(model.state_dict(),
                       path.join(experiment_dir, 'weights',
                                 'model_{:04d}.mdl'.format(epoch)))


@gin.configurable
def main(experiment_dir,
         model_cls,
         optimizer_cls,
         dataloader_cls,
         loss_fn,
         metric_fns,
         num_epochs,
         save_every,
         device_name='cuda',
         phases=None,
         dump_predictions=False):
    if not phases:
        phases = ['train', 'val']

    if device_name == "cuda":
        assert torch.cuda.is_available()

    device = torch.device(device_name)

    model = model_cls()
    model.to(device)

    log('-' * 20)
    log(model)
    log('-' * 20)

    train_set, val_set = load_dataset()
    train_loader = dataloader_cls(train_set)
    val_loader = dataloader_cls(val_set)
    optimizer = optimizer_cls(model.parameters())

    train(experiment_dir, model, optimizer, train_loader, val_loader,
          loss_fn, metric_fns, num_epochs, device, save_every, phases,
          dump_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Configuration file')
    parser.add_argument('-p', '--parameter', action='append',
                        help='Override parameters (\'parameter=value\')')
    args = parser.parse_args()

    experiment_name = '{0:%Y-%m-%d-%H-%M-%S}'.format(
            datetime.now())

    experiment_dir = path.join('.', 'experiments', experiment_name)
    os.makedirs(path.join(experiment_dir, 'weights'))

    # Load configuration file if specified
    gin.parse_config_file(args.config)
    if args.parameter:
        gin.parse_config(args.parameter)

    with open(path.join(experiment_dir, 'config.gin'), 'w') as f:
        f.write(gin.config_str())

    main(experiment_dir)
