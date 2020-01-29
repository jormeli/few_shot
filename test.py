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
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

from dataset import FashionDataset, Resize, ToTensor, Normalize

def train(model_path,
          csv_file,
          data_root='./data'):
    """Create model, run training and save checkpoints."""

    if not csv_file:
        csv_file = './test.csv'

    transform = transforms.Compose([Resize((224, 224)),
                                    ToTensor(),
                                    Normalize(
                                        # Mean and STD for pretrained model
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                    )])
    dataset = FashionDataset(csv_file=csv_file,
                             classes_file='./classes.txt',
                             root_dir=data_root,
                             transform=transform)

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, dataset.num_classes)
    model.load_state_dict(torch.load(model_path))

    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=True, num_workers=12)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    print('Validating model {}'.format(model_path))

    model.eval()

    avg_loss = 0.
    accuracy = 1
    confusion_matrix = torch.zeros(dataset.num_classes, dataset.num_classes)

    for batch in tqdm(dataloader):
        imgs = batch['image'].to(device)
        labels = batch['label'].to(device)

        with torch.set_grad_enabled(False):
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

        _, preds = torch.max(outputs, 1)
        avg_loss += loss.item() * imgs.size(0)
        accuracy += torch.sum(preds == labels.data)

        for label,pred in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[label.long(), pred.long()] += 1

    with open('./top_20_classes.txt', 'r') as f:
        top20_classes = f.readlines()
        top20_classes = [l.replace(',','').strip() for l in top20_classes]

    top20_classes = [dataset._labels.index(c) for c in top20_classes]

    class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)

    print('Confusion matrix:')
    print(confusion_matrix)

    print('Per-class accuracy:')
    print(class_accuracy)

    print('Top-20 class accuracy:', class_accuracy[top20_classes])
    print(' - Avg.', confusion_matrix.diag()[top20_classes].sum() / 
            confusion_matrix[top20_classes].sum())

    total_loss = avg_loss / len(dataset)
    total_acc = accuracy.double() / len(dataset)
    print('Test loss: {:.05f}'.format(total_loss))
    print('Test accuracy: {:.05f}'.format(total_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Trained model path')
    parser.add_argument('-d', '--dataset', required=False,
                        help='Dataset (CSV file)')
    args = parser.parse_args()

    train(args.model, args.dataset)
