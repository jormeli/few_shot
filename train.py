from os import path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, models

from dataset import FashionDataset, Rescale, ToTensor, Normalize

def main(batch_size=64,
         num_epochs=10,
         lr=1e-4):
    transform = transforms.Compose([Rescale((224, 224)),
                                    ToTensor(),
                                    Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                    )])
    dataset = FashionDataset(csv_file='./train_top20.csv',
                             classes_file='./classes.txt',
                             root_dir='/tmp/dataset',
                             transform=transform)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, dataset.num_classes)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        avg_loss = 0.
        print('Epoch {}/{}'.format(epoch, num_epochs))
        model.train()

        for batch in dataloader:
            img = batch['image'].to(device)
            label = batch['label']
            labels = label.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            loss = loss_fn(outputs, labels)
            avg_loss += loss.item() * img.size(0)

            loss.backward()
            optimizer.step()

        print(avg_loss / len(dataset))

        if epoch % 1 == 0:
            torch.save(model.state_dict(),
                    path.join('.', 'weights', 'model_{:04d}.mdl'.format(epoch)))


if __name__ == '__main__':
    main()
