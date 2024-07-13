import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataloader import Dataset_affordance
from torchmetrics import Accuracy
from argparse import ArgumentParser
from tqdm import tqdm

class ResNet50(pl.LightningModule):
    def __init__(self, lr=1e-5, input_channels=4, num_classes=5):
        super(ResNet50, self).__init__()
        # Load pre-trained ResNet50 model
        resnet50 = models.resnet50(pretrained=True)
        self.lr = lr

        # Modify the first layer to accept 4 input channels
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the first layer of ResNet50 with our modified layer
        self.conv1.weight.data[:, :3, :, :] = resnet50.conv1.weight.data
        self.conv1.weight.data[:, 3, :, :] = resnet50.conv1.weight.data[:, 0, :, :]  # Copy the weights of the first channel for the 4th channel
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        # Use the rest of the ResNet50 layers
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        # Modify the fully connected layer to have 5 output classes
        self.avgpool = resnet50.avgpool
        self.fc = nn.Linear(resnet50.fc.in_features, num_classes)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images, labels  # Move data to device
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images, labels  # Move data to device
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images, labels  # Move data to device
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.test_acc(preds, labels)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
    
    def get_affordance(self, data):
        return self(data)
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', nargs='?', type=int, default=4)
    parser.add_argument('--batch_size', nargs='?', type=int, default=32)
    parser.add_argument('--lr', nargs='?', type=float, default=1e-5)
    parser.add_argument('--callback', nargs='?', type=str, default=None)
    args = parser.parse_args()
    
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    callback = args.callback
        
    model = ResNet50(lr=lr)
    np.random.seed(0)
    train_data = Dataset_affordance(mode='train')
    val_data = Dataset_affordance(mode='val')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=1, num_workers=8, shuffle=False)
    if callback is not None:
        trainer = pl.Trainer(max_epochs=epochs, callbacks=[callback])
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, train_dataloader, val_data_loader)
    for _, (image, label) in enumerate(tqdm(val_data)):
        print(model.get_affordance(image))