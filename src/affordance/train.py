from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from dataloader_take_tool import Dataset_affordance
from model import ResNet50
from tqdm import tqdm
from argparse import ArgumentParser


def evaluate(model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        total_size = 0
        correct_sum = 0
        for _, (images, labels) in enumerate(tqdm(data)):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            print(predictions)
            raise()
            predictions = torch.argmax(predictions, dim=1)
            correct = (predictions == labels).sum().item()
            total_size += len(labels)
            correct_sum += correct
        acc = correct_sum/total_size

    return acc
            

# def test(model_name='resnet50'):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dataset = Dataset_affordance(mode='test')
#     data = DataLoader(dataset, batch_size=32, shuffle=False)
#     model_path = 'saved_model/' + model_name + '.pth'
#     if model_name == 'resnet50':
#         model = ResNet50().to(device)
#     else:
#         model = VGG19().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     acc = evaluate(model, data)
#     print(f"Test Accuracy: {acc}")

def train():
    parser = ArgumentParser()
    parser.add_argument('--epoch_num', nargs='?', type=int, default=2)
    parser.add_argument('--batch_size', nargs='?', type=int, default=16)
    parser.add_argument('--lr', nargs='?', type=float, default=1e-5)
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-5)
    args = parser.parse_args()

    epochs = args.epoch_num
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    load_name = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(0)
    dataset = Dataset_affordance(mode='train')
    validset = Dataset_affordance(mode='val')
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validdata = DataLoader(validset, batch_size=batch_size, shuffle=False)
    
    model = ResNet50().to(device)
    
    writer = SummaryWriter(f'log/e_{epochs}_b_{batch_size}_l_{lr}')
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()
    
    if load_name is not None:
        model.load_state_dict(torch.load(load_name))
        
    print("Training")
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        loss_sum = 0
        for i, (images, labels) in enumerate(tqdm(data, ncols=100)):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Loss: {loss_sum}")
        writer.add_scalar('Loss/train', loss_sum, epoch)
        
        if epoch % 1 == 0:
            train_acc= evaluate(model, data)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            print(f"Train Accuracy: {train_acc}")
            eval_acc = evaluate(model, validdata)
            writer.add_scalar('Accuracy/validate', eval_acc, epoch)
            print(f"Valid Accuracy: {eval_acc}")
        if epoch > epochs - 4:
            torch.save(model.state_dict(), f'model/take_tool/{epoch}_{eval_acc}.pth')
    torch.save(model.state_dict(), 'model.pth')
    print("Training Done")



if __name__ == "__main__":
    train()
    # print("Test Resnet50")
    # print("")
    # print("Test VGG19")
