import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import BertTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, criterion, optimizer):
    model.train()
    loss_sum = 0
    loop = tqdm(train_loader, total=len(train_loader))
    for texts, images, labels in loop:
        # print(texts)
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, texts)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss_sum)


def evaluate(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    confuse_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    loop = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for texts, images, labels in loop:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(images, texts)
            # print(outputs)
            predicted = torch.argmax(outputs, dim=1)
            # labels = torch.argmax(labels, dim=1)
            # print(predicted, labels)
            for i in range(len(predicted)):
                confuse_matrix[predicted[i]][labels[i]] += 1
            total_samples += labels.size(0)
            # print(labels)
            # print(predicted)
            total_correct += (predicted == labels).sum().item()
            # print(total_correct, total_samples)
    accuracy = total_correct / total_samples
    return accuracy, confuse_matrix


def predict(model, test_loader):
    guid_list = []
    with open('data/test_without_label.txt', 'r') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            line_split = line.split(',')
            guid_list.append(line_split[0])

    model.eval()
    step = 0
    loop = tqdm(test_loader, total=len(test_loader))
    content = 'guid,tag\n'
    with open('data/test.txt', 'w') as f:
        with torch.no_grad():
            for texts, images, _ in loop:
                images = images.to(device)
                texts = texts.to(device)
                outputs = model(images, texts)
                predicted = torch.argmax(outputs, dim=1)
                if predicted == 0:
                    content += guid_list[step] + ',negative\n'
                if predicted == 1:
                    content += guid_list[step] + ',neutral\n'
                if predicted == 2:
                    content += guid_list[step] + ',positive\n'
                step += 1
        f.write(content)
