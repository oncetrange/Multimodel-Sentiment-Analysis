import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import random


class MultiModelData:
    def __init__(self, data_path, train_label_path, test_label_path, tokenizer, transform=None):
        self.text = {}
        self.image = {}
        self.label = {}
        self.train_label = []
        self.val_label = []
        self.test_label = []
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = 0
        files = os.listdir(data_path)
        for filename in files:
            if filename.endswith('.jpg'):
                img_path = os.path.join(data_path, filename)
                image = Image.open(img_path)
                self.image[int(filename.split('.')[0])] = image
            if filename.endswith('.txt'):
                txt_path = os.path.join(data_path, filename)
                with open(txt_path, 'r', encoding='gb18030') as f:
                    text = f.read()
                    self.text[int(filename.split('.')[0])] = text
                    if len(text) > self.max_length:
                        self.max_length = len(text)

        with open(train_label_path, 'r') as f:
            f.readline()
            lines = f.readlines()
            for line in lines:
                line_split = line.split(',')
                self.train_label.append(int(line_split[0]))
                if line_split[1] == 'negative\n':
                    self.label[int(line_split[0])] = 0
                elif line_split[1] == 'neutral\n':
                    self.label[int(line_split[0])] = 1
                elif line_split[1] == 'positive\n':
                    self.label[int(line_split[0])] = 2

        with open(test_label_path, 'r') as f:
            f.readline()
            lines = f.readlines()
            for line in lines:
                line_split = line.split(',')
                self.test_label.append(int(line_split[0]))
                self.label[int(line_split[0])] = -1

        random.shuffle(self.train_label)
        train_size = int(len(self.train_label) * 0.9)
        self.val_label = self.train_label[train_size:]
        self.train_label = self.train_label[:train_size]

        self.train_dataset = MultiModelDataset(self.image, self.text, self.label, self.train_label, self.tokenizer,
                                               self.transform, self.max_length)
        self.val_dataset = MultiModelDataset(self.image, self.text, self.label, self.val_label, self.tokenizer,
                                             self.transform, self.max_length)
        self.test_dataset = MultiModelDataset(self.image, self.text, self.label, self.test_label, self.tokenizer,
                                              self.transform, self.max_length)


class MultiModelDataset(Dataset):
    def __init__(self, image, text, label, guid, tokenizer, transform, max_length):
        self.image = image
        self.text = text
        self.labels = label
        self.guid = guid
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.guid)

    def __getitem__(self, index):
        transformed_index = self.guid[index]
        text = self.text[transformed_index]
        image = self.image[transformed_index]
        # label = F.one_hot(torch.tensor(self.label[transformed_index]), 3)
        # label = label.float()
        label = torch.tensor(self.labels[transformed_index])
        text_encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        if self.transform:
            image = self.transform(image)
        return text_encoding, image, label
