import os
from torch.utils.data import Dataset
from PIL import Image


class MultiModelDataset(Dataset):
    def __init__(self, data_path, train_label_path, test_label_path):
        self.text = {}
        self.image = {}
        self.label = {}
        self.train_label = []
        self.test_label = []
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
                self.label[line_split[0]] = -1

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        transformed_index = self.train_label[index]
        data = self.text[transformed_index], self.image[transformed_index]
        label = self.label[transformed_index]
        return data, label
