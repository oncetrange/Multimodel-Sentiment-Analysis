import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertModel


class MultimodalSentimentAnalysis(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalSentimentAnalysis, self).__init__()
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Linear(2048, 512)

        self.bert_model = BertModel.from_pretrained('../bert-base-uncased')

        self.fc = nn.Linear((512 + self.bert_model.config.hidden_size) * 142, num_classes)

    def forward(self, images, texts):
        image_features = self.image_model(images)
        # print(texts)
        input_ids = texts.input_ids
        attention_mask = texts.attention_mask
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        # print(input_ids.shape)
        bert_output = self.bert_model(input_ids, attention_mask)
        text_features = bert_output.last_hidden_state
        image_features = image_features.unsqueeze(1).repeat(1, text_features.size(1), 1)
        # print(image_features.size(), text_features.size())
        combined_features = torch.cat((image_features, text_features), dim=2)
        # print(combined_features)
        output = self.fc(combined_features.view(combined_features.size(0), -1))
        # print(output)
        # output = F.softmax(output, dim=1)
        return output


class ImageOnlySentimentAnalysis(nn.Module):
    def __init__(self, num_classes):
        super(ImageOnlySentimentAnalysis, self).__init__()
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Linear(2048, 512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, images, texts):
        image_features = self.image_model(images)
        output = self.fc(image_features)
        return output


class TextOnlySentimentAnalysis(nn.Module):
    def __init__(self, num_classes):
        super(TextOnlySentimentAnalysis, self).__init__()
        self.bert_model = BertModel.from_pretrained('../bert-base-uncased')
        self.fc = nn.Linear((self.bert_model.config.hidden_size) * 142, num_classes)

    def forward(self, images, texts):
        input_ids = texts.input_ids
        attention_mask = texts.attention_mask
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        bert_output = self.bert_model(input_ids, attention_mask)
        text_features = bert_output.last_hidden_state
        output = self.fc(text_features.view(text_features.size(0), -1))
        return output