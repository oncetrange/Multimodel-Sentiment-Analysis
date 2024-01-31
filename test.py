from utils.dataset import MultiModelData
from model.models import MultimodalSentimentAnalysis
from utils.preprocess import img_aligned_scale

from utils.model_relative import *


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    target_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transform = transforms.Compose([
        transforms.Resize(img_aligned_scale(target_size)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = MultiModelData('data/data', 'data/train.txt',
                             'data/test_without_label.txt', tokenizer=tokenizer, transform=image_transform)

    batch_size = 1
    test_data_loader = DataLoader(dataset.test_dataset, batch_size=batch_size)
    model = MultimodalSentimentAnalysis(num_classes=3).to(device)
    model.load_state_dict(torch.load('BR' + '.pth'))

    predict(model, test_data_loader)
