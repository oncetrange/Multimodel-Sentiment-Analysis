from utils.dataset import MultiModelData
from model.models import MultimodalSentimentAnalysis
from utils.preprocess import img_aligned_scale

from utils.model_relative import *


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    target_size = 224
    model = MultimodalSentimentAnalysis(num_classes=3)
    image_transform = transforms.Compose([
        transforms.Resize(img_aligned_scale(target_size)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = MultiModelData('data/data', 'data/train.txt',
                                'data/test_without_label.txt', tokenizer=tokenizer, transform=image_transform)

    batch_size = 32
    train_data_loader = DataLoader(dataset.train_dataset, batch_size=batch_size)
    val_data_loader = DataLoader(dataset.val_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(dataset.test_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = VGGNet()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 8
    for epoch in range(num_epochs):
        train(model, train_data_loader, criterion, optimizer)
        accuracy, _ = evaluate(model, val_data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy}")

    torch.save(model.state_dict(), 'BR' + '.pth')
