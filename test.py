import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import torch
from torchvision import datasets, transforms
from model import CNN



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)


model = CNN().to(device)
model.load_state_dict(
    torch.load("mnist_cnn.pth", map_location=device)
)
model.eval()

correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy:.2f}%")