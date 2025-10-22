from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dog_and_cat = datasets.ImageFolder('data/cats_and_dogs', transform=transform)