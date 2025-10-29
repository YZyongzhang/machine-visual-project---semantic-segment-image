# from torchvision import datasets, transforms

# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])

# dog_and_cat = datasets.ImageFolder('data/cats_and_dogs', transform=transform)

from datasets import load_dataset

dataset = load_dataset("microsoft/cats_vs_dogs")