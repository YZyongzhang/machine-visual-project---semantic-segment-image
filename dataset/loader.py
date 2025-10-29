from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import random
class Loader(Dataset):
    def __init__(self , cat_file , dog_file , config):
        super().__init__()
        self.cat_file = cat_file
        self.dog_file = dog_file
        self.transforms = transforms.Compose([
            transforms.Resize((255, 255)),
            transforms.ToTensor(),
            # 后续这里可以加入噪声。
        ])
        self.data = self.get_numpy_from_img_dir(self.cat_file , self.dog_file)

    def get_numpy_from_img_dir(self , cat , dog):
        cat_tensor_data = []
        for cat_i in cat:
            cat_tensor = self.transforms(Image.open(cat_i))
            cat_tensor_data.append((cat_tensor , 0))

        dog_tensor_data = []
        for dog_i in dog:
            dog_tensor = self.transforms(Image.open(dog_i))
            dog_tensor_data.append((dog_tensor , 1))
        
        total_data = []
        total_data.extend(cat_tensor_data)
        total_data.extend(dog_tensor_data)
        random.shuffle(total_data)
        return total_data

    def __len__(self):
        return len(self.cat_file) + len(self.dog_file)
    
    def __getitem__(self, index):
        return self.data[index][0] , self.data[index][1] # img , label
    