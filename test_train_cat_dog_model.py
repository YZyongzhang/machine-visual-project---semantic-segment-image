cat_path = './data/web_img_data/train/cat'
dog_path = './data/web_img_data/train/dog'
val_cat_path = './data/web_img_data/val/cat'
val_dog_path = './data/web_img_data/val/dog'
from dataset.loader import Loader
from torch.utils.data import DataLoader
import os
cat_files = [os.path.join(cat_path , i) for i in os.listdir(cat_path)]
dog_files = [os.path.join(dog_path , i) for i in os.listdir(dog_path)]
val_cat_files = [os.path.join(val_cat_path , i) for i in os.listdir(val_cat_path)]
val_dog_files = [os.path.join(val_dog_path , i) for i in os.listdir(val_dog_path)]
train_loader = DataLoader(Loader(cat_file=cat_files , dog_file=dog_files) , batch_size=64 , shuffle=True)
val_loader = DataLoader(Loader(cat_file=val_cat_files , dog_file=val_dog_files) , batch_size=64 , shuffle=True)

from model.cognition import SimpleCNN
model = SimpleCNN(num_classes=2)
from train.cognition_train import Trainer
from config.easy_segment_config import segment_config
trainer = Trainer(model=model , train_loader=train_loader,val_loader=val_loader , config=segment_config.TRAIN)
trainer.train()