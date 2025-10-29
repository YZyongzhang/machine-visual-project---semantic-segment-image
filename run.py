from config import config
from PIL import Image
import os
if config.GEN_DATASET.OPEN:
    gen_dataset_config = config.GEN_DATASET
    cat_dir = gen_dataset_config.CATRAW_PATH
    dog_dir = gen_dataset_config.DOGRAW_PATH
    cat_files = [os.path.join( cat_dir , i ) for i in os.listdir(cat_dir)]
    dog_files = [os.path.join( dog_dir , i ) for i in os.listdir(dog_dir)]
    