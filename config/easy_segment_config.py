from box import Box
import yaml
path = './config/easy_segment.yaml'
with open(path , 'rb') as f:
    data = yaml.safe_load(f)

segment_config = Box(data)