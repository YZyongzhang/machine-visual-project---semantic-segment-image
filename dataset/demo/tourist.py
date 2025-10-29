from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from PIL import Image
import random
sam = sam_model_registry["vit_b"](checkpoint="model/ckpt/segment/vit_b/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
img = np.array(Image.open('dataset/demo/cat.jpg'))
predictor.set_image(image=img)
masks, _, _ = predictor.predict()
masks = np.transpose(masks , (1 ,2 ,0))
overlay = img
for i in range(masks.shape[2]):
    masks_class = masks[..., i]

    color = np.array([255, 0, 0], dtype=np.uint8)

    alpha = random.uniform(0.3, 0.7)

    mask_3d = masks_class[:, :, None]

    overlay = np.where(mask_3d, (1 - alpha) * overlay + alpha * color, overlay)

Image.fromarray(overlay.astype(np.uint8)).save("dataset/demo/cat_masked.jpg")