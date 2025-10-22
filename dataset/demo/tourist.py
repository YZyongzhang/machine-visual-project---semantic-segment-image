from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_b"](checkpoint="model/ckpt/segment/vit_b/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
predictor.set_image('dataset/demo/cat.jpg')
masks, _, _ = predictor.predict()