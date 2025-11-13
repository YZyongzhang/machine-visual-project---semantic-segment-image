import cv2
from PIL import Image
from guass import Gaussian_remove_noise
if __name__ == "__main__":
    img_path = './img/cat1.png'
    img  = cv2.imread(img_path)
    img_no_noise = Gaussian_remove_noise(img)
    canny_img = cv2.Canny(img_no_noise, 100, 200)
    cv2.imwrite('./img/cat_canny_1.png' , canny_img)
