import cv2
from PIL import Image
from guass import Gaussian_remove_noise

if __name__ == "__main__":
    img_path = './img/cat1.png'
    img = cv2.imread(img_path)

    # 1. 先进行高斯去噪（你已有的模块）
    img_no_noise = Gaussian_remove_noise(img)

    # 2. 转为灰度图
    gray = cv2.cvtColor(img_no_noise, cv2.COLOR_BGR2GRAY)

    # 3. Sobel 算子求 x、y 方向梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X方向梯度
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y方向梯度

    # 4. 计算梯度强度（边缘强度）
    grad = cv2.magnitude(grad_x, grad_y)

    # 5. 转换为 8 位图像以便保存
    grad = cv2.convertScaleAbs(grad)

    # 6. 保存结果
    cv2.imwrite('./img/cat_sobel_1.png', grad)
