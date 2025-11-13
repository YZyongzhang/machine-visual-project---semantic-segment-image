import cv2


def Gaussian_remove_noise(img):

    denoised = cv2.GaussianBlur(img, (5, 5), 1.5)

    return denoised