import cv2
import numpy as np


def canny(slika, sp_prag, zg_prag):
    slika_robov = cv2.Canny(slika, sp_prag, zg_prag)
    return slika_robov


def showImage(name, img):
    cv2.imshow(name, img)


if __name__ == '__main__':
    print("Hello world!")

    i = cv2.imread("lenna.png", 0)
    showImage("Img: ", i)

    i = canny(i, 80, 20)
    showImage("Img: ", i)

    cv2.waitKey()
    cv2.destroyAllWindows()
