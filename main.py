import cv2
import numpy as np


def my_roberts(slika):
    jedro_x = np.array([[1, 0], [0, -1]])
    jedro_y = np.array([[0, 1], [-1, 0]])

    convolucija_x = convolution(slika, jedro_x)
    convolucija_y = convolution(slika, jedro_y)

    slika_robov = np.sqrt(np.square(convolucija_x) + np.square(convolucija_y))
    slika_robov *= 255 / np.max(slika_robov)

    return slika_robov.astype(np.uint8)


def convolution(slika, jedro):

    s_r, s_c = slika.shape
    j_r, j_c = jedro.shape

    # izračunamo padding, pri npr. 3x3 jedru vemo kje je sredina pri 2x2 ne vemo zato opravimo celoštevilsko deljenje
    pad_r = j_r // 2
    pad_c = j_c // 2

    # padded_img je enake velikosti kot slika in jo napolnimo z 0
    padded_img = np.pad(slika, ((pad_r, pad_r), (pad_c, pad_c)), 'constant')

    output_img = np.zeros((s_r, s_c))

    # delamo konvolucijo za vsak piksel
    for i in range(s_r):
        for j in range(s_c):
            # gremo od i do jedra_vrstice, in od j do jedra_stolpca kar vedno množimo z jedrom in izračunamo vsoto
            output_img[i, j] = (jedro * padded_img[i:i + j_r, j:j + j_c]).sum()

    return output_img


def my_prewitt(slika):
    jedro_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    jedro_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    convolucija_x = convolution(slika, jedro_x)
    convolucija_y = convolution(slika, jedro_y)

    slika_robov = np.sqrt(np.square(convolucija_x) + np.square(convolucija_y))
    # direction = np.arctan2(convolucija_y, convolucija_x)

    return slika_robov.astype(np.uint8)


def my_sobel(slika):
    jedro_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    jedro_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    convolucija_x = convolution(slika, jedro_x)
    convolucija_y = convolution(slika, jedro_y)

    slika_robov = np.sqrt(np.square(convolucija_x) + np.square(convolucija_y))

    return slika_robov.astype(np.uint8)


def canny(slika, sp_prag, zg_prag):
    slika_robov = cv2.Canny(slika, sp_prag, zg_prag)
    return slika_robov


def spremeni_kontrast(slika, alfa, beta):
    return alfa * slika + beta


def showImage(name, img):
    cv2.imshow(name, img)


def my_median(slika, kernel_size):
    slika_median = cv2.medianBlur(slika, kernel_size)
    return slika_median


def try_blur():
    edges_without = my_sobel(myImg)
    # showImage("Sobel", edges_without)

    smoothed = my_median(myImg, 5)
    # showImage("Smoothed", smoothed)
    edges_with = my_sobel(smoothed)

    compare = cv2.hconcat((edges_without, edges_with))
    showImage("Sobel edge detector with blur and without: ", compare)


def try_contrast():
    sob = my_sobel(myImg)
    kontrast = spremeni_kontrast(myImg, 1, 10)
    showImage("Beta is 10", kontrast)
    sobl = my_sobel(kontrast)

    compare = cv2.hconcat((sob, sobl))
    showImage("Sobel detector with original photo and with lightness photo: ", compare)


if __name__ == '__main__':
    print("Hello world!")

    myImg = cv2.imread("lenna.png", 0)
    # showImage("Img: ", myImg)

    #myImg = canny(myImg, 80, 20)
    #showImage("Canny: ", myImg)

    #myImg = spremeni_kontrast(myImg, 500, 0)
    #showImage("Spremenjen kontrast: ", myImg)

    #edges = my_roberts(myImg)
    #showImage("Roberts", edges)

    #edges = my_prewitt(myImg)
    #showImage("Prewitt", edges)

    # try_blur()
    try_contrast()


    cv2.waitKey()
    cv2.destroyAllWindows()
