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


def canny(slika, sp_prag, zg_prag):
    slika_robov = cv2.Canny(slika, sp_prag, zg_prag)
    return slika_robov


def spremeni_kontrast(slika, alfa, beta):
    return alfa * slika + beta


def showImage(name, img):
    cv2.imshow(name, img)


if __name__ == '__main__':
    print("Hello world!")

    myImg = cv2.imread("lenna.png", 0)
    showImage("Img: ", myImg)

    #i = canny(i, 80, 20)
    #showImage("Img: ", i)

    #i = spremeni_kontrast(i, 500, 0)
    #showImage("Img: ", i)

    edges = my_prewitt(myImg)
    showImage("Roberts", edges)

    cv2.waitKey()
    cv2.destroyAllWindows()
