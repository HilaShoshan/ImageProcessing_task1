"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import LOAD_GRAY_SCALE

import cv2
import numpy as np

image = None


""" 4.6
    GUI for gamma correction
    param img_path: Path to the image
    param rep: grayscale(1) or RGB(2)
    return: None
"""


def gammaCorrection2(img: np.ndarray, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(np.power((i / 255.0), invGamma)) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)
    pass


def gammaCorrection1(img: np.ndarray, gamma: int) -> np.ndarray:
    intensity_transform = np.arange(256)
    new_img = np.zeros(img.shape)
    for i in range(1, 256):
        intensity_transform[i] = i**(0.01/gamma)
        new_img[img == i] = intensity_transform[i]
    image = new_img
    return new_img
    pass


def gammaCorrection(img: np.ndarray, gamma: int) -> np.ndarray:
    gamma = 0.01 / gamma
    g_array = np.full(img.shape, gamma)
    new_img = np.power(img, g_array)
    return new_img
    pass


def gammaDisplay(img_path: str, rep: int):
    if rep == 1:  # Gray_Scale representation
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:  # =2, read as BGR image
        img = cv2.imread(img_path)
    cv2.namedWindow("Gamma display")
    cv2.imshow("Gamma display", img)
    global image
    image = img / 255.0
    cv2.createTrackbar("Gamma", "Gamma display", 1, 200, on_trackbar)
    # on_trackbar(100)
    cv2.waitKey()  # Wait until user press some key
    """
    while 1:
        print("im hresadasdasdasr")

        print("im hrer")
        pos = cv2.getTrackbarPos("Gamma", "Gamma display")
        print(pos)
        new_img = gammaCorrection(img, pos)
        cv2.imshow("Gamma display", new_img)
        key = cv2.waitKey(1000)
        if key == 27:
            break
        if cv2.getWindowsProperty("Gamma display", cv2.WND_PROP_VISIABLE) < 1:
            break
        cv2.waitKey(0)  # close window when a key press is detected
        cv2.destroyWindow('image')
        cv2.waitKey(1)
    """
    pass


def on_trackbar(x: int):
    new_img = gammaCorrection(image, x)
    cv2.imshow("Gamma display", new_img)
    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
