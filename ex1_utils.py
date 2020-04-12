import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import List

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    return 207931106


""" 4.1
    Reads an image, and returns in converted as requested
    param filename: The path to the image
    param representation: GRAY_SCALE or RGB
    return: The image object
"""
def imReadAndConvert(filename: str, representation: int) -> np.ndarray:

    if(representation == 1):  # Gray_Scale representation
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:  # =2, RGB
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

    # normalize
    norm_img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    print("type is: ", norm_img.dtype)

    return norm_img
    pass


""" 4.2
    Reads an image as RGB or GRAY_SCALE and displays it
    param filename: The path to the image
    param representation: GRAY_SCALE or RGB
    return: None
"""
def imDisplay(filename: str, representation: int):
    img = imReadAndConvert(filename, representation)

    print(img[20, 55])

    plt.imshow(img)
    plt.show()
    return None
    pass


""" 4.3 (1)
    Converts an RGB image to YIQ color space
    param imgRGB: An Image in RGB
    return: A YIQ in image color space
"""
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    s = imgRGB.shape
    img_reshape = imgRGB.reshape((s[0] * s[1]), s[2])  # s[2] = 3
    transform = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                          [0.212, -0.523, 0.311]])
    new_img = img_reshape.dot(transform)
    return new_img.reshape(s)
    pass


""" 4.3 (2)
    Converts an YIQ image to RGB color space
    param imgYIQ: An Image in YIQ
    return: A RGB in image color space
"""
def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    transform = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                          [0.212, -0.523, 0.311]])
    mat = np.linalg.inv(transform)
    s = imgYIQ.shape
    img_reshape = imgYIQ.reshape((s[0] * s[1]), s[2])
    new_img = img_reshape.dot(mat)
    return new_img.reshape(s)
    pass


""" 4.4
    Equalizes the histogram of an image
    param imgOrig: Original Histogram. grayscale or RGB image to be equalized having values in the range [0, 1].
    return: (imgEq,histOrg,histEQ)
"""
def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):

    isRGB = bool(imgOrig.shape[-1] == 3)
    if(isRGB):
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgOrig = imgYIQ[:, :, 0]  # Y channel of the YIQ image

    histOrg, bin_edges = np.histogram(imgOrig, bins=256, range=(0.0, 255.0), density=True)
    cdf = histOrg.cumsum()  # cumulative distribution function
    cdf = np.round(cdf*255 / cdf[-1])  # 255 is the max value we want to reach

    # mapping the pixels
    imgEq = np.zeros(imgOrig.shape)
    ### g_min = cdf.min()
    ### for pixel in histOrg:
    ###   pixel = np.ceil(256 * )

    histEQ, bin_edges2 = np.histogram(imgEq, bins=256, range=(0.0, 255.0), density=True)

    return imgEq, histOrg, histEQ
    pass


""" 4.5
    Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])
    """
def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):

    # find image's histogram (of probabilities)
    #histOrg, bin_edges = np.histogram(imOrig, bins=256, range=(0.0, 255.0), density=True)
    plt.hist(imOrig, bins=256, range=(0.0, 255.0), density=True)
    plt.show()

    size = 255/nQuant  # The initial size given for each interval (fixed - equal division)
    z = np.zeros(nQuant+1)  # create an empty array representing the boundaries
    for i in range(1, nQuant):
        z[i] = z[i-1] + size
    z[nQuant] = 255  # always start at 0 and ends at 255
    print(z)

    q = np.zeros(nQuant)
    for i in range(nIter):
        q[i] = 0  # formula from the class
    pass

