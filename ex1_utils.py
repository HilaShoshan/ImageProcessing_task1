import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.color

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
    norm_img = img / 255.0

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
    plt.gray()
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


def show_img(img: np.ndarray):
    plt.gray()  # in case of grayscale image
    plt.imshow(img)
    plt.show()
    pass


""" (Auxiliary function for histogramEqualize and quantizeImage)
    function that helps me to check if an image is RGB
    if it is, it takes the Y channel as the new image (to work with)
    also saves the original image in YIQ representation for later use
    return: (isRGB, yiq_img, imgOrig)
"""
def case_RGB(imgOrig: np.ndarray) -> (bool, np.ndarray, np.ndarray):
    isRGB = bool(imgOrig.shape[-1] == 3)  # check if the image is RGB image
    if (isRGB):
        imgYIQ = transformRGB2YIQ(imgOrig)
        #imgYIQ = skimage.color.rgb2yiq(imgOrig)
        imgOrig = imgYIQ[:, :, 0]  # Y channel of the YIQ image
        return True, imgYIQ, imgOrig
    else:
        return False, None, imgOrig
    pass


def back_to_rgb(yiq_img: np.ndarray, y_to_update: np.ndarray) -> np.ndarray:
    #y, i, q = cv2.split(yiq_img)
    #new_img = cv2.merge((y_to_update, i, q))  # alter the y channel to the new one
    yiq_img[:, :, 0] = y_to_update
    rgb_img = transformYIQ2RGB(yiq_img)
    #rgb_img = skimage.color.yiq2rgb(yiq_img)
    return rgb_img
    pass


""" 4.4
    Equalizes the histogram of an image
    param imgOrig: Original image, grayscale or RGB image to be equalized having values in the range [0, 1].
    return: (imgEq,histOrg,histEQ)
"""
def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):

    # display the input image
    show_img(imgOrig)

    isRGB, yiq_img, imgOrig = case_RGB(imgOrig)
    imgOrig = imgOrig * 255
    imgOrig = (np.around(imgOrig)).astype('uint8')  # round & make sure all pixels are integers

    histOrg, bin_edges = np.histogram(imgOrig.flatten(), 256, [0, 256])
    cumsum = np.cumsum(histOrg * np.diff(bin_edges))  # cumulative histogram
    cdf = cumsum * histOrg.max() / cumsum.max()  # normalize to get the cumulative distribution function

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # 255 is the max value we want to reach
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # make sure all pixels are integers

    imgEq = cdf[imgOrig]

    histEQ, bin_edges2 = np.histogram(imgEq.flatten(), 256, [0, 256])

    # display the equalized output
    if(isRGB):
        imgEq = (imgEq/255).astype('float64')
        print(imgEq)
        rgb_img = back_to_rgb(yiq_img, imgEq)
        show_img(rgb_img)
    else:
        show_img(imgEq)

    return imgEq, histOrg, histEQ
    pass


def histogram_equalization(img_in):
    # segregate color streams
    b, g, r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')


    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_out = cv2.merge((img_b, img_g, img_r))
    # validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))
    # print(equ)
    # cv2.imwrite('output_name.png', equ)
    return img_out
    pass


""" 4.5
    Quantized an image in to **nQuant** colors
    param imOrig: The original image (RGB or Gray scale)
    param nQuant: Number of colors to quantize the image to
    param nIter: Number of optimization loops
    return: (List[qImage_i],List[error_i])
"""


# function to initialized the boundaries
def init_z(nQuant: int) -> np.ndarray:
    size = int(255 / nQuant)  # The initial size given for each interval (fixed - equal division)
    z = np.zeros(nQuant + 1, dtype=int)  # create an empty array representing the boundaries
    for i in range(1, nQuant):
        z[i] = z[i - 1] + size
    z[nQuant] = 255  # always start at 0 and ends at 255
    print(z)
    return z
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):

    isRGB, yiq_img, imOrig = case_RGB(imOrig)

    if np.amax(imOrig) <= 1:  # so the picture is normalized
        imOrig = imOrig * 255

    # find image's histogram
    histOrg, bin_edges = np.histogram(imOrig, bins=256, range=(0.0, 255.0))

    new_img = np.zeros(imOrig.shape)

    z = init_z(nQuant)  # boundaries
    q = np.zeros(nQuant)  # the optimal values for each ‘cell’

    min_mse = np.inf  # saving the optimal (minimal) MSE

    # lists to return
    qImage_list = list()
    error_list = list()

    for i in range(nIter):
        for cell in range(len(q)):  # select the values that each of the segments’ intensities will map to
            cell_range = np.arange(z[cell], z[cell+1])
            q[cell] = np.average(cell_range, weights=histOrg[z[cell]:z[cell+1]])  # weighted average
        for bound in range(1, len(z)-1):  # move each boundary to be in the middle of two means
            z[bound] = (q[bound-1] + q[bound]) / 2
            # alter the pixels in the new image
            new_img[np.logical_and(imOrig >= z[bound-1], imOrig <= z[bound])] = q[bound-1]
        MSE = np.square(np.subtract(imOrig, new_img)).mean()
        if MSE < min_mse:  # We've found an image with lower MSE
            min_mse = MSE  # update the minimum error
            error_list.append(MSE)
            if isRGB:
                new_img = back_to_rgb(yiq_img, new_img)  # save in new_img the image in RGB form
            qImage_list.append(new_img)
        plt.plot(error_list)
    return qImage_list, error_list
    pass

