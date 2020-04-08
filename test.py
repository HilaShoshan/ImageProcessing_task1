import numpy as np
import cv2
import matplotlib.pyplot as plt

from ex1_utils import *

def main():

    #imDisplay("bac_con.png", 1)
    #imDisplay("beach.jpg", 2)

    beach_im = imReadAndConvert("beach.jpg", 2)
    print("RBG->YIQ: {}".format(transformRGB2YIQ(beach_im)))

if __name__ == '__main__':
    main()