from ex1_utils import *
import skimage.color


# test that uses skimage.color functions to convert from rgb to yiq (and vise versa)
# to check if my functions (uses on test_YIQ2RGB) are fine
def bgr_yiq_testMyself():
    beach_im = imReadAndConvert("beach.jpg", 2)
    #imYIQ = transformRGB2YIQ(beach_im)
    imYIQ = skimage.color.rgb2yiq(beach_im)
    imRGB = skimage.color.yiq2rgb(imYIQ)
    cv2.imwrite('rgb.jpg', imRGB)
    imDisplay('rgb.jpg', 2)


def test_display():
    imDisplay("bac_con.png", 1)
    imDisplay("beach.jpg", 2)


def test_RGB2YIQ():
    beach_im = imReadAndConvert("beach.jpg", 2)
    ans = transformRGB2YIQ(beach_im)
    print("RBG->YIQ: {}".format(ans))


def test_YIQ2RGB():
    beach_im = imReadAndConvert("beach.jpg", 2)
    yiq = transformRGB2YIQ(beach_im)
    rgb = transformYIQ2RGB(yiq)
    print("YIQ->RBG: {}".format(rgb))
    cv2.imwrite('samp_rgb.jpg', rgb)
    imDisplay('samp_rgb.jpg', 2)


def main():
    #test_display()
    #test_RGB2YIQ()
    test_YIQ2RGB()
    #bgr_yiq_testMyself()


if __name__ == '__main__':
    main()