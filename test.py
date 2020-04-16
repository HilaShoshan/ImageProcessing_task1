from ex1_utils import *
import skimage.color


# test that uses skimage.color functions to convert from rgb to yiq (and vise versa)
# to check if my functions (uses on test_YIQ2RGB) are fine
def bgr_yiq_testMyself():
    #beach_im = imReadAndConvert("beach.jpg", 2)
    beach_im = cv2.cvtColor(cv2.imread("beach.jpg"), cv2.COLOR_BGR2RGB)
    imYIQ = skimage.color.rgb2yiq(beach_im)
    imRGB = skimage.color.yiq2rgb(imYIQ)
    #cv2.imwrite('rgb.jpg', imRGB)
    plt.imshow(imYIQ)
    plt.show()
    #imDisplay('rgb.jpg', 2)


def test_display():
    imDisplay("bac_con.png", 1)
    imDisplay("beach.jpg", 2)


def test_RGB2YIQ():
    beach_im = imReadAndConvert("beach.jpg", 2)
    #beach_im = cv2.cvtColor(cv2.imread("beach.jpg"), cv2.COLOR_BGR2RGB)
    ans = transformRGB2YIQ(beach_im)
    #print("RBG->YIQ: {}".format(ans))
    #cv2.imwrite('y_channel_only.jpg', ans[:, :, 0])
    plt.imshow(ans)
    plt.show()
    #imDisplay("beach.jpg", 2)
    #imDisplay("y_channel_only.jpg", 1)


def test_YIQ2RGB():
    beach_im = imReadAndConvert("beach.jpg", 2)
    yiq = transformRGB2YIQ(beach_im)
    rgb = transformYIQ2RGB(yiq)
    plt.imshow(rgb)
    plt.show()


def test_hist():
    bac_con_im = imReadAndConvert("bac_con.png", 1)
    #print(bac_con_im * 255)
    img = cv2.imread("bac_con.png", cv2.IMREAD_GRAYSCALE)
    #print(img)
    hsitogramEqualize(bac_con_im)


def test_quant():
    bac_con_im = imReadAndConvert("bac_con.png", 1)
    quantizeImage(bac_con_im, 4, 10)


def main():
    #test_display()
    #test_RGB2YIQ()
    #test_YIQ2RGB()
    #bgr_yiq_testMyself()
    test_hist()
    #test_quant()


if __name__ == '__main__':
    main()