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
    imDisplay("beach.jpg", 1)


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
    # bac_con_im = imReadAndConvert("bac_con.png", 1)
    # imgEq, histOrg, histEQ = hsitogramEqualize(bac_con_im)
    corona = imReadAndConvert("beach.jpg", 2)
    imgEq, histOrg, histEQ = hsitogramEqualize(corona)
    plt.bar(np.arange(255), histOrg)
    plt.show()
    # plt.bar(np.arange(255), histEQ)
    # plt.show()


def test_hist2():
    corona = imReadAndConvert("beach.jpg", 1)
    imgEq, histOrg, histEQ = hsitogramEqualize(corona)
    cdf = histOrg.cumsum()
    cdf_normalized = cdf * histOrg.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(corona.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


def test_quant():
    corona_gray = imReadAndConvert("beach.jpg", 1)
    qImage_list, error_list = quantizeImage(corona_gray, 4, 10)
    plt.gray()
    plt.imshow(qImage_list[-1])
    plt.show()


def main():
    #test_display()
    #test_RGB2YIQ()
    #test_YIQ2RGB()
    #bgr_yiq_testMyself()
    #test_hist()
    test_hist2()
    #test_quant()


if __name__ == '__main__':
    main()