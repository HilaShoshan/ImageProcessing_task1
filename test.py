from ex1_utils import *

def test_display():
    imDisplay("bac_con.png", 1)
    imDisplay("beach.jpg", 2)

def test_RGB2YIQ():
    beach_im = imReadAndConvert("beach.jpg", 2)
    ans = transformRGB2YIQ(beach_im)
    print("RBG->YIQ: {}".format(ans))

def test_YIQ2RGB():
    dark_im = imReadAndConvert("beach.jpg", 2)
    yiq = transformRGB2YIQ(dark_im)
    rgb = transformYIQ2RGB(yiq)
    print("YIQ->RBG: {}".format(rgb))
    cv2.imwrite('samp_rgb.jpg', rgb)
    imDisplay('samp_rgb.jpg', 2)

def main():
    #test_display()
    #test_RGB2YIQ()
    test_YIQ2RGB()

if __name__ == '__main__':
    main()