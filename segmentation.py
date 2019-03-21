# Import the modules
import cv2
from matplotlib import cm
import numpy as np
from PIL import Image


def find(Ima):
    digit_images={}
    # Read the input image
    im = cv2.imread(Ima)

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    _,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = (cv2.boundingRect(ctr) for ctr in ctrs)



    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im_th, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (25, 255, 121), 3)
        roi = im_th[rect[1]:rect[1]+rect[3] , rect[0]: rect[0] + rect[2]]
        roi = cv2.dilate(roi, (10, 10))
        #print(rect[0],rect[1])
        outputImage = cv2.copyMakeBorder(
                     roi,
                     10,
                     10,
                     10,
                     10,
                     cv2.BORDER_CONSTANT,
                     value=(0,0,0)
                  )

        outputImage = cv2.resize(outputImage,(28,28))
        digit_images[rect[0]] = str(rect[0])+'.png'
        cv2.imwrite(str(rect[0])+'.png',outputImage)
    digit_images = dict(sorted(digit_images.items()))
    return digit_images
    print(digit_images.keys())
    cv2.imshow("Resulting Image with Rectangular ROIs", im_th)
    cv2.waitKey()
