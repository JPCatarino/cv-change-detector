# Some utils for image processing
import cv2

import numpy as np


# Resizing an image mantaining aspect ratio
# Taken from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# Pre processes a frame
# Resizes, converts to gray and then smooths the image to eliminate noise
def pre_processing(image):
    resized_image = image_resize(image, width=500)
    to_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    pre_processed_image = cv2.GaussianBlur(to_gray, (21, 21), 0)
    return pre_processed_image
