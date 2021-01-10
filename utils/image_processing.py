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


# Adjust image gamma
# Taken from https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python/41061351
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


# Estimate image light
# Taken from https://stackoverflow.com/questions/52505906/find-if-image-is-bright-or-dark/52506830
def img_estim(img, thrshld):
    is_light = np.mean(img) > thrshld
    return True if is_light else False


# Pre processes a frame
# Resizes, converts to gray and then smooths the image to eliminate noise
def pre_processing(image):
    resized_image = image_resize(image, width=500)
    to_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    pre_processed_image = cv2.GaussianBlur(to_gray, (15, 15), 0)
    return pre_processed_image


# Change Detection
def find_changes(img1, img2):
    # Find the absdiff between the two images, to find the visual differences
    diff_img = cv2.absdiff(img1, img2)
    dummy, thresh = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours[0]


def handle_contours(contours, output, area):
    for contour in contours:
        if cv2.contourArea(contour) < area:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (128, 255, 147), 2)


def clean_noise(to_clean):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(1, 1))
    return cv2.morphologyEx(to_clean, cv2.MORPH_OPEN, kernel)
