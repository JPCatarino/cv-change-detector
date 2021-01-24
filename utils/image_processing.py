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
def pre_processing(image, resize_value=500, blur_ksize=(5, 5)):
    resized_image = image_resize(image, width=resize_value)
    to_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    pre_processed_image = cv2.GaussianBlur(to_gray, blur_ksize, 0)
    return pre_processed_image


# Change Detection
def find_changes(img1, img2, threshold_setting):
    # Find the absdiff between the two images, to find the visual differences
    diff_img = cv2.absdiff(img1, img2)
    dummy, thresh = cv2.threshold(diff_img, 0, 255, threshold_setting)

    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours[0], thresh


def handle_contours(contours, output, area):
    change_counter = 0
    for contour in contours:
        if cv2.contourArea(contour) < area:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (128, 255, 147), 2)
        cv2.putText(output, 'Possible Intruder', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 255, 147), 2)
        change_counter += 1

    cv2.putText(output, "Number of Intruders detected: " + str(change_counter), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 255), 2)


def clean_noise(to_clean):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(1, 1))
    return cv2.morphologyEx(to_clean, cv2.MORPH_OPEN, kernel)
