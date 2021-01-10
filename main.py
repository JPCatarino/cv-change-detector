# Python Change Detector using Open CV

import cv2
import argparse

import numpy as np

from utils import image_processing


def main_image_compare(img1, img2):
    output = image_processing.image_resize(img2.copy(), width=500)
    compare = image_processing.image_resize(img1.copy(), width=500)
    pre_proc_img1 = image_processing.pre_processing(img1)
    pre_proc_img2 = image_processing.pre_processing(img2)

    contours, thresh = image_processing.find_changes(pre_proc_img1, pre_proc_img2)

    for contour in contours:
        print(contour)
        if cv2.contourArea(contour) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (128, 255, 147), 2)

    cv2.namedWindow("Image 1", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Image 1", pre_proc_img1)

    cv2.namedWindow("Image 2", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Image 2", pre_proc_img2)

    cv2.namedWindow("Compare Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Compare Image", thresh)

    cv2.namedWindow("Output Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Output Image", output)

    cv2.waitKey(0)

    cv2.destroyWindow("Image 1")
    cv2.destroyWindow("Image 2")
    cv2.destroyWindow("Proc Image")


def main_video_compare():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", "-i1", help="path to the first image")
    parser.add_argument("--image2", "-i2", help="path to the second image")
    args = parser.parse_args()

    if args.image1 and args.image2:
        image1 = cv2.imread(args.image1, cv2.IMREAD_COLOR)
        image2 = cv2.imread(args.image2, cv2.IMREAD_COLOR)

        if np.shape(image1) == () or np.shape(image2) == ():
            # Failed Reading
            print("Image file could not be opened")
            exit(-1)
        main_image_compare(image1, image2)
    else:
        main_video_compare()
