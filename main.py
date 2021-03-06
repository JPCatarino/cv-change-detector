# Python Change Detector using Open CV

import cv2
import argparse

import numpy as np

from utils import image_processing


def main_image_compare(img1, img2):
    output = image_processing.image_resize(img2.copy(), width=500)
    compare = image_processing.image_resize(img1.copy(), width=500)
    pre_proc_img1 = image_processing.pre_processing(img1, blur_ksize=(15, 15))
    pre_proc_img2 = image_processing.pre_processing(img2, blur_ksize=(15, 15))

    contours, thresh = image_processing.find_changes(pre_proc_img1, pre_proc_img2, cv2.THRESH_OTSU)

    image_processing.handle_contours(contours, output, args.area)

    cv2.namedWindow("Image 1", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Image 1", pre_proc_img1)

    cv2.namedWindow("Image 2", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Image 2", pre_proc_img2)

    cv2.namedWindow("Compare Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Compare Image", compare)

    cv2.namedWindow("Output Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Output Image", output)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def main_video_compare():
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        # Skip a few frames for camera to be stable
        for i in range(60):
            val, frame = vc.read()
        val, background = vc.read()
        pre_proc_back = image_processing.pre_processing(background, blur_ksize=(15, 15))
    else:
        val = False

    while val:
        val, frame = vc.read()
        output = image_processing.image_resize(frame.copy(), width=500)
        pre_proc_fore = image_processing.pre_processing(frame, blur_ksize=(15, 15))

        contours, thresh = image_processing.find_changes(pre_proc_back, pre_proc_fore, cv2.THRESH_OTSU)

        image_processing.handle_contours(contours, output, args.area)

        cv2.imshow("Cam", output)
        cv2.imshow("thresh", thresh)
        cv2.imshow("back", pre_proc_back)

        key = cv2.waitKey(20)

        if key == 27:
            break
        elif key == ord('b'):
            val, background = vc.read()
            pre_proc_back = image_processing.pre_processing(background, blur_ksize=(15, 15))

    vc.release()
    cv2.destroyAllWindows()


def mog_video_compare():
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        for i in range(60):
            val, frame = vc.read()
        val, background = vc.read()
        fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        pre_proc_back = image_processing.pre_processing(background, blur_ksize=(15, 15))
        fgbg.apply(pre_proc_back)
        learning_rate = 0.0009
    else:
        val = False

    while val:
        val, frame = vc.read()
        output = image_processing.image_resize(frame, width=500)
        pre_proc_fore = image_processing.pre_processing(frame, blur_ksize=(15, 15))
        mask = fgbg.apply(pre_proc_fore, learningRate=learning_rate)

        cleaned_mask = image_processing.clean_noise(mask.copy())
        contours = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        image_processing.handle_contours(contours, output, args.area)

        cv2.imshow("Cam", output)
        cv2.imshow("Mask", cleaned_mask)

        key = cv2.waitKey(20)

        if key == 27:
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", "-i1", help="path to the first image")
    parser.add_argument("--image2", "-i2", help="path to the second image")
    parser.add_argument("--area", '-a', type=int, default=3000, help='minimum area for intruder identification')
    parser.add_argument("--mog", '-m', help="Use MOG Background Subtractor")
    args = parser.parse_args()

    if args.image1 and args.image2:
        image1 = cv2.imread(args.image1, cv2.IMREAD_COLOR)
        image2 = cv2.imread(args.image2, cv2.IMREAD_COLOR)

        if np.shape(image1) == () or np.shape(image2) == ():
            # Failed Reading
            print("Image file could not be opened")
            exit(-1)
        main_image_compare(image1, image2)
    elif args.mog:
        mog_video_compare()
    else:
        main_video_compare()
