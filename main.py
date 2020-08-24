import numpy as np
import cv2 as cv
from dataloader.dataloader import DataLoader
from dataloader.dataloader import ImageLoader
from imageprocessing.imageprocessing import ImageProcessing
from imageprocessing.textarea import TextArea
from imageprocessing.textdetector import TextDetector

import argparse
import os
from utils import get_groundtruth_boundingbox, calculate_boundingbox_point

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to image in data_clear")
    ap.add_argument("-g", "--ground_truth", required=False, help="path to ground truth of image in data_clear")
    args = vars(ap.parse_args())
    project_folder = os.getcwd()
    ex_image = cv.imread(os.path.join(project_folder, args["image"]))
    resized = False
    if ex_image.shape[0] < 512 and ex_image.shape[1] < 512:
        ex_image = cv.resize(ex_image, (0, 0), fx=2, fy=2)
        resized = True
    contour_image = ex_image.copy()
    cv.imshow("Original image", ex_image)
    card = ex_image
    improc = ImageProcessing(grayscale_form='luminance', thresholding='otsu')
    card_grayscale = improc.get_grayscale(card)
    enhance_image = improc.get_enhance_image(card_grayscale)
    textarea = TextArea(contour_image, card_grayscale, enhance_image)
    bounding_boxes, contour_image = textarea.get_contour_textarea()
    cv.imshow('Predicted Text area', contour_image)
    cv.imshow('Enhance', enhance_image)
    # Compare to ground truth
    if args["ground_truth"] is not None:
        ground_truth_boxes = get_groundtruth_boundingbox(args["ground_truth"], contour_image, resized)
        cv.imshow('Predicted vs Ground truth', contour_image)
        # Calculate Intersection Point
        point, cnt_highrate= calculate_boundingbox_point(ground_truth_boxes, bounding_boxes)
        print('Average point', point)
        print('Bounding box high rate matched', cnt_highrate)
        textdetector = TextDetector(bounding_boxes, card)
        textdetector.detect_all()

    while True:
        k = cv.waitKey(0) & 0xFF
        if k == ord('q'):
            break
    cv.destroyAllWindows()
