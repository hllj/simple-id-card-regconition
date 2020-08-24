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


def evaluate(image, ground_truth):
    ex_image = cv.imread(image)
    resized = False
    if ex_image.shape[0] < 512 and ex_image.shape[1] < 512:
        ex_image = cv.resize(ex_image, (0, 0), fx=2, fy=2)
        resized = True
    contour_image = ex_image.copy()
    #cv.imshow("Original image", ex_image)
    card = ex_image
    improc = ImageProcessing(grayscale_form='luminance', thresholding='otsu')
    card_grayscale = improc.get_grayscale(card)
    enhance_image = improc.get_enhance_image(card_grayscale)
    textarea = TextArea(contour_image, card_grayscale, enhance_image)
    bounding_boxes, contour_image = textarea.get_contour_textarea()
    
    ground_truth_boxes = get_groundtruth_boundingbox(ground_truth, contour_image, resized)
    # Calculate Intersection Point
    point, cnt_highrate= calculate_boundingbox_point(ground_truth_boxes, bounding_boxes)
    print("{} have IoU = {}, high rate matched box : {}".format(image, point, cnt_highrate))
    with open('result_MIDV.csv', 'a+') as result:
        result.write('{},{},{}\n'.format(image.split('/')[-1], point, cnt_highrate))

if __name__ == '__main__':
    project_folder = os.getcwd()
    data_clear_images = os.path.join(project_folder, 'data_clear/images/')
    data_clear_groundtruth = os.path.join(project_folder, 'data_clear/ground_truth/')
    with open('result_MIDV.csv', 'w+') as result:
        result.write("image,avg_IoU,high_rate_matched_box\n")
    with os.scandir(data_clear_images) as itr:
        for entry in itr:
            if entry.is_file():
                image_file = os.path.join(data_clear_images, entry.name)
                ground_truth = entry.name.replace('images', 'ground_truth').replace('tif', 'json')
                ground_truth = os.path.join(data_clear_groundtruth, ground_truth)
                evaluate(image_file, ground_truth)
