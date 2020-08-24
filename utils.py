import numpy as np
import cv2 as cv
import os
import json
from imageprocessing.textarea import BoundingBox

def get_x_y_w_h(bounding_box_quad):
    x = bounding_box_quad[0][0]
    y = bounding_box_quad[0][1]
    w = bounding_box_quad[1][0] - x
    h = bounding_box_quad[3][1] - y
    return x, y, w, h

def get_groundtruth_boundingbox(ground_truth_filename, image, resized):
    project_folder = os.getcwd()
    f = open(os.path.join(project_folder, ground_truth_filename))
    data = json.load(f)
    ground_truth_boxes = []
    for key, value in data.items():
        bounding_box_quad = value['quad']
        #bounding_box_value = value['value']
        x, y, w, h = get_x_y_w_h(bounding_box_quad)
        if resized == True:
            x, y, w, h = x * 2, y * 2, w * 2, h * 2
        ground_truth_boxes.append(BoundingBox(x, y, w, h))
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return ground_truth_boxes

def calculate_boundingbox_point(ground_truth_boxes, bounding_boxes):
    point = 0.0
    cnt_highrate = 0
    cnt_match = 0
    for gbox in ground_truth_boxes:
        for box in bounding_boxes:
            if BoundingBox.is_intersect(gbox, box):
                IoU = BoundingBox.bb_IoU(gbox, box)
                point += IoU
                cnt_match += 1
                if IoU >= 0.5:
                    cnt_highrate += 1
    print('High rate matched bounding box count', cnt_highrate)
    return point / (1.0 * cnt_match), cnt_highrate 
