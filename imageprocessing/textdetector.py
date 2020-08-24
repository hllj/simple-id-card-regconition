import numpy as np
import cv2 as cv
import pytesseract
from imageprocessing.imageprocessing import ImageProcessing
from PIL import Image


class TextDetector():
    def __init__(self, bounding_boxes, original):
        self.bounding_boxes = bounding_boxes
        self.original = original
    def detect_all(self):
        grayscale = cv.cvtColor(self.original, cv.COLOR_BGR2GRAY)
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box.get_values()
            ROI = grayscale[y:y+h, x:x+w]
            #ROI = self.preprocessing(ROI)
            #cv.imshow('ROI {} {} {} {}'.format(x, y, w, h), ROI)
            string = pytesseract.image_to_string(ROI)
            if string != '':
                print(string)

