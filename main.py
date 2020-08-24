import numpy as np
import cv2 as cv
from dataloader.dataloader import DataLoader
from dataloader.dataloader import ImageLoader
from imageprocessing.imageprocessing import ImageProcessing
from imageprocessing.textarea import TextArea
from imageprocessing.textdetector import TextDetector

if __name__ == "__main__":
    dl = DataLoader()
    dl.show_data()
    images_list = dl.get_images_data()
    ex_image = ImageLoader(images_list[8000])
    ex_image.show_image()
    contour_image = ex_image.image.copy()
    card = ex_image.image
    cv.imshow('Card image', card)
    improc = ImageProcessing(grayscale_form='luminance', thresholding='otsu')
    card_grayscale = improc.get_grayscale(card)
    enhance_image = improc.get_enhance_image(card_grayscale)
    textarea = TextArea(contour_image, card_grayscale, enhance_image)
    bounding_boxes, contour_image = textarea.get_contour_textarea()
    textdetector = TextDetector(bounding_boxes, card)
    textdetector.detect_all()
    cv.imshow('Enhance', enhance_image)
    cv.imshow('Contour', contour_image)
    while True:
        k = cv.waitKey(0) & 0xFF
        if k == ord('q'):
            break
    cv.destroyAllWindows()
