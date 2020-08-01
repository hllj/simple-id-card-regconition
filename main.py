import numpy as np
import cv2 as cv
from dataloader.dataloader import DataLoader
from dataloader.dataloader import ImageLoader
from imageprocessing.imageprocessing import ImageProcessing

if __name__ == "__main__":
    dl = DataLoader()
    dl.show_data()
    images_list = dl.get_images_data()
    ex_image = ImageLoader(images_list[4000])
    ex_image.show_image()
    contour_image = ex_image.image.copy()
    card = ex_image.image
    cv.imshow('Card image', card)
    improc = ImageProcessing(grayscale_form='luminance', thresholding='otsu')
    card = improc.get_grayscale(card)
    cv.imshow('Grayscale', card)
    enhance_image = improc.get_enhance_image(card)
    contours, hierachy = cv.findContours(enhance_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(len(contours))
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        cv.rectangle(contour_image, (x,y), (x+w,y+h), (0,255,0), 2)
    cv.imshow('Enhance', enhance_image)
    cv.imshow('Contour', contour_image)
    while True:
        k = cv.waitKey(0) & 0xFF
        if k == ord('q'):
            break
    cv.destroyAllWindows()
