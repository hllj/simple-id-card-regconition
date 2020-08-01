from dataloader import DATA_FOLDER
import json
import cv2 as cv
import numpy as np

class DataLoader:
    def __init__(self):
        f = open(DATA_FOLDER + 'midv500_coco.json')
        self.data = json.load(f)

    def get_images_data(self):
        images = self.data["images"]
        print(type(images))
        return images

    def show_data(self):
        print("Show key in data")
        for key, value in self.data.items():
            print("Key : ", key)

def get_ground_truth_dir(image_dir):
    ground_truth_dir = image_dir.replace('images', 'ground_truth').replace('tif', 'json')
    return ground_truth_dir

class ImageLoader:
    def __init__(self, image_object):
        self.image_object = image_object
        folder_image = self.image_object['file_name'].split('/')[0]
        #self.image_dir = DATA_FOLDER + self.image_object['file_name']
        self.image_dir = DATA_FOLDER + folder_image + '/images/' + folder_image+'.tif'
        self.image = cv.imread(self.image_dir)
        self.width = self.image_object['width']
        self.height = self.image_object['height']
        self.id = self.image_object['id']
    def get_quad(self):
        ground_truth_dir = get_ground_truth_dir(self.image_dir)
        with open(ground_truth_dir) as ground_truth_file:
            data = json.load(ground_truth_file)
            return data['quad']
    def get_card_image(self):
        quad = np.array([self.get_quad()])
        print(quad)
        rect = cv.minAreaRect(quad)
        print(rect)
        center, size, angle = rect[0], rect[1], rect[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))
        M = cv.getRotationMatrix2D(center, angle, 1)
        img_rot = cv.warpAffine(self.image, M, (self.width, self.height))
        crop_width, crop_height = size[0], size[1]
        if crop_width < crop_height:
            M = cv.getRotationMatrix2D(center, 90, 1.0)
            img_rot = cv.warpAffine(img_rot, M, (self.width, self.height), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
            size = (crop_height, crop_width)

        img_crop = cv.getRectSubPix(img_rot, size, center)

        return img_crop

    def show_image(self):
        print(self.image_dir)
        cv.imshow('Example image', self.image) 
