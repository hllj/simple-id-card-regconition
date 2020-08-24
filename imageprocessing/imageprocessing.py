import numpy as np
import cv2 as cv

class ImageProcessing:
    def __init__(self, grayscale_form='luminance', thresholding='otsu'):
        self.grayscale_form = grayscale_form
        self.thresholding = thresholding
        self.kernel_5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5), (1, 1))
        self.kernel_3= cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3), (1, 1))

    def scale_norm_image(self, image):
        return (image - image.min()) / (image.max() - image.min())

    def convert_float2int(self, image):
        return np.uint8(image * 255)

    def get_grayscale(self, image):
        if len(image.shape) == 2:
            return image
        if len(image.shape) != 3:
            return None
        if self.grayscale_form == 'luminance':
            luminance_vector = np.array([[0.114, 0.587, 0.299]]).T
            luminance_image = np.dot(image, luminance_vector)
            luminance_image = self.scale_norm_image(luminance_image)
            luminance_image = self.convert_float2int(luminance_image)
            return luminance_image

    def get_erosion_image(self, image, kernel, iterations):
        erosion_image = cv.erode(image, kernel, iterations=iterations)
        return erosion_image
    def get_dilation_image(self, image, kernel, iterations):
        dilation_image = cv.dilate(image, kernel, iterations=iterations)
        return dilation_image
    def get_opening_image(self, image, kernel):
        opening_image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
        return opening_image
    def get_closing_image(self, image, kernel):
        closing_image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
        return closing_image
    def get_tophat_opening(self, image, kernel):
        opening_image = self.get_opening_image(image, kernel)
        oth = image - opening_image
        oth = self.scale_norm_image(oth)
        oth = self.convert_float2int(oth)
        return oth
    def get_tophat_closing(self, image, kernel):
        closing_image = self.get_closing_image(image, kernel)
        cth = closing_image - image
        cth = self.scale_norm_image(cth)
        cth = self.convert_float2int(cth)
        return cth
    def get_enhance_image(self, image):
        res = image
        #Thresholding T = 100
        ret, res = cv.threshold(image, 90, 255, cv.THRESH_BINARY_INV)
        #cv.imshow('Pre Threshold', res)
        low_pass_kernel = np.ones((5, 5), np.float32) / 25.0
        res = cv.filter2D(res, -1, low_pass_kernel)
        #cv.imshow('Low pass filter', res)
        #Morphology transformation
        res = self.get_dilation_image(res, self.kernel_5, 2)
        res = self.get_closing_image(res, self.kernel_5)
        res = self.scale_norm_image(res)
        res = self.convert_float2int(res)
        #Otsu
        ret, res = cv.threshold(res, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return res 
