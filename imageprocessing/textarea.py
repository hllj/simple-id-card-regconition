import numpy as np
import cv2 as cv

class TextArea():
    def __init__(self, original, grayscale, enhance_image):
        self.grayscale = grayscale
        self.enhance_image = enhance_image
        self.original = original

    def get_contour_textarea(self):
        contours, hierachy = cv.findContours(self.enhance_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        bounding_boxes = []
        for contour in contours:
            (x, y, w, h) = cv.boundingRect(contour)
            new_boundingbox = BoundingBox(x, y, w, h)
            bounding_boxes.append(new_boundingbox)

        bounding_boxes.sort(key=BoundingBox.bboxsort)
        bounding_boxes_parent = {}
        for bbox in bounding_boxes:
            bounding_boxes_parent[bbox] = bbox

        djset = DisjointSet(bounding_boxes, bounding_boxes_parent)
        for i in range(len(bounding_boxes)):
            for j in range(i + 1, len(bounding_boxes)):
                if BoundingBox.straight_on_row(bounding_boxes[i], bounding_boxes[j]) == True:
                    djset.union(bounding_boxes[i], bounding_boxes[j])
        is_boundingbox_parent = {}
        for bbox in bounding_boxes:
            is_boundingbox_parent[bbox] = True
        for bbox in bounding_boxes:
            parent = djset.find(bbox)
            if (bbox != parent):
                is_boundingbox_parent[parent] = True
                is_boundingbox_parent[bbox] = False
                parent.joinbb(bbox)
        joint_boundingboxes = []
        for bbox in bounding_boxes:
            if is_boundingbox_parent[bbox] == True:
                joint_boundingboxes.append(bbox)
        contour_image = self.original.copy()
        print(len(joint_boundingboxes))
        for bounding_box in joint_boundingboxes:
            x, y, w, h = bounding_box.get_values()
            cv.rectangle(contour_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.imshow('rect {} {} {} {}'.format(x, y, w, h), self.original[y:y + h, x:x+w])
        return contours, contour_image

class BoundingBox():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def get_values(self):
        return self.x, self.y, self.w, self.h
    def joinbb(self, bbox):
        newx = min(self.x, bbox.x)
        newy = min(self.y, bbox.y)
        bottomx = max(self.x + self.w, bbox.x + bbox.w)
        bottomy = max(self.y + self.h, bbox.y + bbox.h)
        neww = bottomx - newx
        newh = bottomy - newy
        self.x = newx
        self.y = newy
        self.w = neww
        self.h = newh
        return self
    @staticmethod
    def bboxsort(bb):
        return (bb.y, bb.x)
    @staticmethod
    def straight_on_row(bb1, bb2):
        if (abs(bb1.x + bb1.w - bb2.x) <= 30 or abs(bb2.x + bb2.w - bb1.x) <= 30):
            if (bb1.y <= bb2.y <= bb1.y + bb1.h): 
                return True
            if (bb2.y <= bb1.y <= bb2.y + bb2.h):
                return True
        return False

class DisjointSet():
    def __init__(self, vertices, parent):
        self.vertices = vertices
        self.parent = parent
    def find(self, item):
        if self.parent[item] == item:
            return item
        else:
            return self.find(self.parent[item])
    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        self.parent[root1] = root2
