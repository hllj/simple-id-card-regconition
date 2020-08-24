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
                if (BoundingBox.straight_on_row(bounding_boxes[i], bounding_boxes[j]) == True):
                    djset.union(bounding_boxes[i], bounding_boxes[j])
        for i in range(len(bounding_boxes)):
            for j in range(i + 1, len(bounding_boxes)):
                if (BoundingBox.is_intersect(bounding_boxes[i], bounding_boxes[j]) == True):
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
        return joint_boundingboxes, contour_image

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
        if (abs(bb1.x + bb1.w - bb2.x) <= 15 or abs(bb2.x + bb2.w - bb1.x) <= 15):
            if (bb1.y <= bb2.y <= bb1.y + bb1.h): 
                return True
            if (bb2.y <= bb1.y <= bb2.y + bb2.h):
                return True
        return False
    @staticmethod
    def is_intersect(bb1, bb2):
        r1_left = bb1.x
        r1_right = bb1.x + bb1.w
        r1_bottom = bb1.y
        r1_top = bb1.y + bb1.h
        r2_left = bb2.x
        r2_right = bb2.x + bb2.w
        r2_bottom = bb2.y
        r2_top = bb2.y + bb2.h
        return not(r2_left > r1_right or r2_right < r1_left or r2_top < r1_bottom or r2_bottom > r1_top)
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
