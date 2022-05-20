import cv2
import numpy as np
import os

classes = []
with open("classes.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)

my_img = cv2.imread("kangaroo_1.jpg")

with open("kangaroo_1.txt", 'r') as f:
    boxes = [line.strip().split() for line in f.readlines()]

print(boxes)

ht, wt, _ = my_img.shape

def reg_image(ht, wt):
    class_name, x, y, w, h = boxes[-1]
    #print(ht, wt)
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    x *= wt
    y *= ht
    w *= wt
    h *= ht
    cent_x = int(x)
    cent_y = int(y)
    w = int(w)
    h = int(h)
    print(class_name, cent_x, cent_y, w, h)
    return class_name, cent_x, cent_y, w, h

def flip_image(image, boxes, direction="x"):
    class_name, x, y, w, h = boxes[-1]
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    if direction == "x":
        y = 1-y
        my_img = cv2.flip(image, 0)
    elif direction == "y":
        x = 1-x
        my_img = cv2.flip(image, 1)
    elif direction == "both":
        x = 1-x
        y = 1 - y
        one_way = cv2.flip(image, 0)
        my_img = cv2.flip(one_way, 1)
    else:
        print("Please enter an x, y, or both for the direction parameter")
    x *= wt
    y *= ht
    w *= wt
    h *= ht
    w = int(w)
    h = int(h)
    cent_x = int(x)
    cent_y = int(y)
    #print(class_name, cent_x, cent_y, w, h)
    return my_img, class_name, cent_x, cent_y, w, h

def rotate_image(image, boxes, rotation_value):
    class_name, x, y, w, h = boxes[-1]
    print(x, y, w, h, image.shape)
    if rotation_value == 270:
        my_img = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
        img_width, img_height, channels = image.shape
        my_img = cv2.resize(my_img, (img_width, img_height))
        x, y = y, x
        w, h = h, w
        ht, wt, _ = my_img.shape
    else:
        my_img = image
        print("Please enter a good rotation value")
    print(x, y, w, h, my_img.shape)
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    x *= wt
    y *= ht
    w *= wt
    h *= ht
    cent_x = int(x)
    cent_y = int(y)
    w = int(w)
    h = int(h)
    print(class_name, cent_y, cent_x, w, h, my_img.shape)
    return my_img, class_name, cent_y, cent_x, w, h


def yoloFormattocv(x1, y1, x2, y2, H, W):
    bbox_width = x2 * W
    bbox_height = y2 * H
    center_x = x1 * W
    center_y = y1 * H

    voc = []

    voc.append(center_x - (bbox_width / 2))
    voc.append(center_y - (bbox_height / 2))
    voc.append(center_x + (bbox_width / 2))
    voc.append(center_y + (bbox_height / 2))

    return [int(v) for v in voc]

# convert from opencv format to yolo format
# H,W is the image height and width
def cvFormattoYolo(corner, H, W):
    bbox_W = corner[3] - corner[1]
    bbox_H = corner[4] - corner[2]

    center_bbox_x = (corner[1] + corner[3]) / 2
    center_bbox_y = (corner[2] + corner[4]) / 2

    return corner[0], round(center_bbox_x / W, 6), round(center_bbox_y / H, 6), round(bbox_W / W, 6), round(bbox_H / H,6)

class yoloRotatebbox:
    def __init__(self, filename, image_ext, angle):
        """Takes in a Yolo image and rotates the image and boxes
            Args:
                filename: filename
                image_ext: image format
                angle: rotation angle
        """
        assert os.path.isfile(filename + image_ext)
        assert os.path.isfile(filename + '.txt')

        self.filename = filename
        self.image_ext = image_ext
        self.angle = angle

        # Read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)

        # create a 2D-rotation matrix
        rotation_angle = self.angle * np.pi / 180
        self.rot_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])

    def rotate_image(self):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        height, width = self.image.shape[:2]  # image shape has 3 dimensions
        image_center = (width / 2,
                        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origin) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(self.image, rotation_mat, (bound_w, bound_h))
        return rotated_mat, bound_w, bound_h

    def rotateYolobbox(self):

        new_height, new_width = self.rotate_image()[0].shape[:2]

        f = open(self.filename + '.txt', 'r')

        f1 = f.readlines()

        new_bbox = []

        green_bbox = []
        H, W = self.image.shape[:2]
        #print(H,W, new_height, new_width)
        for x in f1:
            bbox = x.strip('\n').split(' ')
            if len(bbox) > 1:
                (center_x, center_y, bbox_width, bbox_height) = yoloFormattocv(float(bbox[1]), float(bbox[2]),
                                                                               float(bbox[3]), float(bbox[4]), H, W)

                # shift the origin to the center of the image.
                upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
                upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
                lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
                lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

                new_lower_right_corner = [-1, -1]
                new_upper_left_corner = []

                for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift,
                          lower_right_corner_shift):
                    new_coords = np.matmul(self.rot_matrix, np.array((i[0], -i[1])))
                    green_bbox.append(new_coords)
                    x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                    if new_lower_right_corner[0] < x_prime:
                        new_lower_right_corner[0] = x_prime
                    if new_lower_right_corner[1] < y_prime:
                        new_lower_right_corner[1] = y_prime

                    if len(new_upper_left_corner) > 0:
                        if new_upper_left_corner[0] > x_prime:
                            new_upper_left_corner[0] = x_prime
                        if new_upper_left_corner[1] > y_prime:
                            new_upper_left_corner[1] = y_prime
                    else:
                        new_upper_left_corner.append(x_prime)
                        new_upper_left_corner.append(y_prime)
                #             print(x_prime, y_prime)

                new_bbox.append([bbox[0], new_upper_left_corner[0], new_upper_left_corner[1],
                                 new_lower_right_corner[0], new_lower_right_corner[1]])

        return new_bbox, green_bbox



font = cv2.FONT_HERSHEY_PLAIN

angle = 5

my_img = yoloRotatebbox("kangaroo_1", ".jpg", angle)
new_mat, bound_w, bound_h = my_img.rotate_image()
print(bound_w, bound_h)
temp, green_bbox = my_img.rotateYolobbox()
print(green_bbox)
for i in temp:
    class_name = i[0]
    upper_left_x = int(i[1])
    upper_left_y = int(i[2])
    lower_right_x = int(i[3])
    lower_right_y = int(i[4])

upper_left_g, upper_right_g, lower_left_g, lower_right_g = green_bbox
upper_left_g[0] = upper_left_g[0] + (bound_w/2)
upper_left_g[1] = (0-upper_left_g[1]) + (bound_h/2)
upper_right_g[0] = upper_right_g[0] + (bound_w/2)
upper_right_g[1] = (0-upper_right_g[1]) + (bound_h/2)
lower_left_g[0] = lower_left_g[0] + (bound_w/2)
lower_left_g[1] = (0-lower_left_g[1]) + (bound_h/2)
lower_right_g[0] = lower_right_g[0] + (bound_w/2)
lower_right_g[1] = (0-lower_right_g[1]) + (bound_h/2)

print(upper_left_g, upper_right_g, lower_left_g, lower_right_g)

#class_name, cent_x, cent_y, w, h = reg_image()


cv2.rectangle(new_mat, (int(upper_left_x), int(upper_left_y)), (int(lower_right_x), int(lower_right_y)), (255,0,0), 2)
cv2.putText(new_mat, "ORIGINAl", (upper_left_x, upper_left_y-10), font, 2, (255,0,0), 2)




height, width, _ = new_mat.shape
class_name, cent_x, cent_y, w, h = reg_image(height, width)
cv2.rectangle(new_mat, (cent_x-w//2, cent_y-h//2), (cent_x+w//2, cent_y+h//2), (0,0,255), 2)
cv2.putText(new_mat, "CONTAINER", (cent_x+w//2, cent_y+h//2-10), font, 2, (0,0,255), 2)


cv2.rectangle(new_mat, (int(upper_left_g[0]), int(upper_left_g[1])), (int(lower_right_g[0]), int(lower_right_g[1])), (0,56,0), 2)
#cv2.putText(new_mat, "ROTATED", (cent_x+w//2, cent_y+h//2-10), font, 2, (0,255,0), 2)

box = cv2.boxPoints(((cent_x, cent_y), (w, h), 360-angle))
box = np.int0(box)
cv2.drawContours(new_mat, [box], 0, (0,255,0), 2)

#cv2.line(new_mat, (width//2, 0), (width//2, height), (255,255,255), 4)

cv2.imshow("img", new_mat)
cv2.waitKey(0)
cv2.destroyWindow()
