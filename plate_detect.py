from absl import flags
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf

from yolov3_tf2.dataset_plate import transform_images

flags.DEFINE_integer('size', 416, 'resize images to')
#flags.DEFINE_string('output_path', './data/plate_uncut/', 'path to output images')


def cut_plate(img, x1, y1, x2, y2):
    siz = (int(x2 - x1)*10, int(y2 - y1)*10)
    std = np.array([[0, 0], [siz[0], 0], [siz[0], siz[1] - 1], [0, siz[1] - 1]], dtype=np.float32)
    pos = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    warped = cv2.getPerspectiveTransform(src=pos, dst=std)
    result = cv2.warpPerspective(img.numpy(), warped, siz)
    return result


def detect_single_img(img_raw, yolo):
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)
    boxes1, scores1, classes1, nums1 = yolo(img)

    img_raw_crop = img_raw[16:-24]
    img = tf.expand_dims(img_raw_crop, 0)
    img = transform_images(img, FLAGS.size)
    boxes2, scores2, classes2, nums2 = yolo(img)

    if nums1[0] == 0 and nums2[0] == 0:
        return None, None

    if nums1[0] == 0:
        return img_raw_crop, boxes2
    if nums2[0] == 0:
        return img_raw, boxes1

    if scores1[0][0] > scores2[0][0]:
        return img_raw, boxes1
    else:
        return img_raw_crop, boxes2


def get_xy_tensor_from_boxes(img, boxes, W_delta, H_delta):
    H, W, _ = img.shape
    return boxes[0][0].numpy() * np.array([W,H,W,H]) + np.array([-W_delta, -H_delta, W_delta, H_delta])


def detect_and_cut_single_img(img_raw, yolo, W_delta=0, H_delta=0):
    img, boxes = detect_single_img(img_raw, yolo)
    if boxes is None:
        return None
    x1, y1, x2, y2 = get_xy_tensor_from_boxes(img, boxes, W_delta, H_delta)
    img_cut_plate = cut_plate(img, x1, y1, x2, y2)
    return img_cut_plate