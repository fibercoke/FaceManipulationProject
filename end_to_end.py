from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf

from chars_cut import cut
from chars_detect import detect
from chars_model import CModel
from plate_detect import detect_and_cut_single_img
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
import yolov3_tf2.dataset_chars as dataset
import os
import matplotlib.pyplot as plt

flags.DEFINE_string('image_path', './data/Plate_dataset/AC/test/jpeg/88.jpg', 'path to input image')
flags.DEFINE_string('classifier_classes', './data/chars_data.names', 'path to classes file')
flags.DEFINE_string('detection_weights', './tiny/yolov3_tiny_train_230.tf', 'path to weights file')
flags.DEFINE_string('classifier_weights', './chars_checkpoints/resnet101v2_train_98.tf', 'path to weights file')
flags.DEFINE_integer('detection_size', 416, 'resize images to')
flags.DEFINE_integer('classifier_size', 32, 'size of each character should be resize to')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    model = YoloV3Tiny(size=FLAGS.detection_size, classes=1)
    model.load_weights(FLAGS.detection_weights).expect_partial()
    logging.info('yolo weights loaded')

    filename = FLAGS.image_path

    img_raw = tf.image.decode_image(open(FLAGS.image_path, 'rb').read(), channels=3)
    img_cut_plate = detect_and_cut_single_img(img_raw, model, W_delta=0, H_delta=0)
    if img_cut_plate is None:
        print('`%s` can not detect!' % filename)
        pass

    plt.imshow(img_cut_plate)
    plt.show()

    chrs = cut(img_cut_plate)

    for idx, img in enumerate(chrs):
        plt.subplot(2, 3, idx+1).imshow(img)
        pass
    plt.show()

    chrs = [tf.image.resize_with_pad(tf.image.grayscale_to_rgb(tf.convert_to_tensor(np.expand_dims(ch, axis=2))),
                                     32, 32, antialias=False) for ch in chrs]

    class_map_inv = {idx:name for idx, name in enumerate(open(FLAGS.classifier_classes).read().splitlines())}
    class_num = len(class_map_inv)

    logging.info('stacking')
    chrs = tf.stack([dataset.transform_images(img, FLAGS.classifier_size) for img in chrs])

    for idx, img in enumerate(chrs):
        plt.subplot(2, 3, idx+1).imshow(img.numpy())
        pass
    plt.show()

    model = CModel(size=FLAGS.classifier_size, class_num=class_num, weights=None)
    model.load_weights(FLAGS.classifier_weights).expect_partial()

    logging.info('classifying')
    lst = [class_map_inv[i.numpy()] for i in detect(chrs, model)]
    print(lst)




if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
