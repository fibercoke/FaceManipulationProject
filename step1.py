import tqdm
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf

from plate_detect import detect_single_img, get_xy_tensor_from_boxes
from tools.dataset_utils import generate_xml
from yolov3_tf2.models import (
    YoloV3Tiny
)
import os


#flags.DEFINE_string('output_path', './data/plate_uncut/', 'path to output images')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if not os.path.exists(FLAGS.xml_root_path):
        os.mkdir(FLAGS.xml_root_path)

    yolo = YoloV3Tiny(classes=1)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')


    input_images = list(file for file in os.listdir(FLAGS.image_root_path) if file.endswith('.jpg'))
    total_None = 0
    for filename in tqdm.tqdm(input_images):
        img_raw = tf.image.decode_image(open(os.path.join(FLAGS.image_root_path, filename), 'rb').read(), channels=3)
        img, boxes = detect_single_img(img_raw, yolo)
        if boxes is None:
             print('`%s` can not detect!' % filename)
             total_None += 1
             x1, y1, x2, y2 = 0,0,img_raw.shape[0],img_raw.shape[1]
        else:
            x1, y1, x2, y2 = np.round(get_xy_tensor_from_boxes(img, boxes, 0, 0)).astype(int)
            pass

        xml_str = generate_xml(filename=filename, xy=(x1, y1, x2, y2))
        with open(os.path.join(FLAGS.xml_root_path, filename.replace('jpg', 'xml')), 'wb') as fp:
            fp.write(xml_str)
        # img_cut_plate = detect_and_cut_single_img(img_raw, yolo, W_delta=0, H_delta=0)
        # if img_cut_plate is None:
        #     print('`%s` can not detect!' % filename)
        #     total_None += 1
        # else:
        #     tf.io.write_file(os.path.join(FLAGS.output_path, filename), tf.image.encode_jpeg(img_cut_plate))
        #     pass
        # pass
    print('total None:', total_None)


if __name__ == '__main__':
    flags.DEFINE_string('weights', './tiny/yolov3_tiny_train_230.tf',
                        'path to weights file')
    flags.DEFINE_string('image_root_path', './data/Plate_dataset/AC/test/jpeg/', 'path to input images')
    flags.DEFINE_string('xml_root_path', './data/Plate_dataset/AC/test/xml_pred/', 'path to output xmls')
    try:
        app.run(main)
    except SystemExit:
        pass