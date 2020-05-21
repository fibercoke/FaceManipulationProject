import lxml
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from chars_cut import cut
from chars_detect import detect
from chars_model import CModel
from generate_dataset_plate import parse_xml
from plate_detect import detect_and_cut_single_img, cut_plate
from tools.dataset_utils import generate_xml
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
import yolov3_tf2.dataset_chars as dataset
import os
import matplotlib.pyplot as plt


def main(_argv):
    class_map_inv = {idx: name for idx, name in enumerate(open(FLAGS.classifier_classes).read().splitlines())}
    class_num = len(class_map_inv)

    if FLAGS.save_image:
        if not os.path.exists(FLAGS.save_image_root):
            os.mkdir(FLAGS.save_image_root)
            pass
        for cls in class_map_inv.values():
            path = os.path.join(FLAGS.save_image_root, cls)
            if not os.path.exists(path):
                os.mkdir(path)

    if not FLAGS.save_image:
        model = CModel(size=FLAGS.classifier_size, class_num=class_num, weights=None)
        model.load_weights(FLAGS.classifier_weights).expect_partial()

    acc = 0
    tol = 0
    for xml in tqdm(filename for filename in os.listdir(FLAGS.xml_in_root_path) if filename.endswith('.xml')):
        logging.info(xml)
        annotation_xml = lxml.etree.fromstring(open(os.path.join(FLAGS.xml_in_root_path, xml)).read())
        annotation = parse_xml(annotation_xml)['annotation']
        img_file_name = annotation['filename']
        img_path = os.path.join(FLAGS.image_root_path, img_file_name)
        img_raw = tf.image.decode_jpeg(open(img_path, 'rb').read())
        xmin = int(annotation['object'][0]['bndbox']['xmin'])
        ymin = int(annotation['object'][0]['bndbox']['ymin'])
        xmax = int(annotation['object'][0]['bndbox']['xmax'])
        ymax = int(annotation['object'][0]['bndbox']['ymax'])
        img_cut_plate = cut_plate(img_raw, xmin, ymin, xmax, ymax)
        chrs = cut(img_cut_plate)

        #if FLAGS.painting:
        #    plt.imshow(img_cut_plate)
        #    plt.show()
        #    for idx, img in enumerate(chrs):
        #        plt.subplot(2, 3, idx+1).imshow(img)
        #        pass
        #    plt.show()

        chrs = [tf.image.resize_with_pad(tf.image.grayscale_to_rgb(tf.convert_to_tensor(np.expand_dims(ch, axis=2))),
                                         FLAGS.classifier_size, FLAGS.classifier_size, antialias=False) for ch in chrs]

        if FLAGS.save_image:
            gt = annotation['object'][0]['platetext']
            for idx, (ch, label) in enumerate(zip(chrs, gt)):
                path = os.path.join(FLAGS.save_image_root, label, '%s-%d.jpg' % (img_file_name[:-4], idx))
                tf.cast(ch, dtype=tf.uint8)
                tf.io.write_file(path,
                                 tf.image.encode_jpeg(
                                        tf.cast(ch, dtype=tf.uint8),
                                 )
                                 )




        #if FLAGS.painting:
        #    for idx, img in enumerate(chrs):
        #        plt.subplot(2, 3, idx+1).imshow(img.numpy())
        #        pass
        #    plt.show()
        if not FLAGS.save_image:
            logging.info('stacking')
            chrs = tf.stack([dataset.transform_images(img, FLAGS.classifier_size) for img in chrs])
            logging.info('classifying')
            lst = [class_map_inv[i.numpy()] for i in detect(chrs, model)]
            plate_text = "".join(lst)
            print("gt:", annotation['object'][0]['platetext'])
            print("pred:", plate_text)
            tol += 1
            if plate_text == annotation['object'][0]['platetext']:
                acc += 1
            xml_str = generate_xml(filename=img_file_name, plate_text=plate_text)
            with open(os.path.join(FLAGS.xml_out_root_path, img_file_name.replace('jpg', 'xml')), 'wb') as fp:
                fp.write(xml_str)
            print('acc:', acc / tol)





if __name__ == '__main__':
    flags.DEFINE_string('image_root_path', './data/Plate_dataset/AC/test/jpeg/', 'path to input image')
    flags.DEFINE_string('xml_in_root_path', './data/Plate_dataset/AC/test/xml/', 'path to input xml')
    flags.DEFINE_string('xml_out_root_path', './data/Plate_dataset/AC/test/xml_pred/', 'path to output xml')
    flags.DEFINE_string('classifier_classes', './data/chars_data.names', 'path to classes file')
    flags.DEFINE_string('classifier_weights', './chars_checkpoints/new_my_resnet101v2_train_acc_0.8614_19_2020-05-21-235521.tf', 'path to weights file')
    flags.DEFINE_integer('classifier_size', 32, 'size of each character should be resize to')
    flags.DEFINE_bool('painting', False, 'if plt.show()')
    flags.DEFINE_bool('save_image', False, 'should save image for training?')
    flags.DEFINE_string('save_image_root', './data/new_my_chars_test_data/', 'saved image path for training')
    try:
        app.run(main)
    except SystemExit:
        pass
