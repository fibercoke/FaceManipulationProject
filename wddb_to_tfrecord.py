import os.path
import io
import glob
import math
import random

import PIL.Image
import PIL.ImageDraw
import tensorflow as tf

from object_detection.utils import dataset_util

flags = tf.compat.v1.app.flags
flags.DEFINE_string('output_path_prefix', 'data/out', 'Path prefix for output TFRecords')

flags.DEFINE_string(
    'annonations_file', 'FDDB-folds/FDDB-fold-*-ellipseList.txt', 'Annonations file')

flags.DEFINE_string(
    'img_folder', './originalPics', 'The image folder where images are')

flags.DEFINE_string(
    'img_file_extension', 'jpg', 'The file extensions for image files')

flags.DEFINE_bool(
    'debug', False, 'Create a debug image that has the bounding box drawn')

flags.DEFINE_bool(
    'shuffle', False, 'Shuffle images before taking the train and validation division')

flags.DEFINE_float(
    'distribution', 0.75, 'Training / Validation distribution, where 0.0 - 1.0 is the training portion')

flags.DEFINE_integer(
    'debug_number', 0, 'Which image is used to create the debug image')

flags.DEFINE_string(
    'debug_image', 'debug.jpg', 'Debug image name if the script is run with debug flag')

FLAGS = flags.FLAGS


class BoundingBox:
    def __init__(self, major_axis, minor_axis, angle, x, y, _):
        self.major_axis = float(major_axis)
        self.minor_axis = float(minor_axis)
        self.angle = float(angle)
        self.x = float(x)
        self.y = float(y)

    def _get_xoffset(self):
        return max(
            abs(math.sin(self.angle) * self.minor_axis),
            abs(math.cos(self.angle) * self.major_axis))

    def _get_yoffset(self):
        return max(
            abs(math.cos(self.angle) * self.minor_axis),
            abs(math.sin(self.angle) * self.major_axis))

    def get_xmin(self):
        x_offset = self._get_xoffset()
        return self.x - x_offset

    def get_xmax(self):
        x_offset = self._get_xoffset()
        return self.x + x_offset

    def get_ymin(self):
        y_offset = self._get_yoffset()
        return self.y - y_offset

    def get_ymax(self):
        y_offset = self._get_yoffset()
        return self.y + y_offset

    def print_coordinates(self, height=0, width=0):
        coords = 'xmin, ymin: ({}, {}) - xmax, ymax: ({}, {})'
        xmin = self.get_xmin()
        xmax = self.get_xmax()
        ymin = self.get_ymin()
        ymax = self.get_ymax()
        # normalize coordinates if width and height is given
        if width != 0 and height != 0:
            xmin = min(xmin / width, 1.0)
            xmax = min(xmax / width, 1.0)
            ymin = min(ymin / height, 1.0)
            ymax = min(ymax / height, 1.0)
        coords = coords.format(xmin, ymin, xmax, ymax)
        print(coords)


def parse_annotations_file(filename):
    """Returns a list of tuples (filename, [bounding_boxes])
  """
    assert os.path.isfile(filename), 'file not found {}'.format(filename)
    with open(filename, 'r') as f:
        result = []
        for line in f:
            image_file = line.strip()
            faces_number = int(f.readline())
            bounding_boxes = []
            for _ in range(faces_number):
                bounding_boxes.append(BoundingBox(*f.readline().split()))
            result.append((image_file, bounding_boxes))
        return result


def load_image(filename):
    img_folder = FLAGS.img_folder
    img_file_extension = FLAGS.img_file_extension
    img_filename = '{}.{}'.format(filename, img_file_extension)
    path = os.path.join(img_folder, img_filename)
    assert os.path.isfile(path), 'file not found {}'.format(path)
    return PIL.Image.open(path)


def create_tf_record(image_filename, bounding_boxes):
    img = load_image(image_filename)
    if img.format == 'JPEG':
        image_format = b'jpeg'
    elif img.format == 'PNG':
        image_format = b'jpg'
    else:
        raise AssertionError(
            'unsupported filetype {} - supported filetypes are jpeg and png'.format(img.format))
    width, height = img.size

    encoded_image_data = io.BytesIO()
    img.save(encoded_image_data, img.format)

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for bb in bounding_boxes:
        xmins.append(min(bb.get_xmin() / width, 1.0))
        xmaxs.append(min(bb.get_xmax() / width, 1.0))

        ymins.append(min(bb.get_ymin() / height, 1.0))
        ymaxs.append(min(bb.get_ymax() / height, 1.0))

        classes_text.append(b'face')
        classes.append(0)

    max_vals = [
        max(ymins),
        max(xmins),
        max(ymaxs),
        max(xmaxs),
    ]

    if any([v > 1.0 for v in max_vals]):
        print(max_vals)
        draw_bb(image_filename, bounding_boxes)
        raise AssertionError('Normalized coordinate over 1.0')

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_filename.encode()),
        'image/source_id': dataset_util.bytes_feature(image_filename.encode()),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data.getvalue()),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_record


def draw_bb(image_filename, bounding_boxes):
    src_img = load_image(image_filename)
    width, height = src_img.size
    draw = PIL.ImageDraw.Draw(src_img)
    for bb in bounding_boxes:
        draw.rectangle([bb.get_xmin(), bb.get_ymin(), bb.get_xmax(), bb.get_ymax()])
        bb.print_coordinates(height=height, width=width)
    src_img.save(FLAGS.debug_image, "JPEG")


def main(_):
    annonations_file = FLAGS.annonations_file
    if not annonations_file:
        raise ValueError('You must supply the annonations file with --annonations_file')

    file_glob = glob.glob(annonations_file)
    assert len(file_glob) != 0, 'no files with that pattern {}'.format(annonations_file)
    annonations = []
    for f in file_glob:
        annonations += parse_annotations_file(f)

    if FLAGS.debug:
        draw_bb(*annonations[FLAGS.debug_number])
        return

    print('Number of samples: {}'.format(len(annonations)))

    if FLAGS.shuffle:
        random.shuffle(annonations)

    training_set = int(round(FLAGS.distribution * len(annonations)))

    file_prefix = FLAGS.output_path_prefix

    max_box = 0

    if training_set > 0:
        writer = tf.io.TFRecordWriter(file_prefix + '-training.tfrecord')

        for (image_filename, bounding_boxes) in annonations[:training_set]:
            max_box = max(max_box, len(bounding_boxes))
            tf_record = create_tf_record(image_filename, bounding_boxes)
            writer.write(tf_record.SerializeToString())

        writer.close()

    validation_set = len(annonations) - training_set
    if validation_set > 0:
        writer = tf.io.TFRecordWriter(file_prefix + '-validation.tfrecord')

        for (image_filename, bounding_boxes) in annonations[training_set:]:
            max_box = max(max_box, len(bounding_boxes))
            tf_record = create_tf_record(image_filename, bounding_boxes)
            writer.write(tf_record.SerializeToString())

        writer.close()
    print('Max boxes: {0}'.format(max_box))


if __name__ == '__main__':
    tf.compat.v1.app.run()
