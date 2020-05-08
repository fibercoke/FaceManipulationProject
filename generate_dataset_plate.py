import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

from tools.dataset_utils import parse_xml, build_example


def main(_argv):
    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    writer = tf.io.TFRecordWriter(FLAGS.output_file)
    xml_list = [xml for xml in os.listdir(os.path.join(FLAGS.data_dir, 'xml')) if xml.endswith('.xml')]
    logging.info("XML list loaded: %d", len(xml_list))
    for xml in tqdm.tqdm(xml_list):
        annotation_xml = lxml.etree.fromstring(open(os.path.join(FLAGS.data_dir, 'xml', xml)).read())
        annotation = parse_xml(annotation_xml)['annotation']
        tf_example = build_example(annotation, class_map)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")


if __name__ == '__main__':
    flags.DEFINE_string('output_file', './data/train.tfrecord', 'outpot dataset')
    flags.DEFINE_string('classes', './data/plate.names', 'classes file')
    app.run(main)
