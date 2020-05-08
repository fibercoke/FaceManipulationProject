from absl import app, flags, logging
from absl.flags import FLAGS
from chars_model import CModel
import yolov3_tf2.dataset_chars as dataset
import tensorflow as tf

from chars_model import CModel


def detect(img, model):
    if img.ndim == 3:
        img = tf.expand_dims(img, axis=0)
    result = model(img)
    #print(result)
    cls = tf.math.argmax(result, axis=1)
    return cls

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    class_map_inv = {idx:name for idx, name in enumerate(open(FLAGS.classes_path).read().splitlines())}
    class_num = len(class_map_inv)
    model = CModel(FLAGS.classifier_size, class_num, None)
    model.load_weights(FLAGS.ckpt_name).expect_partial()

    img = tf.image.decode_image(open(FLAGS.char_image_path, 'rb').read(), channels=3)
    img = dataset.transform_images(img, FLAGS.classifier_size)

    lst = [class_map_inv[i] for i in detect(img, model)]

    print(lst)

if __name__ == '__main__':
    flags.DEFINE_string('ckpt_name', './chars_checkpoints/resnet101v2_train_98.tf', 'checkpoint name')
    flags.DEFINE_integer('classifier_size', 32, 'size of each character should be resize to')
    flags.DEFINE_string('classes_path', './data/chars_data.names', 'path to output class file')
    flags.DEFINE_string('char_image_path', './data/Chars_data/A/11-5.jpg', 'path to input image')
    try:
        app.run(main)
    except SystemExit:
        pass
