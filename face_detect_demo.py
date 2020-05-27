import time
import subprocess
from subprocess import Popen

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from filters import MeanFilter, KalmanFilters
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
import matplotlib.pyplot as plt
import face_alignment
from scipy.spatial import Delaunay
from faceMorph import morphTriangle
import os


flags.DEFINE_string('classes', './data/wddb_classes.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_tiny_wddb_step220200524-125021_012.ckpt', 'path to weights file')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('t_path', './trump.jpg', 'dest path')
flags.DEFINE_string('input', 'stream', 'input file or `stream`')
flags.DEFINE_string('out_file', '', 'video out name')
flags.DEFINE_bool('save_each', False, 'Save each picture?')
flags.DEFINE_bool('face_only', False, 'only face?')

flags.DEFINE_string('alpha', '0', 'alpha')
flags.DEFINE_string('beta', '1', 'beta')
flags.DEFINE_enum('filter', 'none', ['kalman', 'linear', 'none'], 'which filter should be used')


def transform_images(x_train, size):
    x_train = tf.image.resize_with_pad(x_train, size, size)
    x_train = (x_train - 127.5) / 128
    return x_train


def get_landmarks(img_rgb, model, detector):
    img = transform_images(img_rgb, FLAGS.size)
    boxes, scores, classes, nums = model.predict_on_batch(tf.expand_dims(img, 0))
    boxes_new = boxes[0]
    wh = np.flip(img.shape[0:2])
    boxes_new[:, 0:2] *= wh
    boxes_new[:, 2:4] *= wh
    bbox = [i for i in np.hstack([boxes_new, scores.transpose()]) if i[4] > 0.4]
    return detector.get_landmarks_from_image(img_rgb, bbox)


def plot_tri_map(tri, points):
    plt.figure(figsize=(16, 16))
    plt.triplot(-points[:, 0], -points[:, 1], tri.simplices)
    plt.plot(-points[:, 0], -points[:, 1], 'o')
    plt.show()


def get_border_points(shape, split_num=10):
    first = np.arange(split_num + 1) * ((shape[0] - 1) / split_num)
    second = np.arange(split_num + 1) * ((shape[1] - 1) / split_num)
    return np.hstack([np.stack([first, np.zeros(first.shape[0])]),
                      np.stack([np.zeros(second.shape[0]), second])[:, 1:],  # remove duplicate points
                      np.stack([first, np.ones(first.shape[0]) * (shape[1] - 1)])[:, 1:],  # remove duplicate points
                      np.stack([np.ones(second.shape[0]) * (shape[0] - 1), second])[:, 1:-1]]).transpose()


def apply_s_to_t(img1_raw, img2_raw, model, detector, filters=None, alpha=0.0, beta=1.0, points2=None,
                 face_only=False):
    img1 = tf.image.resize_with_pad(img1_raw, FLAGS.size, FLAGS.size).numpy()
    img2 = tf.image.resize_with_pad(img2_raw, FLAGS.size, FLAGS.size).numpy()

    points1 = np.vstack([get_landmarks(img1, model, detector)[0], get_border_points(img1.shape)])
    if filters is not None:
        filters.correct(points1)
        points1 = filters.predict()
    if face_only:
        points1 = points1[:68]
    tri1 = Delaunay(points1)

    if points2 is None:
        points2 = np.vstack([get_landmarks(img2, model, detector)[0], get_border_points(img2.shape)])

    # tri2 = Delaunay(points2)
    # plot_tri_map(tri2, points2)
    # plot_tri_map(tri1, points1)

    # Read array of corresponding points
    points = []

    # Compute weighted average point coordinates
    for (x1, y1), (x2, y2) in zip(points1, points2):
        x = (1 - alpha) * x1 + alpha * x2
        y = (1 - alpha) * y1 + alpha * y2
        points.append((x, y))

    # Allocate space for final output
    result_img = np.zeros(img2.shape, dtype=img2.dtype)
    triangle_list = tri1.simplices
    # Read triangles from delaunay_output.txt
    for x, y, z in triangle_list:
        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morphTriangle(img1, img2, result_img, t1, t2, t, beta=beta)
    return result_img.astype(np.uint8)


def main(_argv):
    alpha = eval(FLAGS.alpha)
    beta = eval(FLAGS.beta)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    cam = cv2.VideoCapture(0 if FLAGS.input == 'stream' else FLAGS.input)
    detector = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType._2D)
    img_t = tf.image.decode_image(open(FLAGS.t_path, 'rb').read(), channels=3)

    img2 = tf.image.resize_with_pad(img_t, FLAGS.size, FLAGS.size).numpy()
    points2 = np.vstack([get_landmarks(img2, yolo, detector)[0], get_border_points(img2.shape)])

    # Kalman filter
    if FLAGS.filter == 'kalman':
        flt = KalmanFilters(filter_num=68)
    elif FLAGS.filter == 'linear':
        flt = MeanFilter(history_size=4)
    else:
        flt = None

    current_frame = 0
    time_str = time.strftime("%Y%m%d-%H%M%S")

    if FLAGS.save_each:
        out_file = FLAGS.out_file
        tmp = os.path.join('outputs', os.path.split(FLAGS.t_path)[-1])
        if not os.path.exists(tmp):
            os.mkdir(tmp)
        tmp = os.path.join('outputs', os.path.split(FLAGS.t_path)[-1], os.path.split(FLAGS.input)[-1])
        if not os.path.exists(tmp):
            os.mkdir(tmp)
        if out_file == '':
            lst = ['output',
                   'alpha', FLAGS.alpha,
                   'beta', FLAGS.beta,
                   'filter', FLAGS.filter]
            if FLAGS.face_only:
                lst.append('face_only')
            lst.append(time_str)
            out_file = '_'.join(lst) + '.mp4'
            out_file = os.path.join('outputs', os.path.split(FLAGS.t_path)[-1], os.path.split(FLAGS.input)[-1], out_file)
        command = ['ffmpeg',
                   '-y',
                   '-f', 'image2pipe',
                   '-r', '30',
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   out_file]
        proc = Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        total_frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        # out_dir = 'outputs/{0}'.format(time_str)
        # print('out dir: {0}'.format(out_dir))
        # os.mkdir(out_dir)
        pbar = tqdm(total=total_frame)
    while cam.isOpened():
        current_frame += 1
        success, img_raw = cam.read()
        if not success:
            break
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        # try:
        img_to_show = apply_s_to_t(img_raw, img_t, yolo, detector, filters=flt, alpha=alpha, beta=beta, points2=points2, face_only=FLAGS.face_only)
        # except Exception as e:
        #    print(e)
        #    img_to_show = img_raw
        if FLAGS.save_each:
            pbar.update(1)
            data = tf.image.encode_png(tf.convert_to_tensor(img_to_show)).numpy()
            proc.stdin.write(data)
            # with open('outputs/{time_str}/{total_frame:04d}.png'.format(
            #         total_frame=current_frame, time_str=time_str), 'wb') as f:
            #     f.write(data)
        else:
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)
            cv2.imshow('webcam', img_to_show)

            if cv2.waitKey(1) == 27:
                break  # esc to quit
    cam.release()
    if FLAGS.save_each:
        pbar.close()
        proc.stdin.close()
        proc.wait()
        # print("run: `ffmpeg -i {0}/%04d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p {1}`".format(out_dir,
        #                                                                                          FLAGS.out_file))

    cv2.destroyAllWindows()
    # cv2.imwrite(FLAGS.output, img)
    # logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
