#!/usr/bin/env python3
# coding: utf-8
import tensorflow as tf
from tensorflow.python.keras.losses import Loss

from utils.params import *


def _parse_param_batch(param: tf.Tensor):
    """Work for both numpy and tensor"""
    N = param.shape[0]
    p_ = tf.reshape(param[:, :12], shape=(N, 3, -1))
    p = p_[:, :, :3]
    offset = tf.reshape(p_[:, :, -1], shape=(N, 3, 1))
    alpha_shp = tf.reshape(param[:, 12:52], shape=(N, -1, 1))
    alpha_exp = tf.reshape(param[:, 52:], shape=(N, -1, 1))
    return p, offset, alpha_shp, alpha_exp


class VDCLoss(Loss):
    def __init__(self, opt_style='all'):
        super(VDCLoss, self).__init__()

        self.u = tf.convert_to_tensor(u)
        self.param_mean = tf.convert_to_tensor(param_mean)
        self.param_std = tf.convert_to_tensor(param_std)
        self.w_shp = tf.convert_to_tensor(w_shp)
        self.w_exp = tf.convert_to_tensor(w_exp)

        self.keypoints = tf.convert_to_tensor(keypoints)
        self.u_base = tf.convert_to_tensor(u[keypoints])
        self.w_shp_base = tf.convert_to_tensor(w_shp[keypoints])
        self.w_exp_base = tf.convert_to_tensor(w_exp[keypoints])

        self.w_shp_length = w_shp.shape[0] // 3

        self.opt_style = opt_style

    def reconstruct_and_parse(self, input: tf.Tensor, target: tf.Tensor):
        # reconstruct
        param = input * self.param_std + self.param_mean
        param_gt = target * self.param_std + self.param_mean

        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(param_gt)

        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def __call__(self, input: tf.Tensor, target: tf.Tensor, **kwargs):
        target = tf.reshape(target, shape=input.shape)
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)

        N = input.shape[0]
        offset = tf.concat((offset[:, :-1], offsetg[:, -1:]), axis=1)

        gt_vertex = pg @ tf.transpose(
            tf.reshape((self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg), shape=(N, -1, 3)),
            perm=(0, 2, 1)) + offsetg
        vertex = p @ tf.transpose(
            tf.reshape((self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp), shape=(N, -1, 3)),
            perm=(0, 2, 1)) + offset

        diff = tf.square(gt_vertex - vertex)
        ret = tf.reduce_mean(diff)
        return ret
