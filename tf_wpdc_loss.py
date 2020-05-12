#!/usr/bin/env python3
# coding: utf-8
from math import sqrt

import tensorflow as tf
from tensorflow.python.keras.losses import Loss

from utils.params import *


def _parse_param_batch(param):
    N = tf.shape(param)[0]
    p_ = tf.reshape(param[:, :12], shape=(N, 3, 4))
    p = p_[:, :, :3]
    offset = tf.reshape(p_[:, :, -1], shape=(N, 3, 1))
    alpha_shp = tf.reshape(param[:, 12:52], shape=(N, 40, 1))
    alpha_exp = tf.reshape(param[:, 52:], shape=(N, 10, 1))
    return p, offset, alpha_shp, alpha_exp


class WPDCLoss(Loss):
    def __init__(self, opt_style='resample', resample_num=132):
        super(WPDCLoss, self).__init__()
        self.opt_style = opt_style
        self.param_mean = tf.convert_to_tensor(param_mean)
        self.param_std = tf.convert_to_tensor(param_std)

        self.u = tf.convert_to_tensor(u)
        self.w_shp = tf.convert_to_tensor(w_shp)
        self.w_exp = tf.convert_to_tensor(w_exp)
        self.w_norm = tf.convert_to_tensor(w_norm)

        self.w_shp_length = self.w_shp.shape[0] // 3
        self.keypoints = tf.convert_to_tensor(keypoints)
        self.resample_num = resample_num

    def reconstruct_and_parse(self, input: tf.Tensor, target: tf.Tensor):
        # reconstruct
        param = input * self.param_std + self.param_mean
        param_gt = target * self.param_std + self.param_mean

        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(param_gt)

        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def _calc_weights_resample(self, input_: tf.Tensor, target_: tf.Tensor):
        # resample index
        # if self.resample_num <= 0:
        #    keypoints_mix = self.keypoints
        # else:
        #    tf.random.
        #    index = torch.randperm(self.w_shp_length)[:self.resample_num].reshape(-1, 1)
        #    keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1).cuda()
        #    keypoints_mix = torch.cat((self.keypoints, keypoints_resample))

        keypoints_mix = self.keypoints
        w_shp_base = tf.gather(self.w_shp, keypoints_mix)
        u_base = tf.gather(self.u, keypoints_mix)
        w_exp_base = tf.gather(self.w_exp, keypoints_mix)

        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input_, target_)

        input = self.param_std * input_ + self.param_mean
        target = self.param_std * target_ + self.param_mean

        N = tf.shape(input)[0]

        offset = tf.concat((offset[:, :-1], offsetg[:, -1:]), axis=1)

        tmpv = u_base + tf.matmul(w_shp_base, alpha_shp) + tf.matmul(w_exp_base, alpha_exp)
        tmpv = tf.reshape(tmpv, shape=(tf.shape(tmpv)[0], -1, 3))
        tmpv = tf.transpose(tmpv, perm=(0, 2, 1))

        tmpv_norm = tf.norm(tmpv, axis=2)
        offset_norm = sqrt(w_shp_base.shape[0] // 3)

        # for pose
        param_diff_pose = tf.abs(input[:, :11] - target[:, :11])
        weight_lst = []
        for ind in range(11):
            if ind in [0, 4, 8]:
                weight_lst.append(param_diff_pose[:, ind] * tmpv_norm[:, 0])
            elif ind in [1, 5, 9]:
                weight_lst.append(param_diff_pose[:, ind] * tmpv_norm[:, 1])
            elif ind in [2, 6, 10]:
                weight_lst.append(param_diff_pose[:, ind] * tmpv_norm[:, 2])
            else:
                weight_lst.append(param_diff_pose[:, ind] * offset_norm)

        ## This is the optimizest version
        # for shape_exp
        magic_number = 0.00057339936  # scale
        param_diff_shape_exp = tf.abs(input[:, 12:] - target[:, 12:])
        # weights[:, 12:] = magic_number * param_diff_shape_exp * self.w_norm
        w = tf.concat((w_shp_base, w_exp_base), axis=1)
        w_norm = tf.norm(w, axis=0)
        # print('here')
        # weights[:, 12:] = magic_number * param_diff_shape_exp * w_norm
        weights_mid = tf.zeros(shape=tf.shape(input_)[0])
        weights_mid = tf.expand_dims(weights_mid, axis=1)

        eps = 1e-6

        weights_A = tf.stack(weight_lst, axis=1)
        weights_B = magic_number * param_diff_shape_exp * w_norm

        weights = tf.concat((weights_A + eps, weights_mid, weights_B + eps), axis=1)

        # normalize the weights
        maxes = tf.reduce_max(weights, axis=1)
        # maxes, _ = weights.max(dim=1)
        # maxes = maxes.view(-1, 1)
        maxes = tf.reshape(maxes, shape=(-1, 1))
        weights /= maxes

        return weights

    def call(self, y_true, y_pred):
        weights = self._calc_weights_resample(y_pred, y_true)
        loss = weights * tf.square(y_pred - y_true)
        return tf.reduce_mean(loss)


if __name__ == '__main__':
    pass
