import tensorflow as tf
import cv2
import numpy as np
import scipy.stats as st
from scipy import signal
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_bool('fixed_cut', False, 'if use fixed cut')
flags.DEFINE_bool('cv_cut', False, 'if use cv cut')

def bw_process(img, gk_size, bias):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    equ = cv2.equalizeHist(img_grey)
    img_bw = cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, gk_size, bias)
    #_, img_bw = cv2.threshold(img_raw,127,255,cv2.THRESH_BINARY)
    img_bw = 255 - img_bw
    return img_bw

def gkern(kernlen, sigma=None):
    """Returns a 2D Gaussian kernel."""
    if sigma is None:
        sigma = np.sqrt(kernlen/5)
    x = np.linspace(-sigma, sigma, kernlen)
    return st.norm.pdf(x)*2


def rush_forward(pos, array, step):
    while pos + step < array.shape[0] and array[pos] > array[pos + step]:
        pos += step
    return pos


def rush_backward(pos, array, step):
    while pos - step >= 0 and array[pos - step] < array[pos]:
        pos -= step
    return pos


def rush_b(ret_l, ret_r, array, step=1, minimum=0):
    l, r = ret_l, ret_r
    ret_l = rush_backward(ret_l, array, step)
    ret_l = rush_forward(ret_l, array, step)
    ret_r = rush_forward(ret_r, array, step)
    ret_r = rush_backward(ret_r, array, step)
    if ret_r - ret_l <= minimum:
        return l, r
    return ret_l, ret_r


def find_half_bogu_area(array, lmb=3.8):
    # array = np.diff(np.diff(array))
    # l, r = np.min(array), np.max(array)
    mid = l = r = int(np.max(array) / lmb)
    r += 10
    ret_l, ret_r = 0, array.shape[0] - 1
    while l < r:
        mid = (l + r + 1) // 2
        tmp = np.diff(array > mid)
        poses = np.argwhere(tmp == 1).flatten()
        if poses.shape[0] < 2:
            r = mid - 1
            continue
        d = np.diff(poses)
        pos = d.argmax()
        tmp_l, tmp_r = rush_b(poses[pos], poses[pos + 1], array)
        if d[pos] > array.shape[0] * 0.4:
            ret_l, ret_r = tmp_l, tmp_r
            l = mid
        else:
            r = mid - 1
            pass
        pass
    ret_l, ret_r = rush_b(ret_l, ret_r, array)
    return ret_l, ret_r


def get_poses_sizez(mid, zong):
    poses = np.argwhere(zong<mid).flatten()
    poses = np.hstack([0, poses, zong.shape[0] - 1])
    sizez = sorted(enumerate(np.diff(poses)), key=lambda x: -x[1])
    return poses, sizez

def get_var(mid, zong):
    poses, sizez = get_poses_sizez(mid, zong)
    if len(sizez) < 6:
        return np.inf
    sizez = sizez[:6]
    mids = []
    delta = (sizez[0][1] + sizez[1][1])/2
    for i,j in sorted(sizez, key=lambda x: x[0]):
        mids.append((poses[i+1] + poses[i]) / 2)
    if delta * 6 < zong.shape[0] * 0.7:
        return np.inf
    fff = np.diff(mids)
    mid_var = np.square(fff - np.mean(fff)).sum()
    fff = [b for a,b in sizez]
    sizez_var = np.square(fff - np.mean(fff)).sum()
    return mid_var + sizez_var

def cut(img_rawer):
    img_raw = cv2.GaussianBlur(img_rawer, (15, 15), 0)
    img_bw = bw_process(img_raw, gk_size=77, bias=9)
    sum_column = np.sum(img_bw, axis=1).astype(np.int32)
    step = sum_column.max() / 200
    thresh = sum_column.max() / 7 - step
    bogus = []
    while len(bogus) < 2:
        thresh += step
        bogus = [bogu for bogu in signal.find_peaks(-sum_column, distance=10)[0] if sum_column[bogu] < thresh]
        if len(bogus) >= 2:
            if np.diff(bogus).max() / img_raw.shape[0] < 0.5:
                bogus = []

    bogus_diff = np.diff(bogus)
    pos = np.argmax(bogus_diff)
    L, R = bogus[pos], bogus[pos + 1]
    row_cuted = img_raw[L:R]

    letter_k = []
    wide = []
    img_bw = bw_process(row_cuted, gk_size=77, bias=9)

    sum_row = np.sum(img_bw, axis=0).astype(np.int32)
    row_cuted_blured = cv2.medianBlur(img_bw, 3)
    contours, hierarchy = cv2.findContours(row_cuted_blured, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= row_cuted_blured.shape[0] * 0.75 and w >= row_cuted_blured.shape[1] * 0.05:
            if h > w and cv2.contourArea(cnt) < row_cuted_blured.shape[0] * row_cuted_blured.shape[1] * 0.2:
                wide.append((x, y, w, h))
            else:
                letter_k.append((x, y, w, h))
        else:
            row_cuted_blured[y:y + h, x:x + w] = 0
    if len(letter_k) == 0:
        width_of_each_char = int(row_cuted_blured.shape[1] / 6)
    else:
        width_of_each_char = np.median([x[2] for x in letter_k])
    for x,y,w,h in wide:
        slices = int(w/width_of_each_char+0.499)
        if slices > 0:
            ww = int(w / slices + 0.499)
            for i in range(slices):
                letter_k.append((x+i*ww, y, ww, h))

    if len(letter_k) > 6:
        letter_k = sorted(letter_k, key=lambda x: -x[2] * x[3])[:6]
    letter_k = sorted(letter_k, key=lambda x: x[0])
    chrs = [row_cuted_blured[y:y + h, x:x + w] for x, y, w, h in letter_k]
    return chrs
