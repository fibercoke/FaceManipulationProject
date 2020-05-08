import tensorflow as tf
import cv2
import numpy as np
import scipy.stats as st

def bw_process(img, a=199, b=6):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    equ = cv2.equalizeHist(img_grey)
    img_bw = cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, a, b)
    img_bw = 255 - img_bw
    return img_bw

def gkern(kernlen, sigma=None):
    """Returns a 2D Gaussian kernel."""
    if sigma is None:
        sigma = np.sqrt(kernlen/5)
    x = np.linspace(-sigma, sigma, kernlen)
    return st.norm.pdf(x)*2


def rush_forward(pos, array, step):
    while pos < array.shape[0] - step and array[pos] > array[pos + step]:
        pos += step
    return pos


def rush_backward(pos, array, step):
    while pos - step >= 0 and array[pos - step] < array[pos]:
        pos -= step
    return pos


def rush_b(ret_l, ret_r, array, step=1):
    ret_l = rush_backward(ret_l, array, step)
    ret_l = rush_forward(ret_l, array, step)
    ret_r = rush_forward(ret_r, array, step)
    ret_r = rush_backward(ret_r, array, step)
    return ret_l, ret_r


def find_half_bogu_area(array):
    # array = np.diff(np.diff(array))
    # l, r = np.min(array), np.max(array)
    l = r = int(np.max(array) / 3.8)
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


def find_half_bogu_area(array, lmb=6):
    # array = np.diff(np.diff(array))
    # l, r = np.min(array), np.max(array)
    l = r = np.max(array) // lmb
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

def cut(img_raw):
    img_bw = bw_process(img_raw, a=199, b=48)
    l1, r1 = find_half_bogu_area(np.sum(img_bw, axis=1).astype(np.int32), lmb=8)
    img_bw = img_bw[l1:r1]
    l2, r2 = find_half_bogu_area(np.sum(img_bw, axis=1).astype(np.int32), lmb=6)
    tmp = img_raw[l1:r1]
    tmp = tmp[l2:r2]
    hang_ok = tmp.copy()
    img_bw_heng = bw_process(tmp,a=77,b=18)
    tmp = np.argwhere(np.sum(img_bw_heng, axis=0) == 0).flatten()

    L = max([x for x in tmp if x < (img_bw_heng.shape[1] // 15)], default=0)
    R = min([x for x in tmp if img_bw_heng.shape[1] - x < img_bw_heng.shape[1] // 15], default=img_bw_heng.shape[1] - 1)
    img_bw_heng = bw_process(hang_ok[:, L:R])

    height = img_bw_heng.shape[0]
    ker = 1 - gkern(height)
    zong=ker@img_bw_heng

    maximum = zong.max()
    ans = (np.inf, np.inf)
    for i in range(100):
        mid = maximum * i / 100
        cur = get_var(mid, zong)
        if cur < ans[0]:
            ans = (cur, mid)

    poses, sizez = get_poses_sizez(ans[1], zong)
    sizez = sizez[:6]
    lrs = []
    delta = (sizez[0][1] + sizez[1][1]) / 2
    for i, j in sorted(sizez, key=lambda x: x[0]):
        l,r=rush_b(poses[i], poses[i + 1], zong, step=5)
        lrs.append((l,r))
        if l + img_bw_heng.shape[1]//17 >= r:
            lrs = []
            break
        pass
    if len(lrs) != 6:
        chrs = []
        l = img_bw_heng.shape[1] // 6
        for i in range(6):
            chrs.append(img_bw_heng[:, i*l:(i+1)*l])
    else:
        chrs = [img_bw_heng[:, l:r] for l, r in lrs]
    return chrs