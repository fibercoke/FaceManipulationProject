import tensorflow as tf
import cv2
import numpy as np
import scipy.stats as st
from scipy import signal
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_bool('fixed_cut', False, 'if use fixed cut')
flags.DEFINE_bool('cv_cut', True, 'if use cv cut')

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

def cut(img_raw):
    img_raw_backup = img_raw.copy()
    ok = False
    y_min = img_raw.shape[0]
    y_max = 0
    img_rawer = img_raw.copy()
    if FLAGS.cv_cut:
        delta_00 = int(img_raw.shape[0] * 0.16)
        delta_01 = int(img_raw.shape[0] * 0.15)
        delta_10 = int(img_raw.shape[1] * 0.00)
        delta_11 = int(img_raw.shape[1] * 0.02)
        img_raw = img_rawer[delta_00:img_rawer.shape[0] - delta_01, delta_10:img_rawer.shape[1] - delta_11]
        img_bw = bw_process(img_raw,a=99,b=6)
        img_bw_bak = bw_process(img_raw, a=199, b=10)
        letter_k = []
        copy_img = img_bw.copy()
        contours, hierarchy = cv2.findContours(copy_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        y_min = img_bw.shape[0]
        y_max = 0
        hs = []
        ws = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                if h >= img_bw.shape[0] * 0.4 and w >= img_bw.shape[1] * 0.05:
                    # letter_img.append(lines_img[i][y:y+h, x:x+w])
                    letter_k.append((x, y, w, h))
                    y_min = min(y_min, y)
                    y_max = max(y_max, y + h)
                    hs.append(h)
                    ws.append(w)
        hs = np.array(hs)
        ws = np.array(ws)
        h_median = np.median(hs)
        w_median = np.median(ws)
        hs = (hs - h_median) / np.max(hs)
        ws = (ws - w_median) / np.max(ws)

        if len(letter_k) < 6:
            ok = False
        else:
            if len(letter_k) > 6:
                letter_k = sorted(letter_k, key=lambda x: (x[3] - h_median) ** 2)[:6]
            ok = True
            letter_k = sorted(letter_k, key=lambda x: -x[2])
            while int(round(letter_k[0][2] / (img_bw.shape[1] / 7))) > 1:
                tmp = letter_k[1:]
                x, y, w, h = letter_k[0]
                tol = int(round(w / (img_bw.shape[1] / 7)))
                ww = w // tol
                for i in range(tol):
                    tmp.append((x + ww * i, y, ww, h))
                letter_k = tmp
            if len(letter_k) > 6:
                letter_k = sorted(letter_k, key=lambda x: (x[3] - h_median) ** 2)[:6]
            letter_k = sorted(letter_k, key=lambda x:x[0])
            chrs = [img_bw_bak[y:y + h, x:x + w] for x, y, w, h in letter_k]


    if not ok:
        img_raw = img_raw_backup
        if not FLAGS.fixed_cut:
            img_bw = bw_process(img_raw, a=199, b=48)
            if y_max > 0:
                tmp = img_raw[y_min:y_max]
            else:
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
            if len(lrs) == 6:
                chrs = [img_bw_heng[:, l:r] for l, r in lrs]
                ok = True

        if not ok:
            img_rawer = img_raw.copy()
            delta_00 = int(img_raw.shape[0] * 0.16)
            delta_01 = int(img_raw.shape[0] * 0.15)
            delta_10 = int(img_raw.shape[1] * 0.025)
            delta_11 = int(img_raw.shape[1] * 0.03)
            img_raw = img_raw[delta_00:img_rawer.shape[0] - delta_01, delta_10:img_rawer.shape[1] - delta_11]
            img_bw = bw_process(img_raw, a=199, b=14)
            #heng = np.sum(img_bw, axis=1).astype(np.int32)
            #heng = np.convolve(heng, gkern(int(img_bw.shape[0] * 0.1)), 'same')
            zong = np.sum(img_bw, axis=0).astype(np.int32)
            zong = np.convolve(zong, gkern(int(img_bw.shape[1] * 0.1)), 'same')

            lrs = []
            ddd = img_bw.shape[1] // 6

            for i in range(6):
                k = i * ddd + ddd // 2
                l = max(0, int(k - ddd * 0.45))
                r = min(img_bw.shape[1] - 1, int(k + ddd * 0.45))
                l, r = rush_b(l, r, zong, step=2, minimum=img_bw.shape[1] / 8)
                lrs.append((l, r))
            img_bw = bw_process(img_rawer, a=189)
            chrs = [img_bw[:, l:r] for l, r in lrs]
            fx = []
            for i in range(6):
                heng = np.sum(chrs[i], axis=1).astype(np.int32)
                heng = np.convolve(heng, gkern(int(chrs[i].shape[0] * 0.1)), 'same')
                k = chrs[i].shape[0] // 2
                de = int(chrs[i].shape[0] * 0.3)
                u = k - de
                d = k + de
                u, d = rush_b(u, d, heng, minimum=chrs[i].shape[0] * 0.6)
                fx.append(chrs[i][u:d])
            chrs = fx
    return chrs