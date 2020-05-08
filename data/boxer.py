import cv2
import numpy as np
img = cv2.imread('girl.png')
h,w,_ = img.shape
print("h:", h)
print("w:", w)
labels = [
    [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
    [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
    [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
]

for xmin,ymin,xmax,ymax,c in labels:
    x1y1=tuple(np.array([xmin*w, ymin*h]).astype(np.int32))
    x2y2=tuple(np.array([xmax*w, ymax*h]).astype(np.int32))
    print(x1y1, x2y2)
    cv2.rectangle(img, x1y1, x2y2, (0,0,255),2)

cv2.imwrite('girl_box.png', img)