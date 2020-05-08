## 需要将检测得到的bndbox以xml文件的形式储存在'./Plate_dataset/AC/test/xml_pred/'，文件名一一对应
import os
import numpy as np
import xml.etree.ElementTree as ET


def compute_iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))  # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # 计算交并比

    return iou


if __name__ == '__main__':
    xml_gt = './Plate_dataset/AC/test/xml'
    xml_pred = './Plate_dataset/AC/test/xml_pred'
    ious = []
    for file in os.listdir(xml_gt):
        anno_gt = ET.ElementTree(file=os.path.join(xml_gt, file))
        xmin = anno_gt.find('object').find('bndbox').find('xmin').text
        ymin = anno_gt.find('object').find('bndbox').find('ymin').text
        xmax = anno_gt.find('object').find('bndbox').find('xmax').text
        ymax = anno_gt.find('object').find('bndbox').find('ymax').text
        bbox_gt = [xmin, ymin, xmax, ymax]
        bbox_gt = [int(b) for b in bbox_gt]

        anno_pred = ET.ElementTree(file=os.path.join(xml_pred, file))
        xmin = anno_pred.find('object').find('bndbox').find('xmin').text
        ymin = anno_pred.find('object').find('bndbox').find('ymin').text
        xmax = anno_pred.find('object').find('bndbox').find('xmax').text
        ymax = anno_pred.find('object').find('bndbox').find('ymax').text
        bbox_pred = [xmin, ymin, xmax, ymax]
        bbox_pred = [int(b) for b in bbox_pred]

        iou = compute_iou(bbox_gt, bbox_pred, wh=False)
        if iou < 0.5:
            print(file, iou, 'gt:', bbox_gt, 'pred:', bbox_pred)
        ious.append(iou)

    print('所有样本的平均iou:{}'.format(np.mean(ious)))
    print('检测正确的样本数目:{}'.format(len([iou for iou in ious if iou > 0.5])))

