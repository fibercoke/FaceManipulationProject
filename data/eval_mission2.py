## 需要将检测得到的车牌号以xml文件的形式储存在'./Plate_dataset/AC/test/xml_pred/'，文件名一一对应
import os
import xml.etree.ElementTree as ET


if __name__ == '__main__':
    xml_gt = './Plate_dataset/AC/test/xml'
    xml_pred = './Plate_dataset/AC/test/xml_pred'
    total = 0
    pred_true = 0
    for file in os.listdir(xml_gt):
        total += 1
        anno_gt = ET.ElementTree(file=os.path.join(xml_gt, file))
        label_gt = anno_gt.find('object').find('platetext').text

        anno_pred = ET.ElementTree(file=os.path.join(xml_pred, file))
        label_pred = anno_pred.find('object').find('platetext').text

        if label_pred == label_gt:
            pred_true += 1


    print('车牌预测准确率为{}'.format(pred_true/total))

