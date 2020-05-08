#!/bin/bash
python train.py --batch_size 8 --classes ./data/plate.names --dataset ./data/train.tfrecord --epochs 50 --learning_rate 0.001 --mode fit --num_classes 1 --size 360 --val_dataset ./data/test.tfrecord\ 

