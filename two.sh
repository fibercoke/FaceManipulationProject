#!/bin/bash
python step2.py --image_root_path ./data/Plate_dataset/AC/train/jpeg/ --xml_in_root_path ./data/Plate_dataset/AC/train/xml/ --save_image --save_image_root ./data/new_my_chars_train_data/
python step2.py --image_root_path ./data/Plate_dataset/AC/test/jpeg/ --xml_in_root_path ./data/Plate_dataset/AC/test/xml/ --save_image --save_image_root ./data/new_my_chars_test_data/

python generate_dataset_chars.py --dataset_path ./data/new_my_chars_train_data/ --dataset_output_path ./data/new_my_chars_fix_data_train.tfrecord --classes_output_path ./data/new_my_chars_fix_data_train.names
python generate_dataset_chars.py --dataset_path ./data/new_my_chars_test_data/ --dataset_output_path ./data/new_my_chars_fix_data_test.tfrecord --classes_output_path ./data/new_my_chars_fix_data_test.names

python chars_train.py --test_dataset_path ./data/new_my_chars_fix_data_test.tfrecord --train_dataset_path ./data/new_my_chars_fix_data_train.tfrecord
python step2.py --image_root_path ./data/Plate_dataset/AC/test/jpeg/ --xml_in_root_path ./data/Plate_dataset/AC/train/xml/ --xml_out_root_path ./data/Plate_dataset/AC/test/xml_pred/
