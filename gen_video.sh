python face_detect_demo.py --input report/VID_20200526_131730.mp4 --save_each --alpha 0.4 --beta 1 --filter none --face_only

python face_detect_demo.py --input report/VID_20200526_131730.mp4 --save_each --alpha 0 --beta 1 --filter none
python face_detect_demo.py --input report/VID_20200526_131730.mp4 --save_each --alpha 0 --beta 1 --filter linear
python face_detect_demo.py --input report/VID_20200526_131730.mp4 --save_each --alpha 0 --beta 1 --filter kalman

python face_detect_demo.py --input report/VID_20200526_131730.mp4 --save_each --alpha 0.4 --beta 1 --filter none
python face_detect_demo.py --input report/VID_20200526_131730.mp4 --save_each --alpha 0.4 --beta 1 --filter linear
python face_detect_demo.py --input report/VID_20200526_131730.mp4 --save_each --alpha 0.4 --beta 1 --filter kalman

python face_detect_demo.py --input report/VID_20200526_131730.mp4 --save_each --alpha 0 --beta 0.6 --filter none
python face_detect_demo.py --input report/VID_20200526_131730.mp4 --save_each --alpha 0 --beta 0.6 --filter linear
python face_detect_demo.py --input report/VID_20200526_131730.mp4 --save_each --alpha 0 --beta 0.6 --filter kalman
