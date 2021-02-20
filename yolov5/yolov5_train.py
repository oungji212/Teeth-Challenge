# yolov5 folder에서
python3 train.py --data ./data/Aug.yaml --cfg ./models/yolov5x.yaml --weights yolov5x.pt --rect --batch 16 --name Aug --notest

# 연속으로 훈련하고 싶은 경우는 이전 weight인 ./runs/train/Aug12/weights/best.pt를 불러와서 계속 학습진행시켜도 될 듯

# 0220_2: img-size:1024 / batch-4
python3 train.py --data ./data/Aug.yaml --cfg ./models/yolov5x.yaml --weights yolov5x.pt --rect --img-size 1024 --batch 4 --name Aug_0220_2 --notest
