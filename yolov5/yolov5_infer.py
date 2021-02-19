## TO PREDICT IMAGE FILE ##
!python3 detect.py --source ../../../DATA/data_teeth/test/ --weights ./runs/train/Aug12/weights/best.pt --save-txt --save-conf --exist-ok

# !python3 yolov5/detect.py --source ../../../DATA/data_teeth/test/\
# --weights './runs/exp0_nochange/weights/best_nochange.pt'\
# --save-txt\
# --save-conf\
# --exist-ok