## TO PREDICT IMAGE FILE ##
# sub_0220_Aug12.json : 0.9255973300
# infer in yolov5 dir
python3 detect.py --source ../dataset_Aug/images/test/ --weights ./runs/train/Aug12/weights/best.pt --save-txt --save-conf --exist-ok 
# make submission file
python3 submission_json.py --infer yolov5/runs/detect/exp/labels --name sub_0220_Aug12
# submit
python3 submit.py --task_no 1 --user_id kimseoha --pwd rlatjgk! --modelnm Aug12_0220 --result ./TEETH/submission/sub_0220_Aug12.json

# sub_0220_Aug12_v2.json (submission_json.py 완성전) : 0.9167225900
# infer in yolov5 dir
python3 detect.py --source ../dataset_Aug/images/test/ --weights ./runs/train/Aug12/weights/best.pt --save-txt --save-conf --conf-thres 0.5 --iou-thres 0.7 --name Aug12_0220_v2
# make submission file
python3 submission_json.py --infer yolov5/runs/detect/Aug12_0220_v2/labels --name sub_0220_Aug12_v2
# submit
python3 submit.py --task_no 1 --user_id kimseoha --pwd rlatjgk! --modelnm Aug12_0220_v2 --result ./TEETH/submission/sub_0220_Aug12_v2.json

# sub_0220_Aug12_v3.json (submission_json.py 완성전) : 0.9244494600
# infer in yolov5 dir
python3 detect.py --source ../dataset_Aug/images/test/ --weights ./runs/train/Aug12/weights/best.pt --save-txt --save-conf --iou-thres 0.2 --name Aug12_0220_v3
# make submission file
python3 submission_json.py --infer yolov5/runs/detect/Aug12_0220_v3/labels --name sub_0220_Aug12_v3
# submit
python3 submit.py --task_no 1 --user_id kimseoha --pwd rlatjgk! --modelnm Aug12_0220_v3 --result ./TEETH/submission/sub_0220_Aug12_v3.json

# sub_0220_Aug12_v4.json (submission_json.py - confidence 기준) : 0.7190379900
# infer in yolov5 dir
python3 detect.py --source ../dataset_Aug/images/test/ --weights ./runs/train/Aug12/weights/best.pt --save-txt --save-conf --name Aug12_0220_v4
# make submission file
python3 submission_json.py --infer yolov5/runs/detect/Aug12_0220_v4/labels --name sub_0220_Aug12_v4 --choose conf
# submit
python3 submit.py --task_no 1 --user_id kimseoha --pwd rlatjgk! --modelnm Aug12_0220_v4 --result ./TEETH/submission/sub_0220_Aug12_v4.json

# sub_0220_Aug12_v5.json (submission_json.py - info_strong 기준) : 
# make submission file
python3 submission_json.py --infer yolov5/runs/detect/Aug12_0220_v4/labels --name sub_0220_Aug12_v5 --choose info_strong
# submit
python3 submit.py --task_no 1 --user_id kimseoha --pwd rlatjgk! --modelnm Aug12_0220_v5 --result ./TEETH/submission/sub_0220_Aug12_v5.json

# sub_0220_Aug12_v6.json (submission_json.py - info_weak 기준) : 
# make submission file
python3 submission_json.py --infer yolov5/runs/detect/Aug12_0220_v4/labels --name sub_0220_Aug12_v6 --choose info_weak
# submit
python3 submit.py --task_no 1 --user_id kimseoha --pwd rlatjgk! --modelnm Aug12_0220_v6 --result ./TEETH/submission/sub_0220_Aug12_v6.json




# Aug_0220_2
# infer in yolov5 dir
python3 detect.py --source ../dataset_Aug/images/test/ --weights ./runs/train/Aug_0220_2/weights/best.pt --img-size 1024 --save-txt --save-conf --name Aug_0220_2 --conf-thres 0.5 --iou-thres 0.7
# make submission file
python3 submission_json.py --infer yolov5/runs/detect/Aug_0220_2/labels --name sub_0220_2
# submit
python3 submit.py --task_no 1 --user_id kimseoha --pwd rlatjgk! --modelnm Aug12_0220 --result ./TEETH/submission/sub_0220_2.json

