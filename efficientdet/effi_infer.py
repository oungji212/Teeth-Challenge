import sys
sys.path.insert(0,"../efficientdet-pytorch")

import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms
import random
from multiprocessing import cpu_count
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import random
import json
import argparse
from effiDataload import *
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict, DetBenchTrain, create_model_from_config
from effdet.efficientdet import HeadNet
from torch.cuda.amp import autocast, GradScaler
import cv2
# from ensemble_boxes import *
import warnings
warnings.filterwarnings("ignore")

# WBF function from weighted-boxes-fusion
# =========================================================================================================

def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn('X1 < 0 in box. Set it to 0.')
                x1 = 0
            if x1 > 1:
                warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x1 = 1
            if x2 < 0:
                warnings.warn('X2 < 0 in box. Set it to 0.')
                x2 = 0
            if x2 > 1:
                warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x2 = 1
            if y1 < 0:
                warnings.warn('Y1 < 0 in box. Set it to 0.')
                y1 = 0
            if y1 > 1:
                warnings.warn('Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y1 = 1
            if y2 < 0:
                warnings.warn('Y2 < 0 in box. Set it to 0.')
                y2 = 0
            if y2 > 1:
                warnings.warn('Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            b = [int(label), float(score) * weights[t], x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[2:] += (b[1] * b[2:])
        conf += b[1]
        conf_list.append(b[1])
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    :param allows_overflow: false if we want confidence score not exceed 1.0
    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels

# =========================================================================================================


def seed_everyting(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

def make_nochange_Test(): 
	# tset data
	test_imgs_path = '../../../../DATA/data_teeth/test'
	os.makedirs('../dataset_nochange/images/test')
	img_names = os.listdir(test_imgs_path)
	for img_name in tqdm(img_names,desc='{}'.format('test_preprocess')):
		# img
		img_path = test_imgs_path+'/'+img_name
		img_gray = np.array(Image.open(img_path).convert('L')) 
		img_h, img_w = img_gray.shape[0], img_gray.shape[1]
		# eda.py를 바탕으로 크롭할 부분 지정하기 (나중에)
		clahe1 = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
		clahe2 = cv2.createCLAHE(clipLimit=2, tileGridSize=(30,30))
		img_cl1 = clahe1.apply(img_gray)
		img_cl2 = clahe2.apply(img_gray)
		img_cl_twice = clahe1.apply(img_cl2)
		final = np.dstack((img_cl_twice, img_cl2, img_cl1))
		save_img = Image.fromarray(final.astype(np.uint8))
		save_img.save('../dataset_nochange/images/test/{}'.format(img_name))


def collate_fn(batch):
	return tuple(zip(*batch))

# model load
def Load_Model(checkpoint_path, which_model='effi5', img_size=1280):
	if which_model=='effi5':
		config = get_efficientdet_config('tf_efficientdet_d5') # tf_effi5 model structure 

	elif which_model=='effi4':
		config = get_efficientdet_config('tf_efficientdet_d4') # tf_effi4 model structure 

	elif which_model=='effi6':
		config = get_efficientdet_config('tf_efficientdet_d6') # tf_effi4 model structure 

	config.image_size = (img_size, img_size)
	config.num_classes = 32
	config.norm_kwargs = dict(eps=.001, momentum=.01) 
	net = EfficientDet(config, pretrained_backbone=False)
	net.class_net = HeadNet(config, num_outputs=config.num_classes)

	ckp = torch.load(checkpoint_path)
	net.load_state_dict(ckp['model_state_dict'])
	del ckp

	net = DetBenchPredict(net)
	net.eval()
	return net


# 5개의 fold의 model result를 모두 모으기
def fold5_predictions(images, targets, model_ls, score_threshold, device, imsize):
	with torch.no_grad():
		images = torch.stack(images).float().to(device)
		predictions = [] # 총 5개의 모델별 (이미지들의)prediction이 predictions 안에 존재
		for model in model_ls: # fold별 best.pt
			result = []

			target_res = {}
			target_res['img_size'] = torch.tensor([(imsize, imsize) for target in targets]).to(device).float()
			target_res['img_scale'] = torch.tensor([target['img_scale'].to(device).float() for target in targets])

			detections = model(images, target_res)
			# predict한 box를 refine
			# DetBenchPredict와 DetBenchTrain은 xyxy로 output
			for i in range(images.shape[0]): # 결국 이미지 하나당 5개의 model로 predict
				boxes = detections[i].detach().cpu().numpy()[:,:4] # xmin ymin xmax ymax
				scores = detections[i].detach().cpu().numpy()[:,4]
				labels = detections[i].detach().cpu().numpy()[:,5]
				indexes = np.where(scores > score_threshold)[0]
				result.append({
					'boxes':boxes[indexes],
					'scores': scores[indexes],
					'labels': labels[indexes],
					})
			predictions.append(result)
	return predictions


# 5개의 fold의 result들에 대해 WBF 실행
def run_wbf(predictions, image_index, image_size, iou_thr, skip_box_thr, weight=None):
	boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist() for prediction in predictions]
	scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
	labels = [prediction[image_index]['labels'].tolist() for prediction in predictions]
	boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
	boxes = boxes*(image_size-1) # 앞에서 prediction[image_index]['boxes']/(image_size-1)을 했기 때문에 다시 원상복구
	return boxes, scores, labels
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Inference setting')
	parser.add_argument('--obj_thr', default=0.25, type=float, help='objectness score threshold')
	parser.add_argument('--iou_thr', default=0.55, type=float, help='WBF iou threshold')
	parser.add_argument('--skip_box_thr', default=0.21, type=float, help='WBF skip box threshold')
	parser.add_argument('--model', default='effi4', type=str, help='efficient model version')
	parser.add_argument('--imsize', default=640, type=int, help='image size')
	parser.add_argument('--batch', default=8, type=int, help='batch size')
	parser.add_argument('--nw', default=4, type=int, help='num_workers')
	parser.add_argument('--ckp_nm', default='effi4_0223', type=str, help='saved checkpoint name')
	parser.add_argument('--name', default=None, type=str, help='submission json name')
	args = parser.parse_args()

	# make data
	if not os.path.isdir('../dataset_nochange/images/test'):
		make_nochange_Test()

	# clean 
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	with torch.cuda.device(device):
		torch.cuda.empty_cache()

	# for reproduce
	SEED = 42
	seed_everyting(42)

	# dataset
	test_img_path = '../dataset_nochange/images/test'
	test_image_names = np.array(os.listdir(test_img_path))
	test_dataset = LoadDataset(image_names = test_image_names, imsize = args.imsize, which = 'test')
	test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=args.nw, collate_fn=collate_fn, shuffle=False)

	# load weight & model
	checkpoint_ls = ['models/{}_{}_best.pt'.format(args.ckp_nm, i) for i in range(5)]
	models = [Load_Model(checkpoint_path=weight, which_model=args.model, img_size=args.imsize).to(device) for weight in checkpoint_ls]	
	
	# Predict
	results = {}
	for (image_names, images, targets) in tqdm(test_loader, total=len(test_loader)):
		predictions = fold5_predictions(images=images, targets=targets, model_ls=models, 
			score_threshold=args.obj_thr, device=device, imsize=args.imsize)
		for i, (image_name, image) in enumerate(zip(image_names, images)):
			H, W = np.array(Image.open('{}/{}'.format(test_img_path, image_name))).shape[:2] ; print(H,W) # torch의 경우 CHW, numpy의 경우 HWC
			boxes, scores, labels = run_wbf(predictions=predictions, image_index=i, image_size=args.imsize,
				iou_thr=args.iou_thr, skip_box_thr=args.skip_box_thr, weight=None)
			original_transformed = original_transforms(height=H, width=W)
			original = original_transformed(image=image.permute(1,2,0).numpy(), bboxes=boxes, labels=labels)
			original_bboxes = original['bboxes']
			# boxes[:,0] = boxes[:,0]*(W/args.image_size).round().astype(np.int32).clip(min=0, max=W) # xmin
			# boxes[:,2] = boxes[:,2]*(W/args.image_size).round().astype(np.int32).clip(min=0, max=W) # xmax
			# boxes[:,1] = boxes[:,1]*(H/args.image_size).round().astype(np.int32).clip(min=0, max=H) # ymin
			# boxes[:,3] = boxes[:,3]*(H/args.image_size).round().astype(np.int32).clip(min=0, max=H) # ymax
			results[image_name] = []
			for values in zip(labels, scores, original_bboxes):
				one_dict = {}
				effi_class = int(values[0])  
				teeth_class = convtori_label(effi_class) ; one_dict['class'] = teeth_class
				confidence = values[1] ; one_dict['confidence_score'] = float(confidence)
				coords = [int(round(values[2][0])), int(round(values[2][1])), int(round(values[2][2])), int(round(values[2][3]))]
				one_dict['coord'] = coords
				results[image_name].append(one_dict)

	# json
	os.makedirs('./submission', exist_ok=True)
	if args.name == None:
		save_path = './submission/{}.json'.format(args.ckp_nm)
	else:
		save_path = './submission/{}.json'.format(args.name)
	with open(save_path, 'w') as json_file:
		json.dump(results, json_file, ensure_ascii=False, indent=2)

