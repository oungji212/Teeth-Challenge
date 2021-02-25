import sys
sys.path.insert(0,"./efficientdet-pytorch")

import csv
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms
import random
from multiprocessing import cpu_count
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import random
import json
import argparse
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict, DetBenchTrain
from effdet.efficientdet import HeadNet

def seed_everyting(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

# label이 0부터 시작하지 않으면, CUDA error 발생 (device-side assert triggered error)
# label이 0부터 시작하지 않을 경우, 반드시 convert
def convtoeffi_label(teeth_class):
	teeth_labels = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
	return teeth_labels.index(teeth_class)

def convtori_label(effi_class):
	teeth_labels = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
	return teeth_labels[effi_class]

def mk_train_csv(data_path, label_path):
	label_json = label_path
	img_base_path = data_path
	with open(label_json) as f:
		json_data = json.load(f)

	img_names = list(json_data.keys())
	img_name_ls = [] ; H_ls = [] ; W_ls = [] ; effi_ls = []
	class_ls = [] ; xmin_ls = [] ; ymin_ls = [] ; xmax_ls = [] ; ymax_ls = []
	for img_name in tqdm(img_names,desc='{}'.format('make train csv')):
		img_path = img_base_path + '/' + img_name
		img = np.array(Image.open(img_path))
		H,W = img.shape[:2]

		img_data = json_data[img_name]
		for data in img_data:
			teeth_class = data['class']
			teeth_coord = data['coord']
			xmin, ymin, xmax, ymax = teeth_coord
			# ls 추가
			class_ls.append(teeth_class) ; effi_ls.append(convtoeffi_label(teeth_class))
			xmin_ls.append(xmin) ; ymin_ls.append(ymin)
			xmax_ls.append(xmax) ; ymax_ls.append(ymax)
			img_name_ls.append(img_name)
			H_ls.append(H) ; W_ls.append(W)

	print(len(img_name_ls), len(H_ls), len(class_ls), len(xmin_ls))
	# class: teeth class / effi_class: effi class
	df = pd.DataFrame({'img_name':img_name_ls, 'H':H_ls, 'W':W_ls, 'effi_class':effi_ls,
		'class':class_ls, 'xmin':xmin_ls, 'ymin':ymin_ls, 'xmax':xmax_ls, 'ymax':ymax_ls})
	df.to_csv('./src/train.csv', index=False)

# Albumentaion
# alb를 사용할 때는 image를 255단위로 보내줘야한다 (float형으로 보내질 경우 에러발생)
def train_transforms(p, imsize):
	return A.Compose([
		A.GaussianBlur(blur_limit=3, p=p),
		A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p),
		A.ShiftScaleRotate(shift_limit=0.035, scale_limit=0.03, rotate_limit=5, p=p),
		A.Resize(height=imsize, width=imsize, p=1),
		ToTensorV2(p=1.0),
		], bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels']))


def val_transforms(imsize):
	return A.Compose([
		A.Resize(height=imsize, width=imsize, p=1),
		ToTensorV2(p=1.0),
		], bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels']))

def test_transforms(imsize):
	return A.Compose([
		A.Resize(height=imsize, width=imsize, p=1),
		ToTensorV2(p=1.0),
		])

def original_transforms(height, width):
	return A.Compose([
		A.Resize(height=height, width=width, p=1),
		ToTensorV2(p=1.0),
		], bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels']))

# DataLoader에 넣을 데이터 불러오기
# image_names는 Load할 이미지 이름들의 numpy array
# non-default가 default보다 앞에 와야함 
class LoadDataset(Dataset):
	def __init__(self, image_names, train_csv=None, which='train', root='./src/images/', p=0.3, imsize=1280):
		super().__init__()
		self.train_csv = train_csv
		self.image_names = image_names
		self.which = which
		self.root = root
		self.p = p
		self.imsize = imsize

	def __getitem__(self, index: int):
		image_name = self.image_names[index]

		target = {}
		if self.which != 'test':
			image, boxes, labels = self.load_image_and_boxes(index)
			if self.which == 'train':
				transform = train_transforms(self.p, self.imsize)
			else: # self.which == 'val'
				transform = val_transforms(self.imsize)
			transformed = transform(image=image, bboxes=boxes, labels=labels)
			assert len(transformed['bboxes']) == labels.shape[0], 'not equal!'

			transformed_img = transformed['image']
			target['bbox'] = torch.tensor(transformed['bboxes'])
			target['bbox'][:,[0,1,2,3]] = target['bbox'][:,[1,0,3,2]] # tf_effi의 yxyx의 format 따르기
			target['cls'] = torch.tensor(transformed['labels'])
			target['img_scale'] = torch.tensor([1.])
			target['img_size'] = torch.tensor([(self.imsize,self.imsize)])
			return transformed_img, target

		else:
			image = self.load_image_and_boxes(index)
			transform = test_transforms(self.imsize)
			transformed = transform(image=image)
			transformed_img = transformed['image']
			target['img_scale'] = torch.tensor([1.])
			target['img_size'] = torch.tensor([(self.imsize,self.imsize)])
			return image_name, transformed_img, target

	def __len__(self) -> int:
		return self.image_names.shape[0] # 데이터 개수

	
	def load_image_and_boxes(self,index):
		image_name = self.image_names[index]
		if self.which != 'test':
			img_path = self.root + 'train/'+ image_name
			image = np.array(Image.open(img_path)).astype(np.uint8)
			records = self.train_csv[self.train_csv['img_name'] == image_name].reset_index(drop=True)
			boxes = records[['xmin','ymin','xmax','ymax']].values 
			labels = np.array(records['effi_class'])
			return image, boxes, labels
		else:
			img_path = self.root + 'test/' + image_name
			image = np.array(Image.open(img_path)).astype(np.uint8)
			return image


