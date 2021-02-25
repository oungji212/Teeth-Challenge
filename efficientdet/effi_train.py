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
from ranger import Ranger
import warnings
warnings.filterwarnings("ignore")


def seed_everyting(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

# Model
def Train_Model(which_model='effi5', img_size=1280): # 모델 및 pretrained weight 불러오기
	if which_model=='effi5':
		config = get_efficientdet_config('tf_efficientdet_d5') # tf_effi5 model structure 
		config.image_size = (img_size, img_size)
		config.norm_kwargs = dict(eps=.001, momentum=.01) 
		# net = create_model_from_config(config, bench_task='train',
		# 	num_classes=32, checkpoint_path='../efficientdet-pytorch/tf_efficientdet_d5-ef44aea8.pth')
		net = EfficientDet(config, pretrained_backbone=False)
		pretrained_weight = torch.load('../efficientdet-pytorch/tf_efficientdet_d5-ef44aea8.pth')
		net.load_state_dict(pretrained_weight)
		# 기본 net에다가 pretrained weight을 불러온 후 class를 바꿔야함 (reset)

	elif which_model=='effi4':
		config = get_efficientdet_config('tf_efficientdet_d4') # tf_effi4 model structure 
		config.image_size = (img_size, img_size)
		config.norm_kwargs = dict(eps=.001, momentum=.01) 
		net = EfficientDet(config, pretrained_backbone=False)
		pretrained_weight = torch.load('../efficientdet-pytorch/tf_efficientdet_d4-5b370b7a.pth')
		net.load_state_dict(pretrained_weight)

	elif which_model=='effi6':
		config = get_efficientdet_config('tf_efficientdet_d6') # tf_effi4 model structure 
		config.image_size = (img_size, img_size)
		config.norm_kwargs = dict(eps=.001, momentum=.01) 
		net = EfficientDet(config, pretrained_backbone=False)
		pretrained_weight = torch.load('../efficientdet-pytorch/tf_efficientdet_d6-51cb0132.pth')
		net.load_state_dict(pretrained_weight)

	net.reset_head(num_classes=32)
	net.class_net = HeadNet(config, num_outputs=config.num_classes)
	return DetBenchTrain(net, config)

# loss 
class Showloss(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1): # val: loss.item(),  n: batch_size
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum / self.count

# loss를 model에 들어있는 애를 쓰기 때문에 이렇게 setting_config / Training 나눠서 하는게 편함
class setting_config:
	n_epochs = 200
	lr = 0.00025
	# scheduler
	val_scheduler = True
	SchedulerClass = ReduceLROnPlateau
	scheduler_params = dict(mode='min',factor=0.5,patience=2)
	# Early stopping
	patience = 3


# Early Stopping (validation loss 기준)
class EarlyStopping:
	def __init__(self, patience): # loss 기준
		self.patience = patience
		self.counter = 0
		self.early_stop = False

	def __call__(self,now_loss,best_loss):
		if now_loss > best_loss:
			self.counter += 1
			print('Early Stopping: {} / {}'.format(self.counter, self.patience))
		else: # 제일 낮은 loss 달성
			self.counter = 0

		if self.counter >= self.patience:
			print('Early Stopping - END')
			self.early_stop = True


# object detect이기 때문에 다르게 baseline을 짜야함
class Training:
	def __init__(self, model, device, config, name, fold_num, imsize):
		self.config = config
		self.epoch = 0
		self.base_dir = './models/'
		os.makedirs('./models', exist_ok=True)
		self.model = model
		self.best_loss = 10**5
		self.device = device
		self.name = name
		self.fold_num = fold_num
		self.imsize = imsize
		# optimize
		param_optimizer = list(self.model.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.001},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.00}
		]
		self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
		self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
		# Earlystopping
		self.patience = config.patience
		# GradScaler
		self.scaler = GradScaler()

	def train_one_epoch(self, train_loader):
		self.model.train()
		showloss = Showloss()

		for step, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
			self.optimizer.zero_grad()

			with autocast():
				images = torch.stack(images) # 이미지들을 합쳐 Batch 생성 (default: dim=0) [B,C,H,W]
				images = images.to(self.device).float()
				batch_size = images.shape[0]
				boxes = [target['bbox'].to(self.device).float() for target in targets]
				labels = [target['cls'].to(self.device).float() for target in targets]
				img_scale = torch.tensor([target['img_scale'].to(self.device).float() for target in targets])
				img_size = torch.tensor([(self.imsize, self.imsize) for target in targets]).to(self.device).float()

				# update 후로 forward는 image와 target_dict를 인자로 받음
				target_res = {}
				target_res['bbox'] = boxes
				target_res['cls'] = labels
				target_res['img_scale'] = img_scale
				target_res['img_size'] = img_size

				# pred
				output = self.model(images, target_res)
				loss = output['loss']
				showloss.update(loss.detach().item(), batch_size)

			self.scaler.scale(loss).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()

		return showloss

	def val_one_epoch(self, val_loader):
		self.model.eval()
		showloss = Showloss()
		for step, (images, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
			with torch.no_grad():
				images = torch.stack(images)
				batch_size = images.shape[0]
				images = images.to(self.device).float()
				boxes = [target['bbox'].to(self.device).float() for target in targets]
				labels = [target['cls'].to(self.device).float() for target in targets]
				img_scale = torch.tensor([target['img_scale'].to(self.device).float() for target in targets])
				img_size = torch.tensor([(self.imsize, self.imsize) for target in targets]).to(self.device).float()

				target_res = {}
				target_res['bbox'] = boxes
				target_res['cls'] = labels
				target_res['img_scale'] = img_scale
				target_res['img_size'] = img_size

				# loss, _, _ = self.model(images, boxes, labels)
				output = self.model(images, target_res)
				loss = output['loss']
				showloss.update(loss.detach().item(), batch_size)

		return showloss

	def save(self, path): # 모델 및 파라미터 저장
		self.model.eval()
		torch.save({
			'model_state_dict': self.model.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
			'loss': self.best_loss, # val
			'epoch': self.epoch,
			}, path)

	def load(self, path):
		checkpoint = torch.load(path)
		self.model.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		self.best_loss = checkpoint['best_loss'] # val
		self.epoch = checkpoint['epoch'] + 1


	def fit(self, train_loader, val_loader):
		early_stopping = EarlyStopping(self.patience)
		for epoch in range(self.config.n_epochs):
			print('{} / {} Epoch'.format(epoch, self.config.n_epochs))
			train_loss = self.train_one_epoch(train_loader)
			print('[Train] loss: {}'.format(train_loss.avg))
			self.save(self.base_dir+'{}_{}_last.pt'.format(self.name, self.fold_num))

			val_loss = self.val_one_epoch(val_loader)
			print('[Valid] loss: {}'.format(val_loss.avg))

			if val_loss.avg < self.best_loss:
				self.best_loss = val_loss.avg
				self.save(self.base_dir+'{}_{}_best.pt'.format(self.name, self.fold_num))

			# Early stopping
			early_stopping(val_loss.avg, self.best_loss)
			if early_stopping.early_stop:
				break

			if self.config.val_scheduler:
				self.scheduler.step(metrics=val_loss.avg)

			self.epoch += 1
			
# dataset이 variable length(데이터 길이가 다양하)면 바로 못 묶이므로 collate_fn을 만들어서 넘겨줘야함
# 배치로 묶일 모든 데이터를 잘 묶어주는 collate_fn 함수 필요!
def collate_fn(batch):
	return tuple(zip(*batch))

def bbox_label(x):
	if x <= 12:
		return 0
	elif x <= 17:
		return 1
	elif x <= 23:
		return 2
	elif x <= 27:
		return 3
	elif x == 28:
		return 4
	else:
		return 5

def love_teeth(x, label_df):
	df = label_df[label_df['img_name'] == x].reset_index(drop=True)
	class_ls = np.array(df['class'])
	love_teeth_ls = np.array([18,28,38,48])
	return sum(np.isin(class_ls, love_teeth_ls)) # 사랑니의 총 개수


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='model setting')
	parser.add_argument('--model', default='effi4', type=str, help='efficient model version')
	parser.add_argument('--imsize', default=640, type=int, help='image size')
	parser.add_argument('--batch', default=4, type=int, help='batch size')
	parser.add_argument('--nw', default=4, type=int, help='num_workers')
	parser.add_argument('--name', type=str, help='model save name')
	args = parser.parse_args()

	# clean 
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	with torch.cuda.device(device):
		torch.cuda.empty_cache()

	# effi에서 실행
	if not os.path.isfile('train.csv'):
		print('Make train.csv')
		mk_train_csv()
	label_df = pd.read_csv('train.csv')
	print('Load train.csv!! train.csv shape: {}'.format(label_df.shape))

	# for reproduce
	SEED = 42
	seed_everyting(42)

	# stratify by bbox count & 사랑니의 유무[18,28,38,48] (class별로는 힘들어서 ㅠ)
	# bbox count의 경우 너무 다양하므로 크게 6가지로 나눠서 진행 (<=12, 13<= <=17, 18 <= <=23, 24<= <=27 , 28, 28 < )
	df_folds = label_df[['img_name']].copy()
	df_folds.loc[:, 'bbox_count'] = 1
	df_folds = df_folds.groupby('img_name').count() # 이미지별로 bbox의 개수 sum
	df_folds['bbox_label'] = df_folds['bbox_count'].apply(lambda x: bbox_label(x))
	df_folds['love_label'] = pd.Series({x: love_teeth(x, label_df) for x in df_folds.index})
	df_folds['group'] = df_folds['bbox_label'].astype('str') + '/' + df_folds['love_label'].astype('str')
	df_folds.loc[:,'fold'] = 0

	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
	for fold_num, (train_index, val_index) in enumerate(skf.split(X=df_folds.index,y=df_folds['group'])):
		# df_folds의 경우 index가 img_name으로 되어있어서 loc와 .index를 이렇게 사용
		df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_num

	for fold_num in range(5): # fold
		print('=============== Fold - {} ==============='.format(fold_num))
		# train data와 val data 불러오기
		train_dataset = LoadDataset(
			image_names = df_folds[df_folds['fold'] != fold_num].index.values,
			train_csv = label_df,
			imsize = args.imsize,
			which = 'train'
			)

		val_dataset = LoadDataset(
			image_names = df_folds[df_folds['fold'] == fold_num].index.values,
			train_csv = label_df,
			imsize = args.imsize,
			which = 'val'
			)

		# model load
		model = Train_Model(which_model=args.model, img_size=args.imsize)
		model.to(device)
		# dataset (train / valid)
		# train의 경우 마구잡이로 뽑고, valid의 경우 
		train_loader = DataLoader(train_dataset,batch_size = args.batch, num_workers = args.nw,
			sampler=RandomSampler(train_dataset), collate_fn=collate_fn)
		val_loader = DataLoader(val_dataset,batch_size = args.batch, num_workers = args.nw,
			sampler=SequentialSampler(val_dataset), collate_fn=collate_fn, shuffle=False)
		# training
		training = Training(model=model, device=device, config=setting_config, name=args.name, fold_num = fold_num, imsize=args.imsize)
		training.fit(train_loader, val_loader)

		with torch.cuda.device(device):
			torch.cuda.empty_cache()
