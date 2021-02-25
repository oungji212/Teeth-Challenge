import numpy as np
import pandas as pd
import json
from PIL import Image
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy import ndimage
import random
import albumentations as A

def make_nochange_onlyTrain(data_path, label_path): 
	os.makedirs('./src/images/train', exist_ok=True)
	# train data
	train_imgs_path = data_path
	with open(label_path) as f:
		json_data = json.load(f)

	img_names = list(json_data.keys())
	for img_name in tqdm(img_names,desc='{}'.format('train_preprocess')):
		# img
		img_path = train_imgs_path+'/'+img_name
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
		save_img.save('./src/images/train/{}'.format(img_name)) # ./dataset_nochange/images/train/{}

def make_Aug_Test(data_path):
	os.makedirs('./src/images/test', exist_ok=True)
	# test data
	test_imgs_path = data_path # '../../../DATA/data_teeth/test'
	img_names = os.listdir(test_imgs_path)
	for img_name in tqdm(img_names,desc='{}'.format('test_preprocess')):
		# 기본 img
		img_path = test_imgs_path+'/'+img_name
		img_gray = np.array(Image.open(img_path).convert('L')) 
		img_h, img_w = img_gray.shape[0], img_gray.shape[1]
		clahe1 = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
		clahe2 = cv2.createCLAHE(clipLimit=2, tileGridSize=(30,30))
		img_cl1 = clahe1.apply(img_gray)
		img_cl2 = clahe2.apply(img_gray)
		img_cl_twice = clahe1.apply(img_cl2)
		final = np.dstack((img_cl_twice, img_cl2, img_cl1))
		save_img = Image.fromarray(final.astype(np.uint8))
		save_img.save('./src/images/test/{}'.format(img_name))