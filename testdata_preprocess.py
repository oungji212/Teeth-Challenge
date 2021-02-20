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

# 똑같은 이미지 aug해서 여러개의 데이터 갖기
def make_Aug_Test():
	# test data
	test_imgs_path = '../../../DATA/data_teeth/test'
	os.makedirs('./dataset_Aug/images/test', exist_ok=True)
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
		save_img.save('./dataset_Aug/images/test/{}'.format(img_name))

if __name__ == '__main__':
	# /USER/USER_WORKSPACE/TEETH에서 실행
	if not os.path.isdir('./dataset_Aug/images/test'):
		make_Aug_Test()
			