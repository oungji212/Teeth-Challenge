# for yolov5
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

# image & label preprocess 
# /USER/USER_WORKSPACE/TEETH에서 실행
# 이미지 transform X (후에 특이 적은 수의 이가 있거나 치아 개수 자체가 있는 사진들은 aug한 다음에 train/valid img folder에 추가)
def make_nochange_TrainVal(): 
	# train data
	train_json = '../../../DATA/data_teeth/train_label.json' 
	train_imgs_path = '../../../DATA/data_teeth/train'
	with open(train_json) as f:
		json_data = json.load(f)

	img_names = list(json_data.keys())
	name_ls = [] ; num_ls = [] # 치아 개수
	unique_imgs = []# 이미지가 하나밖에 없는 경우 split이 안 되므로 해당 이미지들은 train OR val에다가 직접 넣음
	for img_name in img_names:
		if len(json_data[img_name]) not in [5,8,11]: # 치아의 개수가 5,8,11개인 경우는 하나밖에 없음
			name_ls.append(img_name)
			num_ls.append(len(json_data[img_name]))
		else:
			unique_imgs.append(img_name)

	os.makedirs('./dataset_nochange/images/train', exist_ok=True)
	os.makedirs('./dataset_nochange/labels/train', exist_ok=True)
	os.makedirs('./dataset_nochange/images/val', exist_ok=True)
	os.makedirs('./dataset_nochange/labels/val', exist_ok=True)

	img_names = list(json_data.keys())
	train_imgs, val_imgs, _, _ = train_test_split(np.array(name_ls), np.array(num_ls), test_size=0.1, random_state=42, stratify=np.array(num_ls))
	for img_name in unique_imgs:
		if len(json_data[img_name]) in [5,11]:
			val_imgs = np.concatenate((val_imgs, np.array([img_name])))
		else:
			train_imgs = np.concatenate((train_imgs, np.array([img_name])))

	for which in ['train','val']:
		if which == 'train': folder_imgs = train_imgs
		else: folder_imgs =  val_imgs
		for img_name in tqdm(folder_imgs,desc='{}'.format(which)):
			# img
			img_path = train_imgs_path+'/'+img_name
			img_gray = np.array(Image.open(img_path).convert('L')) 
			img_h, img_w = img_gray.shape[0], img_gray.shape[1]
			# eda.py를 바탕으로 크롭할 부분 지정하기 (나중에)
			clahe1 = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
			clahe2 = cv2.createCLAHE(clipLimit=2, tileGridSize=(30,30))
			wb = cv2.xphoto.createSimpleWB() ; wb.setP(0.4)
			img_cl1 = clahe1.apply(img_gray)
			img_cl2 = clahe2.apply(img_gray)
			img_wb = wb.balanceWhite(img_cl2)
			img_wb_cl = clahe1.apply(img_wb)
			final = np.dstack((img_wb_cl, img_cl2, img_cl1))
			save_img = Image.fromarray(final.astype(np.uint8))
			save_img.save('./dataset_nochange/images/{}/{}'.format(which,img_name))

			# label
			img_data = json_data[img_name]
			label_path = './dataset_nochange/labels/{}/{}.txt'.format(which,img_name.split('.')[0])
			f = open(label_path, 'w')
			for data in img_data:
				teeth_class = data['class']
				teeth_coord = data['coord']
				x_min, x_max, y_min, y_max = teeth_coord[0], teeth_coord[2], teeth_coord[1], teeth_coord[3]
				center_x, center_y = (x_max+x_min)/2, (y_max+y_min)/2
				center_x_yolo, center_y_yolo = center_x/img_w, center_y/img_h
				width_yolo, height_yolo = (x_max-x_min)/img_w, (y_max-y_min)/img_h
				yolo_label = convtoyolo_label(teeth_class)
				f.write('{} {} {} {} {}\n'.format(yolo_label, center_x_yolo, center_y_yolo, width_yolo, height_yolo))
			f.close()


# image & label preprocess 
# /USER/USER_WORKSPACE/TEETH에서 실행
# 이미지 transform X (후에 특이 적은 수의 이가 있거나 치아 개수 자체가 있는 사진들은 aug한 다음에 train/valid img folder에 추가)
# yolov5는 알아서 best.pt를 save하므로 그냥 train시키자
def make_nochange_onlyTrain(): 
	# train data
	train_json = '../../../DATA/data_teeth/train_label.json' 
	train_imgs_path = '../../../DATA/data_teeth/train'
	with open(train_json) as f:
		json_data = json.load(f)

	os.makedirs('./dataset_nochange/images/train', exist_ok=True)
	os.makedirs('./dataset_nochange/labels/train', exist_ok=True)

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
		save_img.save('./dataset_nochange/images/train/{}'.format(img_name))

		# label
		img_data = json_data[img_name]
		label_path = './dataset_nochange/labels/train/{}.txt'.format(img_name.split('.')[0])
		f = open(label_path, 'w')
		for data in img_data:
			teeth_class = data['class']
			teeth_coord = data['coord']
			x_min, x_max, y_min, y_max = teeth_coord[0], teeth_coord[2], teeth_coord[1], teeth_coord[3]
			center_x, center_y = (x_max+x_min)/2, (y_max+y_min)/2
			center_x_yolo, center_y_yolo = center_x/img_w, center_y/img_h
			width_yolo, height_yolo = (x_max-x_min)/img_w, (y_max-y_min)/img_h
			yolo_label = convtoyolo_label(teeth_class)
			f.write('{} {} {} {} {}\n'.format(yolo_label, center_x_yolo, center_y_yolo, width_yolo, height_yolo))
		f.close()


# # data aug
# def blur_img(img):
# 	blurred=ndimage.gaussian_filter(img,sigma=3)
# 	return blurred

# def cutout_img(img):
# 	img2 = img.copy()
#     h,w = img.shape[0], img.shape[1]
#     cutnum = random.randint(5,9)
#     size = 100
#     for num in range(cutnum):
#         x_min = random.randint(1,(w-size-1))
#         y_min = random.randint(1,(h-size-1))
#         img2[y_min:(y_min+size),x_min:(x_min+size),:] = 0
#     return img2

# def shift_img(img):
#     img2 = img.copy()
#     h,w = img.shape[0], img.shape[1]
#     side = int(w*(random.uniform(-0.031,0.031)))
#     updown = int(h*(random.uniform(-0.06,0.06)))
#     print(side, updown)
#     if side < 0: 
#         img2 = np.concatenate((img2[:,-side:,:],np.zeros((h,-side,3))),axis=1)
#     elif side > 0:
#         img2 = np.concatenate((np.zeros((h,side,3)),img2[:,:(w-side),:]),axis=1)
#     if updown < 0:
#         img2 = np.concatenate((img2[-updown:,:,:],np.zeros((-updown,w,3))),axis=0)
#     elif updown > 0:
#         img2 = np.concatenate((np.zeros((updown,w,3)),img2[:(h-updown),:,:]),axis=0)
#     shifted = img2.astype('uint8')
#     return shifted, side, updown # label 역시 shift해야함

def train_transforms(p):
	return A.Compose([
		A.GaussianBlur(blur_limit=(3,5), p=p),
		A.RandomBrightnessContrast(p=p),
		A.ShiftScaleRotate(shift_limit=0.035, scale_limit=0.03, rotate_limit=10, p=p),
		A.Cutout(num_holes=8, max_h_size=100, max_w_size=100, p=p),
		], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def coord_to_yolo(coords,H,W):
    x_min, x_max, y_min, y_max = coords[0], coords[2], coords[1], coords[3]
    center_x, center_y = (x_max+x_min)/2, (y_max+y_min)/2
    center_x_yolo, center_y_yolo = center_x/W, center_y/H
    width_yolo, height_yolo = (x_max-x_min)/W, (y_max-y_min)/H
    return [center_x_yolo, center_y_yolo, width_yolo, height_yolo]

def convtoyolo_label(teeth_class):
	teeth_labels = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
	return teeth_labels.index(teeth_class)

def convtori_label(yolo_class):
	teeth_labels = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
	return teeth_labels[yolo_class]

def alb_bbox_labels(coord_ls,H,W): # Alb의 bbox format으로 + 추가할 이미지 개수
	bboxes = [] ; yolo_labels = [] ; teeth_labels = []
	for i in range(len(coord_ls)):
		class_teeth = coord_ls[i]['class'] ; teeth_labels.append(class_teeth)
		class_yolo = convtoyolo_label(class_teeth)
		yolo_labels.append(class_yolo)

		coords = coord_ls[i]['coord']
		coord_yolo = coord_to_yolo(coords,H,W)
		bboxes.append(coord_yolo)
	# 추가할 이미지의 개수
	gen_imgs = 0
	if len(set(teeth_labels)&set([18,28,38,48])) != 0: # 사랑니인경우
		gen_imgs += 1
	if len(teeth_labels) <= 11: # 총 치아개수가 적은 이미지의 경우
		gen_imgs += 4
	elif len(teeth_labels) <= 18:
		gen_imgs += 3
	elif len(teeth_labels) <=  23:
		gen_imgs += 2
	elif len(teeth_labels) <=  27:
		gen_imgs += 1
	elif len(teeth_labels) == 28: # 보통의 경우. 아마 대부분이 사랑니 제외한 나머지 이만 가지고 있는 경우 일 듯
		gen_imgs += 0.5
	else: # 29 ~ 32개(사랑니 포함. 이미 사랑니인 경우 +1 했으므로 굳이 더 필요 X)
		gen_imgs += 0
	return bboxes, yolo_labels, gen_imgs

def transform_prob(num, total_num):
	if num == 0:
		return 0.1
	else:
		return num / (total_num+1)

def img_label_save(image, bboxes, class_labels, img_num, img_name): # 양쪽 사랑니거나 총 치아개수가 적은 이미지의 경우 다른 경우에 비해 img 더 늘림
	if img_num in [0.5, 1.5]:
		total_num = round(img_num + random.uniform(-0.5,0.5))
	else:
		total_num = img_num # total_num은 추가로 생성할 이미지의 개수

	label_base_path = './dataset_Aug/labels/train/'
	img_base_path = './dataset_Aug/images/train/'

	if total_num == 0:
		# img save
		save_img = Image.fromarray(image.astype(np.uint8))
		img_path = img_base_path + '{}'.format(img_name)
		save_img.save(img_path)
		# label save
		label_path = label_base_path + '{}.txt'.format(img_name.split('.')[0])
		f = open(label_path, 'w')
		for i in range(len(class_labels)):
			yolo_label = class_labels[i]
			yolo_coords = bboxes[i]
			center_x_yolo, center_y_yolo, width_yolo, height_yolo = yolo_coords[0], yolo_coords[1], yolo_coords[2], yolo_coords[3]
			f.write('{} {} {} {} {}\n'.format(yolo_label, center_x_yolo, center_y_yolo, width_yolo, height_yolo))
		f.close()
	else:
		for num in range(total_num+1):
			p = transform_prob(num, total_num)
			transform = train_transforms(p)
			transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels) # yolo_bboxes
			transformed_image = transformed['image']
			transformed_bboxes = transformed['bboxes']
			transformed_class_labels = transformed['class_labels']
			# img save
			save_img = Image.fromarray(transformed_image)
			img_path = img_base_path + '{}_{}.png'.format(img_name.split('.')[0],num)
			save_img.save(img_path)
			# label save
			label_path = label_base_path + '{}_{}.txt'.format(img_name.split('.')[0],num)
			f = open(label_path, 'w')
			for i in range(len(transformed_class_labels)):
				yolo_label = transformed_class_labels[i]
				yolo_coords = transformed_bboxes[i]
				center_x_yolo, center_y_yolo, width_yolo, height_yolo = yolo_coords[0], yolo_coords[1], yolo_coords[2], yolo_coords[3]
				f.write('{} {} {} {} {}\n'.format(yolo_label, center_x_yolo, center_y_yolo, width_yolo, height_yolo))
			f.close()

# 똑같은 이미지 aug해서 여러개의 데이터 갖기
def make_Aug_onlyTrain():
		# train data
	train_json = '../../../DATA/data_teeth/train_label.json' 
	train_imgs_path = '../../../DATA/data_teeth/train'
	with open(train_json) as f:
		json_data = json.load(f)

	os.makedirs('./dataset_Aug/images/train', exist_ok=True)
	os.makedirs('./dataset_Aug/labels/train', exist_ok=True)

	img_names = list(json_data.keys())
	for img_name in tqdm(img_names,desc='{}'.format('train_preprocess')):
		# 기본 img
		img_path = train_imgs_path+'/'+img_name
		img_gray = np.array(Image.open(img_path).convert('L')) 
		img_h, img_w = img_gray.shape[0], img_gray.shape[1]
		clahe1 = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
		clahe2 = cv2.createCLAHE(clipLimit=2, tileGridSize=(30,30))
		img_cl1 = clahe1.apply(img_gray)
		img_cl2 = clahe2.apply(img_gray)
		img_cl_twice = clahe1.apply(img_cl2)
		final = np.dstack((img_cl_twice, img_cl2, img_cl1))

		# 기본 img의 기본 bbox (yolo ver.)
		img_data = json_data[img_name]
		yolo_bboxes, yolo_classes, img_num = alb_bbox_labels(img_data,img_h,img_w)
		
		# img & label save
		img_label_save(final, yolo_bboxes, yolo_classes, img_num, img_name)


if __name__ == '__main__':
	# /USER/USER_WORKSPACE/TEETH에서 실행
	make_Aug_onlyTrain()
			

