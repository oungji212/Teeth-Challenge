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
import argparse

def convtori_label(yolo_class):
	teeth_labels = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
	return teeth_labels[yolo_class]

def yolo_to_teeth(cen_x, cen_y, w_r, h_r, H, W):
	cen_x_pixel = cen_x * W ; cen_y_pixel = cen_y * H
	box_width = w_r * W ; box_height = h_r * H
	x_min = round(cen_x_pixel - box_width/2)
	x_max = round(cen_x_pixel + box_width/2)
	y_min = round(cen_y_pixel - box_height/2)
	y_max = round(cen_y_pixel + box_height/2)
	return [x_min, y_min, x_max, y_max]

def coord_to_yolo(coords,H,W):
    x_min, x_max, y_min, y_max = coords[0], coords[2], coords[1], coords[3]
    center_x, center_y = (x_max+x_min)/2, (y_max+y_min)/2
    center_x_yolo, center_y_yolo = center_x/W, center_y/H
    width_yolo, height_yolo = (x_max-x_min)/W, (y_max-y_min)/H
    return [center_x_yolo, center_y_yolo, width_yolo, height_yolo]

# inference txt to submission file
def submission_file_all(infer_dir, json_name):
	infer_txts = os.listdir(infer_dir)
	now_path = os.getcwd()
	sub_json = {}
	for infer_txt in tqdm(infer_txts, desc='generate_submission'):
		img_name = '{}.png'.format(infer_txt.split('.')[0])
		sub_json[img_name] = []
		# img의 크기 얻기
		img_path = './dataset_Aug/images/test/{}'.format(img_name)
		img = np.array(Image.open(img_path))
		H, W = img.shape[:2] 
		# infer txt 읽기
		txt_path = os.path.join(now_path, infer_dir, infer_txt)
		open_f = open(txt_path,'r')
		lines = open_f.readlines() # list 형식
		open_f.close()
		# write json
		for line in lines:
			line = line.replace('\n','')
			# class x_center_norm y_center_norm width_norm height_norm conf
			yolo_cl, cen_x, cen_y, w_r, h_r, conf = line.split(' ')
			teeth_cl = convtori_label(int(yolo_cl))
			teeth_bboxes = yolo_to_teeth(float(cen_x), float(cen_y), float(w_r), float(h_r), H, W)

			class_json = {}
			class_json['class'] = teeth_cl
			class_json['confidence_score'] = float(conf)
			class_json['coord'] = teeth_bboxes
			sub_json[img_name].append(class_json)

	os.makedirs('./submission', exist_ok=True)
	sub_path = './submission/{}.json'.format(json_name)
	with open(sub_path, 'w') as json_file:
		json.dump(sub_json, json_file, ensure_ascii=False, indent=2)


# inference txt to submission file
# 두개 이상의 class를 detect한 경우에 대비
def submission_file_filter(infer_dir, json_name, choose):
	infer_txts = os.listdir(infer_dir)
	now_path = os.getcwd()
	sub_json = {}
	for infer_txt in tqdm(infer_txts, desc='generate_submission'):
		img_name = '{}.png'.format(infer_txt.split('.')[0])
		sub_json[img_name] = []
		# img의 크기 얻기
		img_path = './dataset_Aug/images/test/{}'.format(img_name)
		img = np.array(Image.open(img_path))
		H, W = img.shape[:2] 
		# infer txt 읽기
		txt_path = os.path.join(now_path, infer_dir, infer_txt)
		open_f = open(txt_path,'r')
		lines = open_f.readlines() # list 형식
		open_f.close()
		# write json
		class_ls = [] ; bboxes_ls = [] ; conf_ls = []
		for line in lines:
			line = line.replace('\n','')
			# class x_center_norm y_center_norm width_norm height_norm conf
			yolo_cl, cen_x, cen_y, w_r, h_r, conf = line.split(' ')
			teeth_cl = convtori_label(int(yolo_cl))
			teeth_bboxes = yolo_to_teeth(float(cen_x), float(cen_y), float(w_r), float(h_r), H, W)

			class_ls.append(teeth_cl)
			bboxes_ls.append(teeth_bboxes)
			conf_ls.append(float(conf))

		unique_teeths = list(set(class_ls))
		for teeth_num in unique_teeths:
			teeth_many = unique_teeths.count(teeth_num)
			if teeth_many == 1:
				ls_where = class_ls.index(teeth_num)
				class_json = {}
				class_json['class'] = teeth_num
				class_json['confidence_score'] = conf_ls[ls_where]
				class_json['coord'] = bboxes_ls[ls_where]
			else:
				ls_where = np.where(np.array(class_ls) == teeth_num)[0]
				if choose == 'conf':
					final_where = ls_where[np.array(conf_ls)[ls_where].argmax()]
					class_json = {}
					class_json['class'] = teeth_num
					class_json['confidence_score'] = conf_ls[final_where]
					class_json['coord'] = bboxes_ls[final_where]
				else:
					with open('./data-info.json') as f:
						json_data = json.load(f)
					final_info = json_data['final'][str(teeth_num)]
					spec_info = json_data['specific'][str(teeth_num)]

					min_cx_ratio = spec_info['min_cx_ratio']
					max_cx_ratio = spec_info['max_cx_ratio']
					min_cy_ratio = spec_info['min_cy_ratio']
					max_cy_ratio = spec_info['max_cy_ratio']
					mean_cx_ratio = final_info[cx_ratio]
					mean_cy_ratio = final_info[cy_ratio]

					final_wheres = []
					if choose == 'info_strong':
						for where in ls_where:
							where_bbox = bboxes_ls[where]
							cx_yolo, cy_yolo, _, _ = coord_to_yolo(where_bbox,H,W)
							if (cx_yolo < min_cx_ratio)|(cx_yolo > max_cx_ratio)|(cy_yolo < min_cy_ratio)|(cy_yolo > max_cy_ratio):
								pass
							else:
								final_wheres.append(where)

						if len(final_wheres) == 0:
							pass
						elif len(final_wheres) == 1:
							final_where = final_wheres[0]
						else: # 2개 이상인 경우 confidence로!
							final_where = final_wheres[np.array(conf_ls)[final_wheres].argmax()]

						class_json = {}
						class_json['class'] = teeth_num
						class_json['confidence_score'] = conf_ls[final_where]
						class_json['coord'] = bboxes_ls[final_where]

					elif choose == 'info_weak':
						cx_ls = [] ; cy_ls = []
						for where in ls_where:
							where_bbox = bboxes_ls[where]
							cx_yolo, cy_yolo, _, _ = coord_to_yolo(where_bbox,H,W)
							cx_ls.append(abs(cx_yolo-mean_cx_ratio)) ; cy_ls.append(abs(cy_yolo-mean_cy_ratio))
						diff = np.array(cx_ls)*np.array(cy_ls)
						final_where = final_wheres[diff.argmin()]

						class_json = {}
						class_json['class'] = teeth_num
						class_json['confidence_score'] = conf_ls[final_where]
						class_json['coord'] = bboxes_ls[final_where]

			sub_json[img_name].append(class_json)

	os.makedirs('./submission', exist_ok=True)
	sub_path = './submission/{}.json'.format(json_name)
	with open(sub_path, 'w') as json_file:
		json.dump(sub_json, json_file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='make submission json')
	parser.add_argument('--infer', type=str, help='infer directory')
	parser.add_argument('--name', type=str, help='submission file name')
	parser.add_argument('--choose', default='None', type=str, help='submission file name')
	args = parser.parse_args()

	if args.choose == 'None':
		submission_file_all(args.infer, args.name)
	else:
		submission_file_filter(args.infer, args.name, args.choose)




