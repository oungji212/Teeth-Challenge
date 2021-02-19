import numpy as np
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm

# /USER/USER_WORKSPACE/TEETH에서 실행
def train_eda(): 
	# train data
	train_json = '../../../DATA/data_teeth/train_label.json' # 경로바꾸기
	train_imgs_path = '../../../DATA/data_teeth/train'
	with open(train_json) as f:
		json_data = json.load(f)

	class_stat = {}
	teeth_label = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,
	31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
	for label in teeth_label:
		class_stat[label] = {'w_ratio':[], 'h_ratio':[], 'cx_ratio':[],'cy_ratio':[], 'size':[]}

	img_names = list(json_data.keys())
	for img_name in tqdm(img_names):
		teeth_img_path = train_imgs_path + '/' + img_name
		# 기본 이미지가 똑같은 channel 3개가 concat되어있는 것. -> 하나의 채널을 다양하게 이용할거임
		teeth_img = np.array(Image.open(teeth_img_path).convert('L'))  
		img_h, img_w = teeth_img.shape[0], teeth_img.shape[1]

		img_data = json_data[img_name]
		max_bottom_ratio = 0 ; max_top_ratio = 1 ; max_left_ratio = 1 ; max_right_ratio = 0
		for data in img_data:
			teeth_class = data['class']
			theeth_coord = data['coord']
			
			x_min, x_max, y_min, y_max = theeth_coord[0], theeth_coord[2], theeth_coord[1], theeth_coord[3]
			h_top_pos, h_bottom_pos = y_min/img_h, y_max/img_h # pos: 위치의 비율
			w_left_pos, w_right_pos = x_min/img_w, x_max/img_w
			teeth_center_x, teeth_center_y = (x_max+x_min)/2, (y_max+y_min)/2
			h_center_pos, w_center_pos = teeth_center_y/img_h, teeth_center_x/img_w
			size = ((x_max-x_min)*(y_max-y_min))/(img_h*img_w)

			class_stat[teeth_class]['w_ratio'].append((x_max-x_min)/img_w)
			class_stat[teeth_class]['h_ratio'].append((y_max-y_min)/img_h)
			class_stat[teeth_class]['cx_ratio'].append(w_center_pos)
			class_stat[teeth_class]['cy_ratio'].append(h_center_pos)
			class_stat[teeth_class]['size'].append(size)

			if max_bottom_ratio < h_bottom_pos:
				max_bottom_ratio = h_bottom_pos
			if max_top_ratio > h_top_pos:
				max_top_ratio = h_top_pos
			if max_left_ratio > w_left_pos:
				max_left_ratio = w_left_pos
			if max_right_ratio < w_right_pos:
				max_right_ratio = w_right_pos

	final_stat = {}
	final_stat['final'] = {} ; final_stat['specific'] = {} 
	final_stat['final']['max_bottom'] = max_bottom_ratio
	final_stat['final']['max_top'] = max_top_ratio
	final_stat['final']['max_left'] = max_left_ratio
	final_stat['final']['max_right'] = max_right_ratio
	print('max-bottom:{} / max-top:{} / max-left: {} / max-right:{}'.format(max_bottom_ratio, max_top_ratio, max_left_ratio, max_right_ratio))
	print('='*50)
	for label in teeth_label:
		w_ratio_ls = class_stat[label]['w_ratio'] 
		h_ratio_ls = class_stat[label]['h_ratio'] 
		cx_ratio_ls = class_stat[label]['cx_ratio'] 
		cy_ratio_ls = class_stat[label]['cy_ratio'] 
		size_ls = class_stat[label]['size']  
		# final_stat[final] - for crop & show
		final_stat['final'][label] = {'w_ratio':np.mean(w_ratio_ls), 'h_ratio':np.mean(h_ratio_ls), 
		'cx_ratio':np.mean(cx_ratio_ls),'cy_ratio':np.mean(cy_ratio_ls), 'size':np.mean(size_ls)}
		print('label:{} - cx:{} / cy:{} / size:{}'.format(label, np.mean(cx_ratio_ls), np.mean(cy_ratio_ls), np.mean(size_ls)))
		# final_stat[specific] - for image aug (copy & paste)
		final_stat['specific'][label] = {'min_w_ratio':min(w_ratio_ls), 'max_w_ratio':max(w_ratio_ls),
		'min_h_ratio':min(h_ratio_ls), 'max_h_ratio':max(h_ratio_ls),
		'min_cx_ratio':min(cx_ratio_ls), 'max_cx_ratio':max(cx_ratio_ls),
		'min_cy_ratio':min(cy_ratio_ls), 'max_cy_ratio':max(cy_ratio_ls),
		'min_size':min(size_ls), 'max_size':max(size_ls)}
	with open('data_info.json','w',encoding='utf-8') as make_file:
		json.dump(final_stat, make_file, ensure_ascii=False, indent='\t')


# 각 이미지당 몇개의 이가 존재하는가
# /USER/USER_WORKSPACE/TEETH에서 실행
def train_class_num(): 
	# train data
	train_json = '../../../DATA/data_teeth/train_label.json' # 경로바꾸기
	with open(train_json) as f:
		json_data = json.load(f)

	num_dict = {}
	teeth_label = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,
	31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
	for num in range(1,len(teeth_label)+1):
		num_dict[num] = 0

	img_names = list(json_data.keys())
	for img_name in tqdm(img_names, len=len(img_names)):
		teeth_nums = len(json_data[img_name])
		num_dict[teeth_nums] += 1
	
	for num in range(1,len(teeth_label)+1):
		print('치아의 개수가 "{}"개인 이미지 개수 - {}'.format(num, num_dict[num]))

if __name__ == '__main__':
	# /USER/USER_WORKSPACE/TEETH에서 실행
	train_eda()
	train_class_num()







