# a
train.py
--model : efficientdet의 몇번째 version을 쓸 것인가 (defult=efficientdet4)
--imsize : 모델에 들어갈 이미지 사이즈
--batch: 배치사이즈
--nw: num_workers
--name: training시킨 weight을 저장할 때 사용하는 이름
** 위는 default로 고정. 아래의 경우만 수정 **
--data : 원본 데이터의 위치
--label : 원본 라벨 데이터의 위치
--weights : 기본이 되는 effidet들의 pre-trained weights이 저장되어 있는 장소

inference.py
--obj_thr : wbf의 threshold1
--iou_thr : wbf의 threshold2 
--skip_box_thr : wbf의 threshold3
--model : train에서 사용한 efficientdet의 version
--imsize : 모델에 들어갈 이미지 사이즈
--batch: 배치사이즈
--nw: num_workers
--ckp_nm: training시킨 weight의 base가 되는 이름 (train.py의 --name과 같아야 함)
** 위는 default로 고정. 아래의 경우만 수정 **
--data : 원본 데이터의 위치
--result : 결과 json 생성 위치

# b
점수: 0.9679727200 / 등수: 5

# c
stratified kfold에 따라 5개의 fold로 나눈 후
각기 다른 efficientdet4를 훈련시킴. 즉 총 5개의 effidet4 훈련.
이후 5개의 effidet4이 predict한 결과를 nms가 아닌 wbf를 통해 통합

# d
pytorch-efficientdet를 깃헙을 통해 다운받은 후 src에 풀어주면 됨