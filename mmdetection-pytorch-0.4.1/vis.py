import sys
sys.path.append('/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/mmcv-master')
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import json
import os
import numpy as np

pic_path = '/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_testA_20190818/'
config = './cloth_config/cloth_faster_rcnn_r101_fpn_1x.py'
# model_path = '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x/epoch_38.pth'
# json_path = '/home/remo/Desktop/cloth_flaw_detection/Results/result_2wei.json'
# model_path = '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_all_data/latest.pth'
# model_path = '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/cloth_faster_rcnn_r101_fpn_1x_2446_1000/latest.pth'
# model_path = '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/cloth_faster_rcnn_r101_fpn_1x_rcnn_ohem_add_crop/latest.pth'
model_path = '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_7/latest.pth'
json_path = '/home/remo/Desktop/cloth_flaw_detection/Results/result_test.json'

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def two_size(data):
    data = str(data)
    data = data.split('.')
    new_data = data[0]+'.'+ data[1][:2]
    return float(new_data)

cfg = mmcv.Config.fromfile(config)
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, model_path)
# test a single image
imgs = os.listdir(pic_path)
meta = []
from tqdm import tqdm
for im in tqdm(imgs):
    img = pic_path+im
    img = mmcv.imread(img)
    result = inference_detector(model, img, cfg)
    re = show_result(img,result,dataset='cloths',show = True)
    if len(re):
        for box in re:
            anno = {}
            anno['name'] = im
            anno['category'] = int(box[5])
            anno['bbox'] = [round(float(i),2) for i in box[0:4]]
            anno['score'] = float(box[4])
            meta.append(anno)
with open(json_path, 'w') as fp:
    json.dump(meta, fp,cls = MyEncoder)


