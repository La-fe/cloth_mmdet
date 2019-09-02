import sys
sys.path.append('/home/remo/cloth_mmdet/mmdetection-pytorch-0.4.1/mmcv-master')
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import json
import os
import numpy as np
import cv2



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

def vis():
    cfg = [mmcv.Config.fromfile(config_) for config_ in config]
    for cfg_ in cfg:
        cfg_.model.pretrained = None

    # construct the model and load checkpoint
    model = [build_detector(cfg_.model, test_cfg=cfg_.test_cfg) for cfg_ in cfg]
    _ = [load_checkpoint(model[i], model_path[i]) for i in range(len(model))]
    # test a single image
    imgs = os.listdir(pic_path)
    from tqdm import tqdm
    for im in tqdm(imgs):
        img = pic_path+im
        img = mmcv.imread(img)
        for i in range(len(model)):
            result = inference_detector(model[i], img, cfg[i])
            re,img = show_result(img,result,dataset='cloths',show = False)
            name = config[i].split('_')[-1].split('.')[0]
            cv2.namedWindow(str(name),0)
            cv2.resizeWindow(str(name),1920,1080)
            cv2.imshow(str(name),img)
        cv2.waitKey(0)



def result():
    cfg = mmcv.Config.fromfile(config2make_json)
    cfg.model.pretrained = None

    # construct the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, model2make_json)
    # test a single image
    imgs = os.listdir(pic_path)
    meta = []
    from tqdm import tqdm
    for im in tqdm(imgs):
        img = pic_path + im
        img = mmcv.imread(img)
        result = inference_detector(model, img, cfg)
        re,img = show_result(img, result, dataset='cloths', show=False)
        if len(re):
            for box in re:
                anno = {}
                anno['name'] = im
                anno['category'] = int(box[5])
                anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                anno['score'] = float(box[4])
                meta.append(anno)
    with open(json_path, 'w') as fp:
        json.dump(meta, fp, cls=MyEncoder)

if __name__ == "__main__":
    pic_path = '/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_testA_20190818/'
    config = ['./cloth_config/cloth_faster_rcnn_r101_fpn_1x_4.py',
              './cloth_config/cloth_faster_rcnn_r101_fpn_1x_4_test.py',
              # './cloth_config/cloth_faster_rcnn_r101_fpn_1x_5.py'
              ]
    model_path = ['/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_4/latest.pth',
                  '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_4/latest.pth',
                  # '/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_5/latest.pth'
                  ]

    model2make_json = "/home/remo/Desktop/cloth_flaw_detection/mmdete_ckpt/faster_rcnn_r101_fpn_1x_7/latest.pth"
    config2make_json = './cloth_config/cloth_faster_rcnn_r101_fpn_1x.py'
    json_path = '/home/remo/Desktop/cloth_flaw_detection/Results/result_test.json'
    # result()
    vis()



