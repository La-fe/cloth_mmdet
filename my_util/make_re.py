import sys
sys.path.append('../mmdetection-pytorch-0.4.1/mmcv-master')
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import json
import os
import numpy as np
import argparse

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
        re,img = show_result(img, result, dataset='cloths', show=False,score_thr = 0.5)
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
    parser = argparse.ArgumentParser(
        description="Generate result"
    )
    parser.add_argument(
        "-p", "--phase",
        default="test",
        help="Test val data or test data",
        type=str,
    )
    parser.add_argument(
        "-m", "--model",
        help="Model path",
        type=str,
    )
    parser.add_argument(
        "-c", "--config",
        help="Config path",
        type=str,
    )
    parser.add_argument(
        '-o',"--out",
        help="Save path",
        type=str,
    )
    args = parser.parse_args()

    if args.phase == 'test':
        pic_path = '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/guangdong1_round1_testA_20190818/'
    else:
        pic_path = '/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/COCO_format/Val_image/'
    model2make_json = args.model
    config2make_json = args.config
    json_path = args.out
    result()



