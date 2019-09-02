import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

cfg = mmcv.Config.fromfile('./my_configs/cloth_faster_rcnn_r101_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, './experiments/latest.pth')
# test a single image
import os
imgs = os.listdir("/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/Raw/Images")
for im in imgs[:3]:
    img = "/home/zhangming/Models/Results/cloth_flaw_detection/Datasets/Raw/Images/"+im
    img = mmcv.imread(img)
    result = inference_detector(model, img, cfg)
    show_result(img, result)
