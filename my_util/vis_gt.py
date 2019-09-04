# encoding:utf/8
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import os
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xml.dom.minidom
from xml.dom.minidom import Document
from tqdm import tqdm
from easydict import EasyDict as edict
import os.path as osp
import math
from tqdm import tqdm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import getpass  # 获取用户名

USER = getpass.getuser()


def load_file(path, cat=None):
    anno = {}
    train_anno = {}
    val_anno = {}
    labels = {}
    # num_each_class = {}
    area = []

    # label2img = {}
    for i in range(len(path)):

        f = json.load(open(path[i], 'r'))  # return list
        total = 0
        total_ = len(f)
        for info in f:
            im_name = info['name']
            label = info['defect_name']
            bbox = info['bbox']
            area.append((int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1])))
            if label not in labels.keys():
                labels[label] = 1
            else:
                labels[label] += 1

            if im_name not in anno.keys():
                anno[im_name] = [[bbox, label]]
            else:
                anno[im_name].append([bbox, label])

            # 划分训练集和验证集 4:1
            total += 1
            if total < (total_ / 5 * 4):
                if im_name not in train_anno.keys():
                    train_anno[im_name] = [[bbox, label]]
                else:
                    train_anno[im_name].append([bbox, label])
            else:
                if im_name not in val_anno.keys():
                    val_anno[im_name] = [[bbox, label]]
                else:
                    val_anno[im_name].append([bbox, label])
            # if cat[label] not in num_each_class.keys():
            #     num_each_class[cat[label]] = 1
            # else:
            #     num_each_class[cat[label]] += 1
            # if cat[label] not in label2img.keys():
            #     label2img[cat[label]] = [im_name]
            # else:
            #     label2img[cat[label]].append(im_name)

    # pro = {}
    # for i in num_each_class.keys():
    #     pro[i] = int(num_each_class[i] / total * 500)
    # pro = sorted(pro.items(), key=lambda x: x[1], reverse=False)
    # added_img = []
    # val = []
    # for ite in pro:
    #     label = ite[0]
    #     num  =ite[1]
    #     num = min(num,len(label2img[label]))
    #     for i in range(num):
    #         val.append()

    # # 划分训练验证集 按概率取
    # for im_name in anno.keys():
    #     boxes = anno[im_name]
    #     flag = True
    #     pro_ = pro
    #     for box in boxes:
    #         if pro[cat[box[1]]] <= 0:
    #             flag = False
    #             break
    #         else:
    #             pro[cat[box[1]]] -= 1
    #     if flag:
    #         val_anno[im_name] = boxes
    #     else:
    #         train_anno[im_name] = boxes
    #         pro = pro_

    return anno, train_anno, val_anno, labels, area


def vis_gt(anno, train_anno, val_anno, flag_make_xml, xml_root, cat):
    # for im in tqdm(anno.keys()):
    for im in tqdm(os.listdir(all_Images)):
        meta = {}
        image = cv2.imread(all_Images + im)
        w = image.shape[1]
        h = image.shape[0]
        meta['image_path'] = im
        meta['dataset'] = "cloths"
        meta['width'] = w
        meta['height'] = h
        meta['boxes'] = []
        if im in anno.keys():
            if im in train_anno.keys():
                flag = "train"
            else:
                flag = 'val'
            bboxes = anno[im]
            for bbox in bboxes:
                box_anno = {}
                box = bbox[0]
                label = bbox[1]
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                box_anno['xmin'] = x1
                box_anno['ymin'] = y1
                box_anno['xmax'] = x2
                box_anno['ymax'] = y2
                box_anno['label'] = label
                meta['boxes'].append(box_anno)
                if not flag_make_xml:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, '%d' % cat[label], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
            if flag_make_xml:
                root = xml_root + flag + '/'
                name = im.replace('.jpg', '.xml')
                write_xml(meta, root + name, cat)
            else:
                image = cv2.resize(image, (1536, 768))
                cv2.imshow('im', image)
                cv2.waitKey(0)


def draw(label_dict, cat):
    label = []
    num = []
    num_box = 0
    for l in label_dict.keys():
        label.append(l)
        num.append(int(label_dict[l]))
        num_box += int(label_dict[l])
    myfont = matplotlib.font_manager.FontProperties(fname='/home/xjx/simkai_downcc/simkai.ttf')
    # plt.xticks(range(len(label)), label,font_properties=myfont,rotation=-60)
    # # 柱状图
    # plt.bar(label, num, color='rgb')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    label_group = [0] * 20
    label = [i for i in range(1, 21)]
    for l in label_dict.keys():
        label_group[int(cat[l]) - 1] += int(label_dict[l])
    plt.xticks(range(1, len(label) + 1), label, font_properties=myfont, rotation=0)
    # 柱状图
    plt.bar(label, label_group, color='rgb')
    plt.legend()
    plt.grid(True)
    plt.show()


def draw_area(area):
    plt.hist(area)
    plt.xlabel("area")
    plt.ylabel("number")
    plt.show()


def write_xml(anno, xml_name, cat):
    # Create the minidom document
    doc = Document()

    # Create the root element: annotation
    anno_root = doc.createElement("annotation")
    doc.appendChild(anno_root)

    # folder
    folder_node = doc.createElement("folder")
    folder = doc.createTextNode('VOC2012')
    folder_node.appendChild(folder)
    anno_root.appendChild(folder_node)

    # filename
    filename_node = doc.createElement("filename")
    # filename = doc.createTextNode(anno['image_path'].split('/')[1])
    filename = doc.createTextNode(anno['image_path'])
    filename_node.appendChild(filename)
    anno_root.appendChild(filename_node)

    # source
    source_node = doc.createElement("source")
    database_node = doc.createElement("database")
    database = doc.createTextNode('The VOC2007 Database')
    database_node.appendChild(database)
    annotation_node = doc.createElement("annotation")
    annotation = doc.createTextNode(anno['dataset'])
    annotation_node.appendChild(annotation)
    image_node = doc.createElement("image")
    image = doc.createTextNode('flickr')
    image_node.appendChild(image)
    source_node.appendChild(database_node)
    source_node.appendChild(annotation_node)
    source_node.appendChild(image_node)
    anno_root.appendChild(source_node)

    # size
    size_node = doc.createElement("size")
    width_node = doc.createElement("width")
    width = doc.createTextNode(str(anno['width']))
    width_node.appendChild(width)
    height_node = doc.createElement("height")
    height = doc.createTextNode(str(anno['height']))
    height_node.appendChild(height)
    depth_node = doc.createElement("depth")
    depth = doc.createTextNode('3')
    depth_node.appendChild(depth)
    size_node.appendChild(width_node)
    size_node.appendChild(height_node)
    size_node.appendChild(depth_node)
    anno_root.appendChild(size_node)

    # segmented
    segmented_node = doc.createElement("segmented")
    segmented = doc.createTextNode('1')
    segmented_node.appendChild(segmented)
    anno_root.appendChild(segmented_node)

    if "boxes" in anno.keys():
        for box in anno["boxes"]:
            object_node = doc.createElement("object")
            name_node = doc.createElement("name")
            name = doc.createTextNode(str(cat[box['label']]))
            name_node.appendChild(name)
            object_node.appendChild(name_node)
            pose_node = doc.createElement("pose")
            pose = doc.createTextNode('pose')
            pose_node.appendChild(pose)
            object_node.appendChild(pose_node)
            truncated_node = doc.createElement("truncated")
            truncated = doc.createTextNode('0')
            truncated_node.appendChild(truncated)
            object_node.appendChild(truncated_node)
            diffcult_node = doc.createElement("difficult")
            diffcult = doc.createTextNode('0')
            diffcult_node.appendChild(diffcult)
            object_node.appendChild(diffcult_node)
            bndbox_node = doc.createElement("bndbox")

            xmin_node = doc.createElement("xmin")
            xmin = doc.createTextNode(str(box['xmin']))
            xmin_node.appendChild(xmin)
            bndbox_node.appendChild(xmin_node)
            ymin_node = doc.createElement("ymin")
            ymin = doc.createTextNode(str(box['ymin']))
            ymin_node.appendChild(ymin)
            bndbox_node.appendChild(ymin_node)
            xmax_node = doc.createElement("xmax")
            xmax = doc.createTextNode(str(box['xmax']))
            xmax_node.appendChild(xmax)
            bndbox_node.appendChild(xmax_node)
            ymax_node = doc.createElement("ymax")
            ymax = doc.createTextNode(str(box['ymax']))
            ymax_node.appendChild(ymax)
            bndbox_node.appendChild(ymax_node)

            object_node.appendChild(bndbox_node)
            anno_root.appendChild(object_node)

    with open(xml_name, "wb") as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))


def make_txt(anno, path2txt, image_root, cat):
    f = open(path2txt, 'w')
    for image in anno.keys():
        image_path = image_root + image
        bboxes = anno[image]
        for boxes in bboxes:
            box = boxes[0]
            label = boxes[1]
            f.write(image_path + ',' + str(box[0]).split('.')[0] + ',' + str(box[1]).split('.')[0] + ',' +
                    str(box[2]).split('.')[0] + ',' + str(box[3]).split('.')[0] + ',' + str(label) + '\n')
    f.close()


# # from easydict import EasyDict as edict
# class edict(EasyDict):
#     def __init__(self,d=None,**kwargs):
#         self.bbox = []
#         self.defect_name = ''
#         self.name = ''
#         self.abs_path = ''
#         self.w = 0
#         self.h = 0
#         self.area = 0
#         self.classes = 0
#
#         super(edict, self).__init__(d=None,**kwargs)
#         if d is None:
#             d = {}
#         if kwargs:
#             d.update(**kwargs)
#         for k, v in d.items():
#             setattr(self, k, v)
#         # Class attributes
#         for k in self.__class__.__dict__.keys():
#             if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
#                 setattr(self, k, getattr(self, k))

class Config:
    def __init__(self):
        self.json_paths = ['']
        self.allimg_path = ''


class DataAnalyze:
    '''
    bbox 分析类，
        1. 每一类的bbox 尺寸统计
        2.
    '''

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.category = {
            '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
            '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
            '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
            '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
        }
        self.num_classes = 20  # 前景类别

        self.all_instance, self.cla_instance, self.img_instance = self._create_data_dict()
        '''
        all_instance 
        [
            {'bbox': [2000.66, 326.38, 2029.87, 355.59],
             'defect_name': '结头',
             'name': 'd6718a7129af0ecf0827157752.jpg',
             'abs_path' : 'xxx/xxx.jpg',
             'classes': 1
             'w':1,
             'h':1,
             'area':1,
             }
        ]

        cla_instance 
            {'1':[], '2':[] }
        '''

        self.num_data = len(self.all_instance)

    def _create_data_dict(self):
        '''

        :return:
            instance:
                {'bbox': [2000.66, 326.38, 2029.87, 355.59],
                 'defect_name': '结头',
                 'name': 'd6718a7129af0ecf0827157752.jpg',
                 'abs_path' : 'xxx/xxx.jpg',
                 'w':1,
                 'h':1,
                 'area':1,
                 'im_w':1
                 'im_h':2
                 }

        all_instance
            [instance1, instance2, instance3]

        cla_instance
            {'1':[instance, instances2], '2'[instance, ]}

        img_instance
            {'xx1.jpg': [instance]  'xxx.jpg':[instance, instance]}

        '''

        all_instance = []
        key_classes = list(range(1, self.num_classes + 1))  # 1 ... num_classes

        cla_instance = edict({str(k): [] for k in key_classes})  # key 必须是字符串
        img_instance = edict()
        if isinstance(self.cfg.json_paths, str):
            self.cfg.json_paths = [self.cfg.json_paths]

        if isinstance(self.cfg.json_paths, list):
            for path in self.cfg.json_paths:
                gt_list = json.load(open(path, 'r'))
                for instance in gt_list:
                    instance = edict(instance)
                    instance.classes = int(self.category[instance.defect_name])  # add classes int
                    w, h = compute_wh(instance.bbox)
                    instance.w = round(w, 2)  # add w
                    instance.h = round(h, 2)  # add h
                    instance.area = round(w * h, 2)  # add area
                    instance.abs_path = osp.join(self.cfg.allimg_path, instance.name)  # add 绝对路径
                    all_instance.append(instance)  # 所有instance

                    cla_instance[str(instance.classes)].append(instance)  # 每类的instance

                    if instance.name not in img_instance.keys():  # 每张图片的instance
                        img_instance[instance.name] = [instance]
                    else:
                        img_instance[instance.name].append(instance)

        return all_instance, cla_instance, img_instance

    def ana_classes(self):
        ws_all = []
        hs_all = []
        for cla_name, bboxes_list in self.cla_instance.items():  # str '1': [edict()]
            ws = []
            hs = []
            for instance in bboxes_list:
                ws.append(instance.w)
                hs.append(instance.h)
                ws_all.append(instance.w)
                hs_all.append(instance.h)
            # plt.title(cla_name, fontsize='large',fontweight = 'bold')
            # plt.scatter(ws, hs, marker='x', label=cla_name, s=30)
        plt.scatter(ws_all, hs_all, marker='x', s=30)

        plt.grid(True)
        plt.show()

    def draw_cls(self):
        myfont = matplotlib.font_manager.FontProperties(fname='/home/remo/Desktop/simkai_downcc/simkai.ttf')
        cls = [i for i in range(1,self.num_classes+1)]
        cls_each = [len(self.cla_instance[str(i)]) for i in range(1,self.num_classes+1)]
        plt.xticks(range(1, len(cls) + 1), cls, font_properties=myfont, rotation=0)
        plt.bar(cls, cls_each, color='rgb')
        plt.legend()
        plt.grid(True)
        plt.show()

    def add_aug_data(self, add_num=500, aug_save_path=None, json_file_path=None):
        '''
        1. 设定补充的数据量
        2. 低于这些类的才需要补充
        3. 补充增广函数
            1. 每张图片增广多少张
        :return:
        '''
        if aug_save_path is None or json_file_path is None:
            raise NameError

        if not osp.exists(aug_save_path):
            os.makedirs(aug_save_path)

        transformer = Transformer()

        aug_json_list = []

        for cla_name, bboxes_list in self.cla_instance.items():  # str '1': [edict()]
            cla_num = len(bboxes_list)
            if cla_num >= add_num:
                continue
            # 补充数据
            cla_add_num = add_num - cla_num  # 需要增广增加的数量
            each_num = int(math.ceil(cla_add_num / cla_num * 1.0))  # 向上取整 每张图片都进行增广
            # 每张图进行增广扩充
            for instance in tqdm(bboxes_list, desc='cla %s ;process %d: ' % (cla_name, each_num)):  # img_box edict
                # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

                img = cv2.imread(instance.abs_path)
                try:  # 检测图片是否可用
                    h, w, c = img.shape
                    img_info = edict({'img_h':h, "img_w":w})
                except:
                    print("%s is wrong " % instance.abs_path)
                import copy
                for i in range(each_num):  # 循环多次进行增广保存
                    img_ins = copy.deepcopy(self.img_instance[instance.name])
                    aug_img, img_info_tmp = transformer.aug_img(img, img_ins, img_info = img_info) # list
                    aug_name = '%s_aug%d.jpg' % (
                    osp.splitext(instance.name)[0], i) # 6598413.jpg -> 6598413_aug0.jpg, 6598413_aug1.jpg
                    aug_abs_path = osp.join(aug_save_path, aug_name)
                    for ins in img_info_tmp:
                        ins.name = aug_name
                        ins.abs_path = aug_abs_path
                        aug_json_list.append(ins)

                    cv2.imwrite(aug_abs_path, aug_img)
        #
        # # 保存aug_json 文件
        with open(json_file_path, 'w') as f:
            json.dump(aug_json_list, f, indent=4, separators=(',', ': '))

    def vis_gt(self, flag_show_raw_img=False):
        '''
        可视化gt， 按键盘d 向前， 按a 向后
        :return:
        '''
        transformer = Transformer()
        cur_node = 0
        set_img_name = list(self.img_instance.keys())  # 所有图片的名称的 list
        while True:
            img_name = set_img_name[cur_node]
            instances_list = self.img_instance[img_name]
            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1920, 1080)
            print('num gt: ', len(instances_list))
            print('img_name: %s ' % (img_name))

            ins_init = instances_list[0]
            img = cv2.imread(ins_init.abs_path)
            img_aug, _ = transformer(img, ins_init)
            for instance in instances_list:
                box = instance.bbox

                cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img_aug, '%d' % instance.classes, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 0), 1)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            cv2.imshow('img', img_aug)
            if flag_show_raw_img:
                cv2.imshow('raw', img)


class Transformer:
    def __init__(self):
        self.aug_img_seq = iaa.Sequential([
            iaa.Fliplr(0.8),
            iaa.Flipud(0.8),
            # iaa.Invert(1.0),
            iaa.Crop(percent=0.1)
        ], random_order=True)
        # pass

    def __call__(self, imgBGR, instance=None):
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

        # imgRGB = self.aug_img_seq.augment_images(imgRGB)
        imgBGR_aug = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

        # save json format
        if instance is not None:

            img_info_tmp = edict()
            img_info_tmp.bbox = instance.bbox
            img_info_tmp.defect_name = instance.defect_name
            img_info_tmp.name = instance.name
            return imgBGR_aug, img_info_tmp
        else:
            return imgBGR_aug, None

    def aug_img(self, imgBGR, instance=None, img_info = None):
        bbs = self._mk_bbs(instance, img_info)
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        imgRGB_aug, bbs_aug = self.aug_img_seq(image = imgRGB, bounding_boxes = bbs)
        bbs_aug = bbs_aug.clip_out_of_image()
        imgBGR_aug = cv2.cvtColor(imgRGB_aug, cv2.COLOR_RGB2BGR)

        # for debug to show
        # imgRGB_aug_with_box = bbs_aug.draw_on_image(imgRGB_aug,size = 2)
        # imgRGB_aug_with_box = cv2.cvtColor(imgRGB_aug_with_box, cv2.COLOR_RGB2BGR)
        # imgRGB_aug_with_box = cv2.resize(imgRGB_aug_with_box,(1333,800))
        # imgRGB_with_box = bbs.draw_on_image(imgRGB, size=2)
        # imgRGB_with_box = cv2.resize(imgRGB_with_box,(1333,800))
        # imgRGB_with_box = cv2.cvtColor(imgRGB_with_box, cv2.COLOR_RGB2BGR)
        # cv2.imshow('aug',imgRGB_aug_with_box)
        # cv2.imshow('raw',imgRGB_with_box)
        # cv2.waitKey(0)

        # save json format
        if bbs_aug is not None:
            instance_aug = instance
            for i in range(len(bbs_aug.bounding_boxes)):
                box = []
                box.append(bbs_aug.bounding_boxes[i].x1)
                box.append(bbs_aug.bounding_boxes[i].y1)
                box.append(bbs_aug.bounding_boxes[i].x2)
                box.append(bbs_aug.bounding_boxes[i].y2)
                instance_aug[i].bbox = box
                instance_aug[i].w = box[2] - box[0]
                instance_aug[i].h = box[3] - box[1]
                return imgBGR_aug, instance_aug
        else:
            return imgBGR_aug, None

    def _mk_bbs(self, instance, img_info):
        BBox = [] #[ Bounding_box, Bounding_box,]
        w = img_info.img_w
        h = img_info.img_h
        for ins in instance:
            box = ins.bbox
            BBox.append(BoundingBox(x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3]))

        return BoundingBoxesOnImage(BBox,shape = (h,w))

def compute_wh(box):
    x1, y1, x2, y2 = box
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = max(x2, 0)
    y2 = max(y2, 0)
    w = x2 - x1
    h = y2 - y1
    return w, h
