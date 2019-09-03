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


def load_file(path,cat):
    anno = {}
    train_anno = {}
    val_anno = {}
    labels = {}
    # num_each_class = {}
    area = []

    # label2img = {}
    for i in range(len(path)):

        f = json.load(open(path[i], 'r'))
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

            #划分训练集和验证集 4:1
            total += 1
            if total < (total_/5*4):
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


    return anno,train_anno, val_anno,labels, area


def vis_gt(anno, train_anno, val_anno,flag_make_xml, xml_root, cat):
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
                root = xml_root+flag+'/'
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
    myfont = matplotlib.font_manager.FontProperties(fname='/home/remo/Desktop/simkai_downcc/simkai.ttf')
    plt.xticks(range(len(label)), label, fontproperties=myfont, rotation=-60)
    # 柱状图
    plt.bar(label, num, color='rgb')
    plt.legend()
    plt.show()

    label_group = [0] * 20
    label = [i for i in range(1, 21)]
    for l in label_dict.keys():
        label_group[int(cat[l]) - 1] += int(label_dict[l])
    plt.xticks(range(1, len(label) + 1), label, fontproperties=myfont, rotation=0)
    # 柱状图
    plt.bar(label, label_group, color='rgb')
    plt.legend()
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
            f.write(image_path + ',' + str(box[0]).split('.')[0] + ',' + str(box[1]).split('.')[0] + ',' + str(box[2]).split('.')[0] + ',' + str(box[3]).split('.')[0] + ',' + str(label)+'\n')
    f.close()

if __name__ == "__main__":
    cat = {
    '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
    '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
    '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
    '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
    }
    flag_make_xml = True
    # json_file = "/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/Annotations/gt_result.json"
    # json_file = "/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train2_20190828/Annotations/anno_train.json"

    json_file = ["/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train2_20190828/Annotations/anno_train.json",
                 "/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190818/Annotations/anno_train.json"]

    # defect_Images = '/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/defect_Images/'
    # normal_Images = '/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/normal_Images/'
    # all_Images = '/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train2_20190828/All_Images/'

    all_Images = '/home/remo/Desktop/cloth_flaw_detection/Dataset/COCO_format/Images/'
    xml_root = "/home/remo/Desktop/cloth_flaw_detection/Dataset/VOC_format/Annotations/"
    path_txt = "/home/remo/Desktop/cloth_flaw_detection/guangdong1_round1_train1_20190809/retinanet_data/train.txt"
    anno, train_anno, val_anno, labels_num, area = load_file(json_file,cat)
    vis_gt(anno,train_anno,val_anno,flag_make_xml,xml_root,cat)
    # draw(labels_num,cat)
    # draw_area(area)
    # make_txt(anno, path_txt, defect_Images, cat)
