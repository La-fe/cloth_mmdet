###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012                   #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012                   #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *
import os

gt_path = "/home/remo/Desktop/cloth_flaw_detection/Results/re_map/groundtruth"
dt_path = '/home/remo/Desktop/cloth_flaw_detection/Results/re_map/result_10_val/detections'
dir = dt_path.split('/')[:-1]
dir_path = '/'.join(dir)
show = False
iou_th = ['0.1','0.3','0.5']
sub_dir = []
for iou in iou_th:
    sub_dir_ = str(iou.split('.')[1])
    sub_dir.append(sub_dir_)
    if not os.path.exists(dir_path+'/'+sub_dir_):
        os.mkdir(dir_path+'/'+sub_dir_)

def getBoundingBoxes():
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = os.path.dirname(os.path.abspath(__file__))
    # folderGT = os.path.join(currentPath, 'groundtruths')
    folderGT = os.path.join(gt_path)
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])  # confidence
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (200, 200),
                BBType.GroundTruth,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    # Read detections
    # folderDet = os.path.join(currentPath, 'detections')
    folderDet = os.path.join(dt_path)
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt", "")
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (200, 200),
                BBType.Detected,
                confidence,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes


def createImages(dictGroundTruth, dictDetected):
    """Create representative images with bounding boxes."""
    import numpy as np
    import cv2
    # Define image size
    width = 200
    height = 200
    # Loop through the dictionary with ground truth detections
    for key in dictGroundTruth:
        image = np.zeros((height, width, 3), np.uint8)
        gt_boundingboxes = dictGroundTruth[key]
        image = gt_boundingboxes.drawAllBoundingBoxes(image)
        detection_boundingboxes = dictDetected[key]
        image = detection_boundingboxes.drawAllBoundingBoxes(image)
        # Show detection and its GT
        cv2.imshow(key, image)
        cv2.waitKey()


# Read txt files containing bounding boxes (ground truth and detections)
boundingboxes = getBoundingBoxes()
# Uncomment the line below to generate images based on the bounding boxes
# createImages(dictGroundTruth, dictDetected)
# Create an evaluator object in order to obtain the metrics
evaluator = Evaluator()
##############################################################
# VOC PASCAL Metrics
##############################################################
total_precision = 0
for i in range(len(iou_th)):
    iou = float(iou_th[i])
    sub_dir_ = sub_dir[i]
    save_path = dir_path+'/'+sub_dir_
    total_precision_ = 0
    ap_each = {}
    # Plot Precision x Recall curve
    evaluator.PlotPrecisionRecallCurve(
        boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iou,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=True,
        savePath=save_path,
        showGraphic = show)  # Plot the interpolated precision curve

    # Get metrics with PASCAL VOC metrics
    metricsPerClass = evaluator.GetPascalVOCMetrics(
        boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iou,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
    # print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics

    with open(dir_path+'/re_map_%s.txt'%str(iou_th[i]),'w') as f:
        for mc in metricsPerClass:
            # Get metric values per each class
            c = mc['class']
            precision = mc['precision']
            recall = mc['recall']
            average_precision = mc['AP']
            ipre = mc['interpolated precision']
            irec = mc['interpolated recall']
            f.write('Class: '+c+'\n')
            f.write('Precision: '+str(precision.tolist())+'\n')
            f.write('Recall: '+str(recall.tolist())+'\n')
            f.write('Average_Precision: '+str(average_precision)+'\n'+'\n')
            total_precision_ += average_precision
            ap_each[c] = average_precision
            # Print AP per class
            # print('%s: %f' % (c, average_precision))
        print('%s: %f' % ('mAP '+str(i), total_precision_/20))
        for cls in ap_each.keys():
            f.write(str(cls)+': '+str(ap_each[cls])+'\n')
        f.write('mAP: '+str(total_precision_/20))
        total_precision += total_precision_/20
print('mAP: '+str(total_precision/len(iou_th)))