
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
import os


def decodeLebel(labelImageIn,x,y):
    labeMat = labelImageIn[(x*3):(x*3+3),(y*3):(y*3+3),0]
    LabeledScale = (labeMat > 128).astype(int)
    sum =0
    for k in range(3):
        for l in range(3):
            sum += LabeledScale[k,l]
    return sum

def cropImage(sample_imgIn,x,y):
    return sample_imgIn[(x*48):(x*48+48),(y*48):(y*48+48),0]
                                                                                                                                                                     


dataPath = './dataset/'
savePathTrain = '/home/kisna/Documents/picSniff/dataset_tfds/train'
savePathVal = '/home/kisna/Documents/picSniff/dataset_tfds/val'


trainPath = dataPath + 'train'
valPath = dataPath + 'val'
testPath = dataPath + 'test'


trainImgList = glob(trainPath + '/images/*.jpg')
trainLabelList = glob(trainPath + '/labels/*.jpg')

valImgList = glob(valPath + '/images/*.jpg')
valLabelList = glob(valPath + '/labels/*.jpg')

numberofFilesTrain = len(trainImgList)
numberofFilesVal = len(valImgList)
print(len(trainImgList))
print(len(trainLabelList))


sample_idx = 0
sample_img = cv2.imread(trainImgList[sample_idx])
sample_label = cv2.imread(trainLabelList[sample_idx])

x_train = np.zeros(shape=(100*numberofFilesTrain,48,48),dtype=np.uint8)
y_train = np.zeros(100*numberofFilesTrain, dtype=np.uint8)

x_val = np.zeros(shape=(100*numberofFilesVal,48,48),dtype=np.uint8)
y_val = np.zeros(100*numberofFilesVal, dtype=np.uint8)

# cv2.imshow("Image",sample_img)
print(sample_label.shape)
print(sample_img.shape)

for f in range(numberofFilesTrain):
    print(trainImgList[f])
    sample_img_train = cv2.imread(trainImgList[f])
    sample_label_train = cv2.imread(trainLabelList[f])
    for i in range(10):
        for j in range(10):
            imgFilename = 'img_{}_{}_{}.jpg'.format(f,i,j)
            faceImage = cropImage(sample_img_train,i,j)
            faceLebel = decodeLebel(sample_label_train,i,j)
            filePath = savePathTrain + '/' + str(faceLebel) 
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            fullPath = savePathTrain + '/' + str(faceLebel) + '/' + imgFilename
            cv2.imwrite(fullPath, faceImage)
            print(fullPath)
     


for f in range(numberofFilesVal):
    print(valImgList[f])
    sample_img_val = cv2.imread(valImgList[f])
    sample_label_val = cv2.imread(valLabelList[f])
    for i in range(10):
        for j in range(10):
            imgFilename = 'img_{}_{}_{}.jpg'.format(f,i,j)
            faceImage = cropImage(sample_img_val,i,j)
            faceLebel = decodeLebel(sample_label_val,i,j)
            filePath = savePathVal + '/' + str(faceLebel) 
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            fullPath = savePathVal + '/' + str(faceLebel) + '/' + imgFilename
            cv2.imwrite(fullPath, faceImage)
            print(fullPath)
      




