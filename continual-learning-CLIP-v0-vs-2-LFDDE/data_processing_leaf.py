import random

import cv2
import os


root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/leaf-dataset/used-data/'
dataroot = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/leaf-dataset/data/'
saveroot = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/leaf-dataset/used-data/images/'

train_write = open(root + "train_img_list.txt", "w")
test_write = open(root + "test_img_list.txt", "w")
cls_write = open(root + "label.txt", "w")
folder_list = os.listdir(dataroot)

count = 0
for folder in folder_list:
    path = dataroot + folder
    imglist = os.listdir(path)
    cls = folder
    cls_write.write(cls + '\n')
    for imgname in imglist:
        img_idx = imgname.split('(')[-1].split(')')[0]
        imgpath = path + '/' + imgname
        print(imgpath)
        img = cv2.imread(imgpath)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        img_name = cls + "-img-" + str(img_idx) + '.jpg'
        savepath = saveroot + img_name
        if random.randint(0, 100) < 80:
            train_write.write(img_name + '\n')
        else:
            test_write.write(img_name + '\n')
        print(savepath)
        cv2.imwrite(savepath, img)
        count = count + 1
        print(count)


