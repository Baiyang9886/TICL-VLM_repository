import random

import cv2
import os


root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/Tsinghua_dogs/used-data1/'
dataroot = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/Tsinghua_dogs/used-data1/data/'
saveroot = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/Tsinghua_dogs/used-data1/images/'

train_write = open(root + "train_img_list.txt", "w")
test_write = open(root + "test_img_list.txt", "w")
cls_write = open(root + "label.txt", "w")
folder_list = os.listdir(dataroot)

count = 0
for folder in folder_list:
    path = dataroot + folder
    imglist = os.listdir(path)
    fol = folder.split('//')[-1].split('\n')[0]
    cls = fol.split('-')[-1]
    cls_write.write(cls + '\n')
    for imgname in imglist:
        imgpath = path + '/' + imgname
        print(imgpath)
        img = cv2.imread(imgpath)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        img_name = cls + "-" + imgname
        savepath = saveroot + img_name
        if random.randint(0, 100) < 80:
            train_write.write(img_name + '\n')
        else:
            test_write.write(img_name + '\n')
        print(savepath)
        cv2.imwrite(savepath, img)
        count = count + 1
        print(count)


