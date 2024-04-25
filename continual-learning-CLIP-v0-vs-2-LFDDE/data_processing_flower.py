import cv2
import os



root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/flower_data/'
saveroot = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/flower_data/arrange_data/'

mode_list = ['train', 'val']
cls_list = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

for mode in mode_list:
    for c in range(len(cls_list)):
        cls = cls_list[c]
        path = root + mode + '/' + cls + '/'
        imglist = os.listdir(path)
        for i in range(len(imglist)):
            print(i)
            imgpath = path + imglist[i]
            print(imgpath)
            img = cv2.imread(imgpath)
            cv2.imshow('img', img)
            cv2.waitKey(1)
            if mode == 'train':
                savepath = saveroot + 'train-data/img_cls_' + str(c) + '_n' + str(i) + '.jpg'
            else:
                savepath = saveroot + 'test-data/img_cls_' + str(c) + '_n' + str(i) + '.jpg'
            print(savepath)
            cv2.imwrite(savepath, img)

