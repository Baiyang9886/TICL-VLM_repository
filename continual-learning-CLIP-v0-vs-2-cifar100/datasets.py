import PIL.Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import os.path
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from PIL import Image
from random import shuffle



class Cifar100Dataset(Dataset):
    def __init__(self, transforms_=None, data=None):
        self.transform = transforms_
        self.sample_list = data

    def __getitem__(self, index):
        sample = self.sample_list[index % len(self.sample_list)]
        label = sample['label']
        img_sam = sample['img']
        img = PIL.Image.fromarray(img_sam)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.sample_list)




task_list = ['aquatic animals', 'flowers and trees', 'food containers', 'fruit and vegetables', 'household electrical devices and furniture',
         'terrestrial animal', 'outdoor scenes', 'people', 'vehicles']

class_list = [['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
       ['orchid', 'poppy', 'rose', 'sunflower', 'tulip', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
       ['bottle', 'bowl', 'can', 'cup', 'plate'],
       ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
       ['clock', 'keyboard', 'lamp', 'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe'],
       ['bear', 'leopard', 'lion', 'tiger', 'wolf', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 'crab', 'lobster', 'snail', 'spider', 'worm', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
       ['bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea'],
       ['baby', 'boy', 'girl', 'man', 'woman'],
       ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

def get_train_loader(root):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root=root, train=True, download=False,
                                                      transform=transform_train)

    return cifar100_training

def get_val_loader(root):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root=root, train=False, download=False,
                                                  transform=transform_test)

    return cifar100_test

def get_data_dict(train_data, classes):
    train_data_dict = {}
    data = train_data.data
    label = train_data.targets
    for i in range(len(label)):
        lab = label[i]
        for cls in classes:
            if lab == classes[cls]:
                for t in range(9):
                    if cls in class_list[t]:
                        sample = {}
                        task = task_list[t]
                        c_l = class_list[t].index(cls)
                        sample['label'] = c_l
                        sample['img'] = data[i]
                        if task not in train_data_dict:
                            train_data_dict[task] = [sample]
                        else:
                            train_data_dict[task].append(sample)
    return train_data_dict

def get_alldata():
    root = "/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/l2p-datasets"
    train_data = get_train_loader(root)
    classes = train_data.class_to_idx
    test_data = get_val_loader(root)

    train_data_dict = get_data_dict(train_data, classes)
    test_data_dict = get_data_dict(test_data, classes)

    return train_data_dict, test_data_dict


def get_dataloader(data_dict, task, preprocess, batch_size, shuffle):
    task_name = task_list[task]
    data = data_dict[task_name]
    dataloader = DataLoader(
        Cifar100Dataset(transforms_=preprocess, data=data),
        batch_size=batch_size, shuffle=shuffle, num_workers=8)

    return dataloader



def get_data(train_data_dict, test_data_dict, task, preprocess, batch_size):
    train_data = get_dataloader(train_data_dict, task, preprocess, batch_size, True)
    test_data = []
    for i in range(task+1):
        data = get_dataloader(test_data_dict, i, preprocess, batch_size, False)
        test_data.append(data)

    return train_data, test_data