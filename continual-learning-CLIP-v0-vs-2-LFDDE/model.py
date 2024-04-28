import torch.nn as nn
import torch
from clip import clip
from cross_attention import CrossAttention



def create_clip_model(model_name, device):
    if model_name == "CLIP-B32":
        model, preprocess = clip.load('ViT-B/32', device)

    elif model_name == "CLIP-B16":
        model, preprocess = clip.load('ViT-B/16', device)

    elif model_name == "CLIP-L14":
        model, preprocess = clip.load('ViT-L/14', device)

    elif model_name == "CLIP-L14-336":
        model, preprocess = clip.load('ViT-L/14@336px', device)

    elif model_name == "RN50":
        model, preprocess = clip.load('RN50', device)

    elif model_name == "RN101":
        model, preprocess = clip.load('RN101', device)

    elif model_name == "RN50-64":
        model, preprocess = clip.load('RN50x64', device)

    for n, p in model.named_parameters():
        p.requires_grad = False

    image_encoder = model.encode_image
    text_encoder = model.encode_text

    return model, image_encoder, text_encoder, preprocess



class FeatureSlection(nn.Module):
    def __init__(self, device=None, clip_model=None):
        super(FeatureSlection, self).__init__()
        self.clip, self.image_encoder, self.text_encoder, self.preprocess = create_clip_model(clip_model, device)

    def forward(self, images):
        feature = self.image_encoder(images)

        return feature


class Head(nn.Module):
    def __init__(self, c_dim=100, device=None, in_dim=768):
        super(Head, self).__init__()
        self.cross_attention = CrossAttention(device, in_dim)
        self.out_dim = 512
        if in_dim == 1024:
            self.out_dim = 768
        self.head = nn.Sequential(
            nn.Linear(self.out_dim, c_dim)
        )

    def forward(self, task_feature, img_feature):
        out_feature = self.cross_attention(img_feature, task_feature)
        out_feature = out_feature.type(torch.float32)
        feature = out_feature.view(out_feature.size(0), -1)
        cls = self.head(feature)

        return out_feature, cls

def get_task_prompt(text_encoder, device):
    task_names = {'leaf':['Identify the categories and disease of leaves in the picture.'],
                   'dog':['Identify the breed of dogs in the picture.'],
                   'food':['Identify the categories of food in the picture.'],
                   'flower':['Identify the categories of flowers in the picture.'],
                   'distraction':['Identify the distraction behaviours categories of driver in the picture'],
                   'expression':['Identify the facial expression categories of face in the picture']}

    task_promts = {}
    for task in task_names:
        task_prompt = task_names[task]
        task_text = clip.tokenize(task_prompt).to(device)
        _, task_feature = text_encoder(task_text)
        task_promts[task] = task_feature

    return task_promts

def get_leaf_prompt(text_encoder, device):
    class_prompts = []
    root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/leaf-dataset/used-data/'
    read_f = open(root + 'label.txt', 'r')
    line = read_f.readline()
    while line:
        cls = line.split('___')[0]
        disease =line.split('___')[-1]
        if disease == 'healthy\n':
            cls_prompt = 'A picture of a healthy ' + cls + ' leaf\n'
        else:
            words = disease.split('_')
            p = ''
            for w in words:
                p = p + ' ' + w
            cls_prompt = 'A picture of a ' + cls + ' leaf with' + p
        class_prompts.append(cls_prompt)
        line = read_f.readline()

    class_prompt = class_prompts
    class_text = clip.tokenize(class_prompt).to(device)

    _, class_feature = text_encoder(class_text)

    return class_feature

def get_dog_prompt(text_encoder, device):
    class_prompts = []
    root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/Tsinghua_dogs/used-data1/'
    read_f = open(root + 'label.txt', 'r')
    line = read_f.readline()
    while line:
        words = line.split('\n')[0].split('_')
        p = ''
        for w in words:
            p = p + ' ' + w
        cls_prompt = 'A picture of the' + p
        class_prompts.append(cls_prompt)
        line = read_f.readline()

    class_prompt = class_prompts
    class_text = clip.tokenize(class_prompt).to(device)
    _, class_feature = text_encoder(class_text)

    return class_feature

def get_food_prompt(text_encoder, device):
    class_prompts = []
    root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/Food-101/food-101/'
    read_f = open(root + 'meta/' + 'classes.txt', 'r')
    line = read_f.readline()
    while line:
        words = line.split('\n')[0].split('_')
        p = ''
        for w in words:
            p = p + ' ' + w
        cls_prompt = 'A picture of the' + p
        class_prompts.append(cls_prompt)
        line = read_f.readline()

    class_prompt = class_prompts
    class_text = clip.tokenize(class_prompt).to(device)
    _, class_feature = text_encoder(class_text)

    return class_feature

def get_flower_prompt(text_encoder, device):
    class_prompts = ['A picture of daisy', 'A picture of dandelion', 'A picture of roses', 'A picture of sunflowers',
                     'A picture of tulips']
    class_prompt = class_prompts
    class_text = clip.tokenize(class_prompt).to(device)
    _, class_feature = text_encoder(class_text)

    return class_feature

def get_distraction_prompt(text_encoder, device):
    root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/Driver-abnormal-behaviour-recognition/SAM-DD/used-data/'
    class_prompts = []
    read_f = open(root + 'label.txt', 'r')
    line = read_f.readline()
    while line:
        words = line.split('\n')[0]
        cls_prompt = 'A driver is ' + words + ' in the picture'
        class_prompts.append(cls_prompt)
        line = read_f.readline()
    class_prompt = class_prompts
    class_text = clip.tokenize(class_prompt).to(device)
    _, class_feature = text_encoder(class_text)

    return class_feature

def get_expression_prompt(text_encoder, device):
    class_prompts = ['A picture of face with angry expression', 'A picture of face with disgust expression',
                      'A picture of face with fear expression',
                      'A picture of face with happy expression', 'A picture of face with sad expression',
                      'A picture of face with surprise expression']
    class_prompt = class_prompts
    class_text = clip.tokenize(class_prompt).to(device)
    _, class_feature = text_encoder(class_text)

    return class_feature

def get_digit_prompt(text_encoder, device):
    class_prompts = ['A picture of handwritten digit zero', 'A picture of handwritten digit one',
                      'A picture of handwritten digit two',
                      'A picture of handwritten digit three', 'A picture of handwritten digit four',
                      'A picture of handwritten digit five',
                      'A picture of handwritten digit six', 'A picture of handwritten digit seven',
                      'A picture of handwritten digit eight',
                      'A picture of handwritten digit nine']
    class_prompt = class_prompts
    class_text = clip.tokenize(class_prompt).to(device)
    _, class_feature = text_encoder(class_text)

    return class_feature

def get_prompt(task, text_encoder, device):
    if task == 'food':
        class_feature = get_food_prompt(text_encoder, device)
    elif task == 'dog':
        class_feature = get_dog_prompt(text_encoder, device)
    elif task == 'leaf':
        class_feature = get_leaf_prompt(text_encoder, device)
    elif task == 'flower':
        class_feature = get_flower_prompt(text_encoder, device)
    elif task == 'distraction':
        class_feature = get_distraction_prompt(text_encoder, device)
    elif task == 'expression':
        class_feature = get_expression_prompt(text_encoder, device)
    elif task == 'digit':
        class_feature = get_digit_prompt(text_encoder, device)

    return class_feature