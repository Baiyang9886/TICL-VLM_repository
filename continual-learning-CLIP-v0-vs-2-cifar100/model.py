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


task_list = ['aquatic animals', 'flowers and trees', 'food containers', 'fruit and vegetables', 'household electrical devices and furniture', 'insects',
         'large terrestrial animal', 'outdoor scenes', 'medium-size animal', 'small-size animal', 'people', 'vehicles']

class_list = [['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
       ['orchid', 'poppy', 'rose', 'sunflower', 'tulip', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
       ['bottle', 'bowl', 'can', 'cup', 'plate'],
       ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
       ['clock', 'keyboard', 'lamp', 'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe'],
       ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
       ['bear', 'leopard', 'lion', 'tiger', 'wolf', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
       ['bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea'],
       ['fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
       ['crab', 'lobster', 'snail', 'spider', 'worm', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
       ['baby', 'boy', 'girl', 'man', 'woman'],
       ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]


def get_task_prompt(text_encoder, device, task_name):
    task_prompts = ['Identify the categories of ' + task_name + ' in the picture.']
    task_prompt = task_prompts
    task_text = clip.tokenize(task_prompt).to(device)
    _, task_feature_s = text_encoder(task_text)

    return task_feature_s


def get_class_prompt(text_encoder, device,  class_name):
    class_prompts = []
    for cls in class_name:
        words = cls.split('_')
        p = ''
        for w in words:
            p = p + ' ' + w
        cls_prompt = 'A picture of the' + p
        class_prompts.append(cls_prompt)

    class_prompt = class_prompts
    class_text = clip.tokenize(class_prompt).to(device)

    _, class_feature = text_encoder(class_text)

    return class_feature


def get_prompt(task, text_encoder, device):
    class_name = class_list[task]
    class_feature = get_class_prompt(text_encoder, device, class_name)
    task_feature_list = []
    for i in range(task + 1):
        task_name = task_list[i]
        task_feature = get_task_prompt(text_encoder, device, task_name)
        task_feature_list.append(task_feature)

    return task_feature_list, class_feature