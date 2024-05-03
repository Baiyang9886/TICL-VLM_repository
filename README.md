# Task Incremental Continual Learning Based on Language Prompt Feature Selection and Pre-trained Visual-language Modle

Visual-language models (VLMs) bridge the gap between the visual and language modalities, allowing humans to pass some prior experience to the model in the form of language, thereby guiding the model to complete visual tasks more accurately and efficiently. VLMs provide a new way for the field of task incremental continuous learning. They enable us to guide the model to focus on target features through textual description of task prompts. Therefore, in this study, we propose a task incremental continual learning method based on the VLM (TICL-VLM). This method allows the model to accumulate knowledge and become stronger and stronger by learning continuously, like humans. Moreover, the computational overhead of this method is not very high, making it with great potential for practical applications. 

# Environment
Ubuntu 20.04
Python 3.9
Pytorch 2.0.1
numpy 1.25.2
torchvision 0.15.2
time
datetime
math
copy
cv2
clip

# Datasets
## 1.Cifar-100 dataset
CIFAR-100 dataset is one of the most commonly used public dataset in the field of continual learning. This dataset contains 100 classes of image samples, and each class includes 600 images with the size of 32 × 32, of which 500 are used as the training set and 100 are used as the testing set. This dataset covers a wide range of different domains, such as animals, plants, and vehicles.
Download link:https://www.cs.toronto.edu/~kriz/cifar.html

## 2. The LFDDE dataset
The LFDDE dataset is constructed by five public image classification datasets from different domains, including: Leaf disease identification dataset32, Food-101 dataset33, Tsinghua dog dataset34, SAM-DD dataset35, and Oulu-CASIA dataset36. The original Tsinghua dog dataset contains the samples of 130 dog breeds. Due to the varying sample sizes across different classes, we selected the data of 100 dog breeds with similar sample sizes for the LFDDE dataset to ensure that the data distribution among different categories is balanced. Furthermore, the SAM-DD dataset includes samples captured from both frontal and lateral perspectives. In the LFDDE dataset, we only selected the frontal-view samples from this dataset. 
Download link: https://pan.baidu.com/s/1ob7gpDtTdZG2wC-aCWJZlA?pwd=btdl

# Getting Started
1.Configure the codes execution environment.

2.Download the corresponding datasets.

3.Execution of the codes.
Run the codes on the LFDDE dataset:
'''
python continual-learning-CLIP-v0-vs-2-LFDDE.py
'''

Run the codes on the cifar-100 dataset:
'''
python continual-learning-CLIP-v0-vs-2-cifar100.py
'''

