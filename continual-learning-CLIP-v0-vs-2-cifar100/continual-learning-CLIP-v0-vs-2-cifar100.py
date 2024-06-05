'''
在 vs-2的基础上修改了数据集中样本的分类
'''

import datetime
import time, copy
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from model import FeatureSlection, Head, get_prompt
from datasets import *



torch.cuda.set_device(1)

def mse_loss(class_feature, features, labels, criterion):
    num_sampel = features.shape[0]
    mse_loss = 0
    for i in range(num_sampel):
        feature = features[i]
        lab = int(labels[i])
        center = class_feature[lab]
        center = center.type(torch.float32)
        loss = criterion(feature, center)
        mse_loss = mse_loss + loss
    mse_loss = mse_loss / num_sampel

    return mse_loss



class TICL_VLM():
    def __init__(self, model_name):
        super(TICL_VLM, self).__init__()
        self.batch_size = 128
        self.n_epochs = 20
        self.lr = 0.00001
        self.b1 = 0.9
        self.b2 = 0.999
        self.embed_dim = 32
        self.lamb = 1
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l2 = torch.nn.MSELoss().to(self.device)
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
        self.task_prompt = ["Classify objects in the images"]
        self.clip_model = model_name
        self.in_dim = 768
        if self.clip_model == 'CLIP-L14':
            self.in_dim = 1024

        self.train_dict, self.test_dict = get_alldata()
        self.get_feature = FeatureSlection(device=self.device, clip_model=self.clip_model)
        self.get_feature = self.get_feature.to(self.device)
        for n, p in self.get_feature.named_parameters():
            p.requires_grad = False
        self.heads = {}
        self.num_dim_list = [10, 10, 5, 5, 10, 5, 10, 10, 10, 10, 5, 10]
        self.task_list = ['aquatic animals', 'flowers and trees', 'food containers', 'fruit and vegetables',
                          'household electrical devices and furniture', 'insects',
                          'large terrestrial animal', 'outdoor scenes', 'medium-size animal', 'small-size animal',
                          'people', 'vehicles']

    def train(self, task):
        self.log_write = open("log_zero_shot_%s_continua_learning_task%s.txt" % (self.clip_model, str(task)), "w")
        self.log_write_train = open("log_zero_shot_%s_continua_learning_task%s_train.txt" % (self.clip_model, str(task)), "w")
        train_data, test_data_list = get_data(self.train_dict, self.test_dict, task, self.get_feature.preprocess, self.batch_size)
        num_cls = self.num_dim_list[task]
        head = Head(num_cls, self.device, self.in_dim).to(self.device)
        for p in head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        task_feature, class_feature = get_prompt(task, self.get_feature.text_encoder, self.device)

        # Optimizers
        self.optimizer = torch.optim.Adam(head.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        print("The train process of task %s, the number of class is %s" % (self.task_list[task], str(num_cls)))
        start_time = time.time()
        acc_list = []
        best_acc = []
        for t in range(task + 1):
            acc_list.append([])
            best_acc.append(0.0)
        for epoch in range(self.n_epochs):
            running_corrects = 0
            numSample = 0
            batch = 0
            head.train()
            for step, data in enumerate(train_data):
                images, labels = data
                # Model inputs
                self.optimizer.zero_grad()
                batch = batch + 1
                images = images.to(self.device)
                label = Variable(labels.type(self.LongTensor))
                numSample = numSample + label.size(0)
                with torch.no_grad():
                    features = self.get_feature(images)
                out_feature, Cls = head(task_feature[task], features)

                _, pred = torch.max(Cls.data, 1)
                running_corrects += torch.sum(pred == label.data)

                # Classification loss
                cls_loss = self.criterion_cls(Cls, label)

                # MSE loss
                cdc_loss = mse_loss(class_feature, out_feature, labels, self.criterion_l2)

                # Total loss
                loss = cls_loss + self.lamb * cdc_loss

                loss.backward(retain_graph=True)
                self.optimizer.step()

                # --------------
                #  Log Progress
                # --------------
                batches_done = epoch * len(train_data) + batch
                batches_left = self.n_epochs * len(train_data) - batches_done
                time_left = datetime.timedelta(
                    seconds=batches_left * (time.time() - start_time) / (batches_done + 1))

                print(
                    "\r[Task %d] [Epoch %d/%d] [Batch %d/%d] [loss: %f] [cls_loss: %f] [cdc_loss: %f] [train_acc: %f]ETA: %s"
                    % (
                        task,
                        epoch,
                        self.n_epochs,
                        batch,
                        len(train_data),
                        loss.item(),
                        cls_loss.item(),
                        cdc_loss.item(),
                        # dis_loss.item(),
                        100.0 * running_corrects / numSample,
                        time_left,
                    )
                )

            # If at sample interval sample and save image
            torch.save(head.state_dict(), "saved_models/head_weights_task%s.pth" % str(task))
            epoch_acc = 100.0 * running_corrects.item() / numSample
            print("{} Acc:{:.4f}%".format('train_epoch', epoch_acc))
            self.log_write_train.write(str(epoch) + "    " + str(epoch_acc) + "\n")

            if (epoch) % 1 == 0:
                best = best_acc[task]
                test_data = test_data_list[task]
                acc = self.test(head, test_data, task_feature[task])
                self.log_write.write("Epoch " + str(epoch) + "    Task" + str(task) + "    acc = " + str(acc) + "\n")
                acc_list[task].append(acc)
                if acc > best:
                    best_acc[task] = acc
                    self.heads[str(task)] = head
                    torch.save(head.state_dict(), "saved_models/head_weights_best_%s_task%s.pth" % (self.clip_model, str(task)))

                for k in range(task):
                    best = best_acc[k]
                    test_data = test_data_list[k]
                    cur_head = self.heads[str(k)]
                    cur_head.load_state_dict(torch.load("saved_models/head_weights_best_%s_task%s.pth" % (self.clip_model, str(task))))
                    acc = self.test(cur_head, test_data, task_feature[k])
                    self.log_write.write("Epoch " + str(epoch) + "    Task" + str(k) + "    acc = " + str(acc) + "\n")
                    acc_list[k].append(acc)
                    if acc > best:
                        best_acc[k] = acc

        for k in range(task + 1):
            aver_acc = np.mean(acc_list[k])
            best = best_acc[k]
            print('The average acc of task %d is: %f' % (k, aver_acc))
            print('The best acc of task %d is: %f' % (k, best))
            self.log_write.write('The average acc of task %d is: %f' % (k, aver_acc) + "\n")
            self.log_write.write('The best acc of task %d is: %f' % (k, best) + "\n")

        self.log_write.close()
        self.log_write_train.close()

    def test(self, model, test_data, task_feature):
        time_open = time.time()
        running_corrects = 0
        numSample = 0
        batch = 0
        model.eval()
        print('\n The test process......')

        # for images, labels in tqdm(DataLoader(self.test_dataloader, batch_size=100)):
        for step, data in enumerate(test_data):
            images, labels = data
            batch = batch + 1
            images = images.to(self.device)
            label = Variable(labels.type(self.LongTensor))
            numSample = numSample + label.size(0)

            with torch.no_grad():
                feature = self.get_feature(images)
                _, y_pred = model(task_feature, feature)
            _, pred = torch.max(y_pred.data, 1)
            running_corrects += torch.sum(pred == label.data)

            if batch % 20 == 0:
                print("Batch {}, Test ACC:{:.4f}%".format(
                    batch, 100.0 * running_corrects / numSample))

        epoch_acc = 100.0 * running_corrects.item() / numSample
        print("{} Acc:{:.4f}%".format('test', epoch_acc))

        time_end = time.time() - time_open
        print("程序运行时间:{}分钟......\n".format(int(time_end / 60)))

        return epoch_acc



def main():
    # model_name = ['CLIP-B16', 'CLIP-L14']
    model_name = ['CLIP-L14']
    for model in model_name:
        num_task = 12
        ticlvlm = TICL_VLM(model)
        for t in range(num_task):
            ticlvlm.train(t)

if __name__ == "__main__":
    main()