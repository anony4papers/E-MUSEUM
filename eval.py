'''Code for evaluating the performance of the attack and protecting'''
import os
import csv
from typing import Any
from torchvision import models
from tqdm import tqdm

from utils.img_utils import load_image
from utils.torch_utils import numpy_to_variable

class evaluate():
    def __init__(self, gt_img_path, test_img_path, net_list):
        self._gt_test_imgs = self._get_img_list(gt_img_path, test_img_path)
        self._store_name = test_img_path.split('/')[-1]
        self._net_list = net_list
    
    def _get_net(self, model_name):
        return getattr(models, model_name)(pretrained = True).cuda().eval()
    
    def _get_img_list(self, gt_path, test_path):
        img_names = os.listdir(gt_path)
        test_suffix = '.JPEG' if gt_path == test_path else '.png'
        rs = []
        for img_name in img_names:
            gt_test_pair = [os.path.join(gt_path,img_name), os.path.join(test_path, img_name.replace('.JPEG', test_suffix))]
            rs.append(gt_test_pair)
        return rs
    
    def __call__(self, out_path):
        net_list = []
        acc_list = []
        for net_name in self._net_list:
            net_list.append(net_name)
            net = self._get_net(net_name)
            total = 0
            correct = 0
            for img_path_pair in tqdm(self._gt_test_imgs):
                total += 1
                ori = load_image(shape = (224,224),data_format="channels_first", abs_path=True, fpath=img_path_pair[0])
                ori_tensor = numpy_to_variable(ori)
                logits = net(ori_tensor)
                pred = logits.argmax()
                test_img = load_image(shape = (224,224),data_format="channels_first", abs_path=True, fpath=img_path_pair[1])
                test_img_tensor = numpy_to_variable(test_img)
                test_logits = net(test_img_tensor)
                test_pred = test_logits.argmax()
                if pred == test_pred:
                    correct += 1
            acc = 100*correct/total
            acc_list.append(acc)
        with open(os.path.join(out_path,self._store_name+'.csv'),'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(net_list)
            csvwriter.writerow(acc_list)

if __name__ == '__main__':
    ori_path = '/media/weiheng/Elements/Data/ImageNet/correct_all_1000'
    generated_img = '/media/weiheng/Elements/Data/ImageNet/generated/exp_feature_attack_protect_multi_layer_4_100_non_random'
    net_list = ['vgg11','vgg16','resnet18','resnet34','resnet50']
    evaluator =evaluate(ori_path, generated_img, net_list)
    evaluator('./result')