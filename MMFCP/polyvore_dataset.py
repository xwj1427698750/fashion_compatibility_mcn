import csv
import gzip
import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class CategoryDataset(Dataset):
    """Dataset for polyvore with 4 categories(upper, bottom, shoe, bag),

    Args:
        root_dir: Directory stores source images
        data_file: A json file stores each outfit index and description
        data_dir: Directory stores data_file and mean_images
        transform: Operations to transform original images to fix size
        change_id: 生成单品的类型， None表示随机选择， 0-3表示选择对应下标类型的单品 ['upper', 'bottom', 'bag', 'shoe']
    """
    def __init__(self, root_dir="../data/images2/",  # images2对应的原始文件目录在/home/ices/xwj/graduation project/dataset/img/img_types/total/
                 data_file='train.json',
                 data_dir="../data", transform=None, change_id=None):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.random_choice = True  # 随机选择生成单品类型
        if change_id is not None:
            self.random_choice = False  # 选择指定的类型
        self.change_id = change_id
        self.data = json.load(open(os.path.join(data_dir, data_file)))
        self.data = [(k, v) for k, v in self.data.items()]  # k:套装id,v:{套装对象}
        self.vocabulary, self.word_to_idx = [], {}
        self.word_to_idx['UNK'] = len(self.word_to_idx)
        self.vocabulary.append('UNK')
        with open(os.path.join(self.data_dir, 'vocab.dat')) as f:
            for line in f:
                name = line.strip().split()[0]
                if name not in self.word_to_idx:
                    self.word_to_idx[name] = len(self.word_to_idx)
                    self.vocabulary.append(name)

    def __getitem__(self, index):
        """
        return:
        input_images: 一套正样本数据以及指定类型的负样本，一共 4 + 1 件单品，前4件为正，后1件为负
        names: 正样本单品的文本描述
        set_id: 正样本套装的id
        labels: 单品的标签
        change_id: 待生成单品的类型
        """
        set_id, parts = self.data[index]
        category = ['upper', 'bottom', 'bag', 'shoe']  # list(parts.keys())  ['upper', 'bottom', 'bag', 'shoe']
        if self.random_choice:
            self.change_id = random.randint(0, 3)  # 在候选套装中，随机选择一件单品的类型，生成的单品就是这一类型

        imgs = []
        labels = []
        names = torch.zeros(len(self.vocabulary))
        # 获取正样本
        for part in ['upper', 'bottom', 'bag', 'shoe']:
            img_path = os.path.join(self.root_dir, str(parts[part]['index']) + '.jpg')
            if category[self.change_id] == part:
                names = torch.Tensor(self.str_to_idx(parts[part]['name']))
            labels.append('{}_{}_{}'.format(set_id, part, parts[part]['index']))
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        # 将目标单品的移动到最后一个位置
        imgs[self.change_id], imgs[3] = imgs[3], imgs[self.change_id]
        labels[self.change_id], labels[3] = labels[3], labels[self.change_id]
        # 负样本的生成方式： 根据给定的套装生成指定类型的负样本
        choice = random.choice(self.data)
        while choice[0] == set_id:
            choice = random.choice(self.data)
        img_path = os.path.join(self.root_dir, str(choice[1][category[self.change_id]]['index'])+'.jpg')

        labels.append('{}_{}_{}'.format(choice[0], category[self.change_id], choice[1][category[self.change_id]]['index']))
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        imgs.append(img)
        input_images = torch.stack(imgs)  # 沿着第0维度拼接图片 [N,C,H,W], N = len(imgs), [5, 3, 128, 128]

        return input_images, names, set_id, labels, self.change_id

    def __len__(self):
        return len(self.data)

    """ 根据文本描述的长度返回对应长度的向量 """
    def str_to_idx(self, name):
        name_len = len(self.vocabulary)  # 生成一个0-1向量  1244+1 = 1245维，额外的1是代表低频单词的字符'UNK'
        desc = [0] * name_len
        for w in name.strip().split():
            if w in self.word_to_idx:
                desc[self.word_to_idx[w]] = 1
        return desc

    def get_fitb_quesiton(self, index, option_len=4, change_id=None):
        """Generate fill in th blank questions.
        Args:
            index: 表示读取的测试集中的套装下标
            option_len: 表示fitb任务选项的个数
            change_id: 表示生成单品的类型，None表示随机选择一个单品类型，否则按照chang_id对应的数字，chang_id的范围是[0,3]，
                       对应关系{0:'upper', 1:'bottom', 2:'bag', 3:'shoe'}
        Return:
            images: 4 parts of a outfit
            labels: store if this item is empty
            question_part: which part to be changed
            options: 3 other item with the same category,
            expect original composition get highest score
        """
        set_id, parts = self.data[index]  # set_id：套装id, parts:套装的组成部分的描述
        category = ['upper', 'bottom', 'bag', 'shoe']
        if change_id is None:
            change_id = random.randint(0, 3)  # 在候选套装中，随机选择一件单品的类型
        else:
            assert 0 <= change_id <= 3  # 确保change_id的范围是0-3
        question_id = "{}_{}".format(set_id, parts[category[change_id]]['index'])  # eg:  parts[question_part]['index'] : be9627e0c2e43ee96017e288b03eed96(图片的编号)
        imgs = []
        names = torch.zeros(len(self.vocabulary))
        labels = []
        for part in ['upper', 'bottom', 'bag', 'shoe']:
            img_path = os.path.join(self.root_dir, str(parts[part]['index'])+'.jpg')
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
            if category[change_id] == part:
                names = torch.Tensor(self.str_to_idx(parts[part]['name']))
            labels.append('{}_{}_{}'.format(set_id, part, parts[part]['index']))
        # 将目标单品移动到最后一个位置
        imgs[change_id], imgs[3] = imgs[3], imgs[change_id]
        labels[change_id], labels[3] = labels[3], labels[change_id]
        items = torch.stack(imgs)  # 正确的搭配
        # 负样本
        option_ids = [set_id]
        options = []
        option_labels = []
        while len(option_ids) < option_len:
            option = random.choice(self.data)
            if (option[0] in option_ids) or (category[change_id] not in option[1]):
                continue
            else:
                option_ids.append(option[0])
                img_path = os.path.join(self.root_dir, str(option[1][category[change_id]]['index'])+'.jpg')
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                options.append(img)
                option_labels.append("{}_{}_{}".format(option[0], category[change_id], option[1][category[change_id]]['index']))

        # Return 4 options for question, 3 incorrect options
        return items, names, labels, category[change_id], question_id, options, option_labels

def collate_fn(data):
    """Need custom a collate_fn"""
    images, names, set_id, labels, change_ids = zip(*data)
    names = torch.stack(names)
    images = torch.stack(images)
    return (images, names, set_id, labels, change_ids)

# Test the loader
if __name__ == "__main__":
    img_size = 224
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor()
    ])
    d = CategoryDataset(transform=transform)
    loader = DataLoader(d, 4, shuffle=True, num_workers=4, collate_fn=collate_fn)
