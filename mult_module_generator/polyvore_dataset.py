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
    each suit has at least 3 items, the missing items will use a mean image.

    Args:
        root_dir: Directory stores source images
        data_file: A json file stores each outfit index and description
        data_dir: Directory stores data_file and mean_images
        transform: Operations to transform original images to fix size
        use_mean_img: Whether to use mean images to fill the blank part
        neg_samples: Whether generate negative sampled outfits
    """
    def __init__(self, root_dir="../data/images2/",  # images2对应的原始文件目录在/home/ices/xwj/graduation project/dataset/img/img_types/total/
                 data_file='train.json',
                 data_dir="../data", transform=None, use_mean_img=True, neg_samples=True, target_type="upper"):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.use_mean_img = use_mean_img
        self.data = json.load(open(os.path.join(data_dir, data_file)))
        self.data = [(k, v) for k, v in self.data.items()]  # k:套装id,v:{套装对象}
        self.neg_samples = neg_samples  # if True, will randomly generate negative outfit samples

        self.target_type = target_type
        self.fashion_item_type = {'upper': 0, 'bottom': 1, 'bag': 2, 'shoe': 3}

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
        """返回一套正样本数据以及指定类型的负样本，一共 4 + 1 件单品，前4件为正，后1件为负 """
        set_id, parts = self.data[index]
        # to_change = False
        # if random.randint(0, 1) and self.neg_samples:  # random.randint(0, 1) 随机生成0和1
        #     to_change = True  # random choose negative items
        imgs = []
        labels = []
        names = []
        # 获取正样本
        for part in ['upper', 'bottom', 'bag', 'shoe']:
            img_path = os.path.join(self.root_dir, str(parts[part]['index']) + '.jpg')
            names.append(torch.LongTensor(self.str_to_idx(parts[part]['name'])))
            labels.append('{}_{}_{}'.format(set_id, part, parts[part]['index']))
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)

        # 负样本的生成方式： 根据给定的套装生成指定类型的负样本
        choice = random.choice(self.data)
        while choice[0] == set_id:
            choice = random.choice(self.data)
        img_path = os.path.join(self.root_dir, str(choice[1][self.target_type]['index'])+'.jpg')
        names.append(torch.LongTensor(self.str_to_idx(choice[1][self.target_type]['name'])))
        labels.append('{}_{}_{}'.format(choice[0], self.target_type, choice[1][self.target_type]['index']))
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        imgs.append(img)
        input_images = torch.stack(imgs)  # 沿着第0维度拼接图片 [N,C,H,W], N = len(imgs), [5, 3, 224, 224]

        return input_images, names, set_id, labels

    def __len__(self):
        return len(self.data)

    """ 根据文本描述的长度返回对应长度的向量 """
    def str_to_idx(self, name):
        return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['UNK']
            for w in name.split()]

    def get_fitb_quesiton(self, index):
        """Generate fill in th blank questions.
        Return:
            images: 4 parts of a outfit
            labels: store if this item is empty
            question_part: which part to be changed
            options: 3 other item with the same category,
            expect original composition get highest score
        """
        set_id, parts = self.data[index]  # set_id：套装id, parts:套装的组成部分的描述
        question_part = self.target_type  # 选择的都是预先设定好的类别
        question_id = "{}_{}".format(set_id, parts[question_part]['index'])  # eg:  parts[question_part]['index'] : be9627e0c2e43ee96017e288b03eed96(图片的编号)
        imgs = []
        labels = []
        for part in ['upper', 'bottom', 'bag', 'shoe']:
            if part in parts.keys():
                img_path = os.path.join(self.root_dir, str(parts[part]['index'])+'.jpg')
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
                labels.append('{}_{}_{}'.format(set_id, part, parts[part]['index']))
        items = torch.stack(imgs)  # 正确的搭配

        option_ids = [set_id]
        options = []
        option_labels = []
        while len(option_ids) < 4:
            option = random.choice(self.data)
            if (option[0] in option_ids) or (question_part not in option[1]):
                continue
            else:
                option_ids.append(option[0])
                img_path = os.path.join(self.root_dir, str(option[1][question_part]['index'])+'.jpg')
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                options.append(img)
                option_labels.append("{}_{}_{}".format(option[0], question_part, option[1][question_part]['index']))

        # Return 4 options for question, 3 incorrect options
        return items, labels, question_part, question_id, options, option_labels

def collate_fn(data):
    """Need custom a collate_fn"""
    # data.sort(key=lambda x: x[0].shape[0], reverse=True)
    images, names, set_id, labels = zip(*data)
    lengths = [i.shape[0] for i in images]
    names = sum(names, [])
    # is_compat = torch.LongTensor(is_compat)
    images = torch.stack(images)
    return (lengths, images, names, set_id, labels)

def lstm_collate_fn(data):
    """Need custom a collate_fn for LSTM DataLoader
    Batch images will be transformed to a long sequence.
    """
    # data.sort(key=lambda x:x[0].shape[0], reverse=True)
    images, names, set_id, labels = zip(*data)
    lengths = [i.shape[0] for i in images]
    names = sum(names, [])
    images = torch.cat(images)
    return (lengths, images, names, set_id, labels)


class TripletDataset(Dataset):
    """Dataset will generate triplet to train conditional similarity network. Each
     element in dataset should be anchor image, positive image, negative image and condition.

     Args:
         root_dir: Image directory
         data_file: A file record all outfit id and items
         data_dir: Directory which save mean image and data_file
         transform:
         is_train: Train phase will genrate triplet and condition, Evaluate phase will generate
             pair, condition and target.
     """
    def __init__(self,
             root_dir="/export/home/wangx/datasets/polyvore-dataset/images/",
             data_file='train_no_dup_with_category_3more_name.json',
             data_dir="../data", transform=None, is_train=True):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.data = json.load(open(os.path.join(data_dir, data_file)))
        self.data = [(k, v) for k, v in self.data.items()]
        self.is_train = is_train
        self.conditions = {
            'upper_bottom': 0,
            'bottom_upper': 0,
            'upper_shoe': 1,
            'shoe_upper': 1,
            'upper_bag': 2,
            'bag_upper': 2,
            'upper_accessory': 3,
            'accessory_upper': 3,
            'bottom_shoe': 4,
            'shoe_bottom': 4,
            'bottom_bag': 5,
            'bag_bottom': 5,
            'bottom_accessory': 6,
            'accessory_bottom': 6,
            'shoe_bag': 7,
            'bag_shoe': 7,
            'shoe_accessory': 8,
            'accessory_shoe': 8,
            'bag_accessory': 9,
            'accessory_bag': 9,
        }

    def load_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        set_id, parts = self.data[index]
        choice = random.sample(list(parts.keys()), k=2)
        anchor_part, pos_part = choice[0], choice[1]
        anchor_img_path = os.path.join(self.root_dir, str(set_id), str(parts[anchor_part]['index'])+'.jpg')
        pos_img_path = os.path.join(self.root_dir, str(set_id), str(parts[pos_part]['index'])+'.jpg')

        neg_choice = self.data[index]
        while (pos_part not in neg_choice[1]) or (neg_choice[0] == set_id):
            neg_choice = random.choice(self.data)
        neg_img_path = os.path.join(self.root_dir, str(neg_choice[0]), str(neg_choice[1][pos_part]['index'])+'.jpg')

        pos_img = self.load_img(pos_img_path)
        anchor_img = self.load_img(anchor_img_path)
        neg_img = self.load_img(neg_img_path)

        condition = self.conditions['_'.join(choice)]

        if self.is_train:
            return anchor_img, pos_img, neg_img, condition
        else:
            target = random.randint(0, 1)
            if target == 0:
                return anchor_img, pos_img, target, condition
            elif target == 1:
                return anchor_img, neg_img, target, condition

    def __len__(self):
        return len(self.data)


# Test the loader
if __name__ == "__main__":
    img_size = 224
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor()
    ])
    d = CategoryDataset(transform=transform, use_mean_img=True)
    loader = DataLoader(d, 4, shuffle=True, num_workers=4, collate_fn=collate_fn)
