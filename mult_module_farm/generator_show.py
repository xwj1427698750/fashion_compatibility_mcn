import itertools
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torchvision import models
import numpy as np
import resnet
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders
from model_MVAE import MultiModuleGenerator
import shutil
import os
import matplotlib.pyplot as plt
# Leave a comment for this training, and it will be used for name suffix of log and saved model
import argparse
parser = argparse.ArgumentParser(description='Fashion Compatibility Evaluation.')
parser.add_argument('--model_path', type=str, default="./data_mix_model_diff_acc_train_generator_fuse_farm_atten(head_num=4)_mlmsff_(feature_size=96).pth")
parser.add_argument('--generator_type', type=str, default="mix")
parser.add_argument('--num_attention_heads', type=int, default=4)
parser.add_argument('--feature_size', type=int, default=96)
args = parser.parse_args()

print(args)
model_path = args.model_path
generator_type = args.generator_type
num_attention_heads = args.num_attention_heads
feature_size = args.feature_size
batch_size = 64
# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
    prepare_dataloaders(batch_size=batch_size, generator_type=generator_type)
)

# Load pretrained weights
device = torch.device("cuda:0")
model = MultiModuleGenerator(vocabulary=len(train_dataset.vocabulary), num_attention_heads=num_attention_heads, device=device, feature_size=feature_size).to(device)
model.load_state_dict(torch.load(model_path))
criterion = nn.BCELoss()

# Compatibility AUC test
model.eval()
is_correct = []
option_len = 4
out_dir = 'generator_show/'
img_source_path = '../data/images2/'
for i, _ in enumerate(test_dataset):
    print("\r#{}/{}".format(i, len(test_dataset)), end="", flush=True)
    items, name, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(i, option_len)
    # print("labels", labels)
    # print("option_labels", option_labels)
    # 给定的图像名
    given_img_ids = []  # 含有4件，前3件为给定图像，最后一件是目标单品
    for label in labels:
        given_img_ids.append(label.split('_')[-1])
    option_img_ids = []  # 含有3件， 为候选单品
    for label in option_labels:
        option_img_ids.append(label.split('_')[-1])

    if not os.path.exists(out_dir + str(i)):
        os.makedirs(out_dir + str(i))

    # 将给定查询单品和候选单品放入文件夹中
    for idx in range(len(given_img_ids)):
        if idx < 3:
            target_path = out_dir + str(i) + '/given_' + str(idx) + '.jpg'
        else:
            target_path = out_dir + str(i) + '/target.jpg'
        shutil.copyfile(img_source_path + given_img_ids[idx] + '.jpg', target_path)

    for idx in range(len(option_img_ids)):
        target_path = out_dir + str(i) + '/option_' + str(idx) + '.jpg'
        shutil.copyfile(img_source_path + option_img_ids[idx] + '.jpg', target_path)

    images = []
    for option in options:
        new_outfit = items.clone()
        option = option.unsqueeze(0)
        new_outfit = torch.cat((new_outfit, option), 0)
        images.append(new_outfit)
    images = torch.stack(images).to(device)
    names = torch.stack([name] * (option_len-1)).to(device)

    low_resolution_img, high_resolution_img, difference_score, z_mean, z_log_var, pos_out, neg_out = model(images, names)
    # low_resolution_img.shape torch.Size([3, 3, 128, 128])
    low_resolution_img0 = low_resolution_img[0].permute([1, 2, 0]).cpu().data.numpy()
    plt.imsave(out_dir + str(i) + '/low_rec.jpg', low_resolution_img0)
    high_resolution_img0 = high_resolution_img[0].permute([1, 2, 0]).cpu().data.numpy()
    plt.imsave(out_dir + str(i) + '/high_rec.jpg', high_resolution_img0)

    difference_score_true = (difference_score >= 0)
    difference_score_sum = torch.sum(difference_score_true.float())
    is_true = 'false'
    if int(difference_score_sum) == len(options):
        is_true = 'true'
    with open(out_dir + str(i) + '/' + is_true + '.txt', 'w', encoding='utf-8') as f:
        f.write(is_true)
        for score in difference_score:
            f.write(str(score.cpu().data.item()))
            f.write('\n')





