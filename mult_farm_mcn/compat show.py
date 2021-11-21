import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torchvision import models
import numpy as np
import resnet
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders
from model import CompatModel

import shutil
import os
import matplotlib.pyplot as plt

# Leave a comment for this training, and it will be used for name suffix of log and saved model
# 分成四类 正确的分成正确的TP，正确的分成错误的TN， 错误的分成正确的FP，错误的分成错误的FN
import argparse
parser = argparse.ArgumentParser(description='Fashion Compatibility Evaluation.')
parser.add_argument('--vse_off', action="store_true")
parser.add_argument('--pe_off', action="store_true")
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--conv_feats', type=str, default="1234")
parser.add_argument('--model_path', type=str, default="./model_train_v5_deep_scale(double_conv)_xavir_(layer_feature_size=160).pth")
parser.add_argument('--layer_feature_size', type=int, default=160)
args = parser.parse_args()

print(args)
vse_off = args.vse_off
pe_off = args.pe_off
mlp_layers = args.mlp_layers
conv_feats = args.conv_feats
model_path = args.model_path
layer_feature_size = args.layer_feature_size
batch_size = 1
# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
    prepare_dataloaders(batch_size=batch_size)
)

# Load pretrained weights
device = torch.device("cuda:1")
model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary),
                    vse_off=vse_off, pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats, layer_feature_size=layer_feature_size).to(device)
model.load_state_dict(torch.load(model_path))
criterion = nn.BCELoss()

out_dir = 'compat_show/'
img_source_path = '../data/images2/'

# Compatibility AUC test
model.eval()
total_loss = AverageMeter()

for i, batch in enumerate(test_loader, 1):
    print("\r#{}/{}".format(i, len(test_loader)), end="", flush=True)
    lengths, images, names, offsets, set_ids, labels, is_compat = batch
    images = images.to(device)
    target = is_compat.float().to(device)

    img_ids = []  # 一套单品含有4件
    labels = labels[0]
    for label in labels:
        img_ids.append(label.split('_')[-1])

    with torch.no_grad():
        output, _, _, _, _ = model._compute_feature_fusion_score(images)
        output = output.squeeze(dim=1)
        loss = criterion(output, target)
    total_loss.update(loss.item(), images.shape[0])

    sub_dir = 'FN/'
    if is_compat.item() == 1 and output >= 0.5:
        sub_dir = 'TP/'
    elif is_compat.item() == 1 and output < 0.5:
        sub_dir = 'TN/'
    elif is_compat.item() == 0 and output < 0.5:
        sub_dir = 'FP/'

    for idx in range(len(labels)):
        target_path = out_dir + sub_dir + str(i)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        target_path = target_path + "/" + img_ids[idx] + '.jpg'
        shutil.copyfile(img_source_path + img_ids[idx] + '.jpg', target_path)

    with open(out_dir + sub_dir + str(i) + '/' + str(output) + '.txt', 'w', encoding='utf-8') as f:
        f.write('\n')
