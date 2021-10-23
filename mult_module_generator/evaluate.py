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

# Leave a comment for this training, and it will be used for name suffix of log and saved model
import argparse
parser = argparse.ArgumentParser(description='Fashion Compatibility Evaluation.')
parser.add_argument('--vse_off', action="store_true")
parser.add_argument('--pe_off', action="store_true")
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--conv_feats', type=str, default="1234")
parser.add_argument('--model_path', type=str, default="./model_train_generator_fuse.pth")
parser.add_argument('--generator_type', type=str, default="upper")
args = parser.parse_args()

print(args)
vse_off = args.vse_off
pe_off = args.pe_off
mlp_layers = args.mlp_layers
conv_feats = args.conv_feats
model_path = args.model_path
generator_type = args.generator_type

# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
    prepare_dataloaders()
)

# Load pretrained weights
device = torch.device("cuda:1")
model = MultiModuleGenerator(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary), vse_off=vse_off,
                             pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats, encoder_path="model_mcn.pth",
                             generator_type=generator_type, device=device).to(device)
model.load_state_dict(torch.load(model_path))
criterion = nn.BCELoss()

# Compatibility AUC test
model.eval()
test_num = 10
auc_epochs = []
fitb_epochs = []
for epoch in range(test_num):
    clf_losses = AverageMeter()
    clf_diff_acc = AverageMeter()
    outputs = []
    targets = []
    for batch_num, batch in enumerate(test_loader, 1):
        print("\r#{}/{}".format(batch_num, len(test_loader)), end="", flush=True)
        lengths, images, names, set_id, labels = batch
        images = images.to(device)
        batch_size, _, _, _, img_size = images.shape  # 训练集中，在batch_size = 8 的情况下，最后一个batch的大小是4，多以在这里加一句，在最后一个batch的时候，更改下batch_size的大小
        pos_outfit_target = torch.ones(size=[batch_size]).to(device)
        neg_outfit_target = torch.zeros(size=[batch_size]).to(device)

        with torch.no_grad():
            out_pos, out_neg, _, _, difference_score, _, _ = model(images, names)
            # out_pos :(batch_size, 1)          low_resolution_img:(batch_size, 3, 224, 224)         difference_score:   (batch_size, 1)
            out_pos = out_pos.squeeze(dim=1)
            out_neg = out_neg.squeeze(dim=1)
            pos_clf_loss = criterion(out_pos, pos_outfit_target)
            neg_clf_loss = criterion(out_neg, neg_outfit_target)
        # 通过分类模型计算的一种方式
        clf_losses.update(pos_clf_loss.item() + neg_clf_loss.item(), images.shape[0])
        output = torch.cat((out_pos, out_neg))
        outputs.append(output)
        target = torch.cat((pos_outfit_target, neg_outfit_target))
        targets.append(target)
        # 通过层级特征得分大小计算的一种方式
        difference_score = difference_score.squeeze(dim=1)
        difference_score_sum = torch.sum((difference_score > 0).float())
        diff_acc = difference_score_sum / difference_score.shape[0]
        clf_diff_acc.update(diff_acc.item(), 1)
    print("Test Loss (clf_loss): {:.4f}".format(clf_losses.avg))
    print("clf_diff_acc: {:.4f}".format(clf_diff_acc.avg))
    outputs = torch.cat(outputs).cpu().data.numpy()
    targets = torch.cat(targets).cpu().data.numpy()
    auc = metrics.roc_auc_score(targets, outputs)
    print("test:{} AUC: {:.4f}".format(epoch + 1, auc))
    predicts = np.where(outputs > 0.5, 1, 0)
    accuracy = metrics.accuracy_score(predicts, targets)
    print("Accuracy@0.5: {:.4f}".format(accuracy))
    positive_loss = -np.log(outputs[targets == 1]).mean()
    print("Positive loss: {:.4f}".format(positive_loss))
    positive_acc = sum(outputs[targets == 1] > 0.5) / len(outputs)
    print("Positive accuracy: {:.4f}".format(positive_acc))
    auc_epochs.append(auc)


    # Fill in the blank evaluation
    is_correct = []
    for i in range(len(test_dataset)):
        print("\r#{}/{}".format(i, len(test_dataset)), end="", flush=True)
        items, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(i)
        question_part_id = {"upper": 0, "bottom": 1, "bag": 2, "shoe": 3}.get(question_part)
        images = []
        for option in options:
            new_outfit = items.clone()
            option = option.unsqueeze(0)
            new_outfit = torch.cat((new_outfit, option), 0)
            images.append(new_outfit)
        images = torch.stack(images).to(device)
        names = []  # 未使用
        # print("images.shape", images.shape) shape没有问题
        out_pos, out_neg, _, _, difference_score, _, _ = model(images, names)
        difference_score = difference_score.squeeze(dim=1)
        difference_score_sum = torch.sum((difference_score > 0).float())
        # print("difference_score_sum:{:.4f}".format(difference_score_sum))
        # print("out_pos.shape = {}, out_neg.shape = {}".format(out_pos, out_neg))
        if int(difference_score_sum) == len(options):
            is_correct.append(True)
        else:
            is_correct.append(False)
    print()
    fitb_acc = sum(is_correct) / len(is_correct)
    print("test:{} FitB ACC: {:.4f}".format(epoch + 1, fitb_acc))
    fitb_epochs.append(fitb_acc)
auc_epochs = np.array(auc_epochs)
fitb_epochs = np.array(fitb_epochs)
print(f"average compat AUC is {auc_epochs.mean()} average fitb acc is {fitb_epochs.mean()}")
