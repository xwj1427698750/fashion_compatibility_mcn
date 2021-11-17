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

# Leave a comment for this training, and it will be used for name suffix of log and saved model
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

# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
    prepare_dataloaders()
)

# Load pretrained weights
device = torch.device("cuda:1")
model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary),
                    vse_off=vse_off, pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats, layer_feature_size=layer_feature_size).to(device)
model.load_state_dict(torch.load(model_path))
criterion = nn.BCELoss()

# Compatibility AUC test
model.eval()
test_num = 10


auc_epochs = []
for epoch in range(test_num):
    total_loss = AverageMeter()
    outputs = []
    targets = []
    for batch_num, batch in enumerate(test_loader, 1):
        print("\r#{}/{}".format(batch_num, len(test_loader)), end="", flush=True)
        lengths, images, names, offsets, set_ids, labels, is_compat = batch
        images = images.to(device)
        target = is_compat.float().to(device)
        with torch.no_grad():
            output, _, _, _, _ = model._compute_feature_fusion_score(images)
            output = output.squeeze(dim=1)
            loss = criterion(output, target)
        total_loss.update(loss.item(), images.shape[0])
        outputs.append(output)
        targets.append(target)
    print()
    print("test:{} Test Loss: {:.4f}".format(epoch+1, total_loss.avg))
    outputs = torch.cat(outputs).cpu().data.numpy()
    targets = torch.cat(targets).cpu().data.numpy()
    auc = metrics.roc_auc_score(targets, outputs)
    print("test:{} AUC: {:.4f}".format(epoch + 1, auc))
    auc_epochs.append(auc)
auc_epochs = np.array(auc_epochs)
auc_mean = (np.sum(auc_epochs) - np.max(auc_epochs) - np.min(auc_epochs)) / (test_num-2)
print("AUC mean is {:.4f}".format(auc_mean))

# Fill in the blank evaluation
for option_len in [4, 5, 6]:
    print("option_len=", option_len)
    is_correct = []
    fitb_epochs = []
    for epoch in range(test_num):
        for i in range(len(test_dataset)):
            print("\r#{}/{}".format(i, len(test_dataset)), end="", flush=True)
            items, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(i, option_len)
            question_part = {"upper": 0, "bottom": 1, "shoe": 2, "bag": 3}.get(question_part)
            images = [items]

            for option in options:
                new_outfit = items.clone()
                new_outfit[question_part] = option
                images.append(new_outfit)
            images = torch.stack(images).to(device)
            output, _, _, _, _ = model._compute_feature_fusion_score(images)

            if output.argmax().item() == 0:
                is_correct.append(True)
            else:
                is_correct.append(False)
        print()
        fitb_acc = sum(is_correct) / len(is_correct)
        print("test:{} FitB ACC: {:.4f}".format(epoch + 1, fitb_acc))
        fitb_epochs.append(fitb_acc)
    fitb_epochs = np.array(fitb_epochs)
    fitb_mean = (np.sum(fitb_epochs) - np.max(fitb_epochs) - np.min(fitb_epochs)) / (test_num-2)
    print(f"average fitb acc is {fitb_mean}")
