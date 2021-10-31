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
parser.add_argument('--model_path', type=str, default="./model_train_wi_deep_multi_task.pth")
parser.add_argument('--target_type', type=str, default="bottom")
args = parser.parse_args()

print(args)
vse_off = args.vse_off
pe_off = args.pe_off
mlp_layers = args.mlp_layers
conv_feats = args.conv_feats
model_path = args.model_path
target_type = args.target_type

# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
    prepare_dataloaders(batch_size=16, target_type=target_type)
)

# Load pretrained weights
device = torch.device("cuda:0")
model = MultiModuleGenerator(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary), vse_off=vse_off,
                             pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats, target_type=target_type, device=device).to(device)
model.load_state_dict(torch.load(model_path))
criterion = nn.BCELoss()

# Compatibility AUC test
model.eval()
test_num = 10
auc_epochs = []
fitb_epochs = []
log_sigmoid = nn.LogSigmoid()
for epoch in range(test_num):
    total_loss = AverageMeter()
    clf_losses = AverageMeter()
    bpr_losses = AverageMeter()
    clf_diff_acc = AverageMeter()
    outputs = []
    targets = []
    for batch_num, batch in enumerate(test_loader, 1):
        print("\r#{}/{}".format(batch_num, len(test_loader)), end="", flush=True)
        lengths, images, names, set_id, labels = batch
        images = images.to(device)
        with torch.no_grad():
            pos_out, neg_out, diff, _ = model.conpute_compatible_score(images)
            # clf_loss
        batch_size, _, _, _, img_size = images.shape  # [16, 5, 3, 224, 224]
        pos_target = torch.ones(batch_size)
        pos_out = pos_out.squeeze(dim=1)
        neg_target = torch.zeros(batch_size)
        neg_out = neg_out.squeeze(dim=1)
        output = torch.cat((pos_out, neg_out))
        target = torch.cat((pos_target, neg_target)).to(device)
        clf_loss = criterion(output, target)
        # bpr_loss
        bpr_param = 1
        bpr_loss = bpr_param * torch.sum(-log_sigmoid(diff))
        # diff score
        diff = diff.squeeze(dim=1)
        diff_sum = torch.sum((diff > 0).float())
        clf_diff_acc.update(diff_sum.item(), images.shape[0])
        clf_losses.update(clf_loss.item(), images.shape[0])
        bpr_losses.update(bpr_loss.item(), images.shape[0])
        outputs.append(output)
        targets.append(target)
    outputs = torch.cat(outputs).cpu().data.numpy()
    targets = torch.cat(targets).cpu().data.numpy()
    logging.info("clf_diff_acc@0.5: {:.4f}".format(clf_diff_acc.avg))
    auc = metrics.roc_auc_score(targets, outputs)
    logging.info("AUC: {:.4f}".format(auc))
    predicts = np.where(outputs > 0.5, 1, 0)
    accuracy = metrics.accuracy_score(predicts, targets)
    logging.info("Accuracy@0.5: {:.4f}".format(accuracy))

    positive_loss = -np.log(outputs[targets == 1]).mean()
    logging.info("Positive loss: {:.4f}".format(positive_loss))
    positive_acc = sum(outputs[targets == 1] > 0.5) / len(outputs)
    logging.info("Positive accuracy: {:.4f}".format(positive_acc))
    print()
    print("test:{} Test Loss: {:.4f}".format(epoch + 1, total_loss.avg))
    outputs = torch.cat(outputs).cpu().data.numpy()
    targets = torch.cat(targets).cpu().data.numpy()
    auc = metrics.roc_auc_score(targets, outputs)
    print("test:{} AUC: {:.4f}".format(epoch + 1, auc))
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
            new_outfit = torch.cat((new_outfit, option), 1)
            images.append(new_outfit)
        images = torch.stack(images).to(device)
        pos_out, neg_out, diff, _ = model.conpute_compatible_score(images)
        diff = diff.squeeze(dim=1)
        diff = torch.sum((diff > 0).float())
        if int(diff) == len(options):
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
