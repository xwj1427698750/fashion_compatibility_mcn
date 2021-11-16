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
test_num = 10
auc_epochs = []
diff_epochs = []
for epoch in range(test_num):
    clf_losses = AverageMeter()
    clf_diff_acc = AverageMeter()
    outputs = []
    targets = []
    for batch_num, batch in enumerate(test_loader, 1):
        print("\r#{}/{} ".format(batch_num, len(test_loader)), end="", flush=True)
        lengths, images, names, set_id, labels, change_ids = batch
        images = images.to(device)
        names = names.to(device)
        batch_size, _, _, _, img_size = images.shape  # 训练集中，在batch_size = 8 的情况下，最后一个batch的大小是4，多以在这里加一句，在最后一个batch的时候，更改下batch_size的大小
        pos_target = torch.ones(size=[batch_size])
        neg_target = torch.zeros(size=[batch_size])

        with torch.no_grad():
            _, _, difference_score, z_mean, z_log_var, pos_out, neg_out = model(images, names)
            # out_pos :(batch_size, 1)          low_resolution_img:(batch_size, 3, 224, 224)         difference_score:   (batch_size, 1)
        pos_out = pos_out.squeeze(dim=1)
        neg_out = neg_out.squeeze(dim=1)
        output = torch.cat((pos_out, neg_out))
        target = torch.cat((pos_target, neg_target)).to(device)
        clf_loss = criterion(output, target)
        # 通过分类模型计算的一种方式
        clf_losses.update(clf_loss.item(), images.shape[0])
        outputs.append(output)
        targets.append(target)
        # 通过层级特征得分大小计算的一种方式
        difference_score = difference_score
        difference_score_true = (difference_score >= 0)
        difference_score_sum = torch.sum(difference_score_true.float())
        diff_acc = difference_score_sum / difference_score.shape[0]
        clf_diff_acc.update(diff_acc.item(), 1)

        # # 准确率的另外一种计算方式
        # pos_gt_neg = (out_pos >= out_neg)
        # pos_gt_neg_sum = torch.sum(pos_gt_neg.float())
        # pos_gt_neg_acc.update(pos_gt_neg_sum, batch_size)

        # # 或运算
        # mix_true = difference_score_true + pos_gt_neg
        # mix_true_sum = torch.sum(mix_true.float())
        # mix_true_acc.update(mix_true_sum, batch_size)
    print()
    print("Test Loss clf_diff_acc: {:.4f}".format(clf_diff_acc.avg))
    # print("mix_true_acc: {:.4f}".format(mix_true_acc.avg))
    # print("pos_gt_neg_acc: {:.4f}".format(pos_gt_neg_acc.avg))
    outputs = torch.cat(outputs).cpu().data.numpy()
    targets = torch.cat(targets).cpu().data.numpy()
    auc = metrics.roc_auc_score(targets, outputs)
    print("test:{} AUC: {:.4f}".format(epoch + 1, auc))
    # predicts = np.where(outputs > 0.5, 1, 0)
    # accuracy = metrics.accuracy_score(predicts, targets)
    # print("Accuracy@0.5: {:.4f}".format(accuracy))
    # positive_loss = -np.log(outputs[targets == 1]).mean()
    # print("Positive loss: {:.4f}".format(positive_loss))
    # positive_acc = sum(outputs[targets == 1] > 0.5) / len(outputs)
    # print("Positive accuracy: {:.4f}".format(positive_acc))
    auc_epochs.append(auc)
    diff_epochs.append(clf_diff_acc.avg)
auc_epochs = np.array(auc_epochs)
diff_epochs = np.array(diff_epochs)
auc_mean = (np.sum(auc_epochs) - np.max(auc_epochs) - np.min(auc_epochs)) / (test_num-2)
diff_acc_mean = (np.sum(diff_epochs) - np.max(diff_epochs) - np.min(diff_epochs)) / (test_num-2)
print(f"average compat AUC is {auc_mean} , average diff auc is {diff_acc_mean}")

# fitb测试
for option_len in [4, 5, 6]:
    print("option_len=", option_len)
    fitb_epochs = []
    for epoch in range(test_num):
        is_correct = []
        for i, _ in enumerate(test_dataset):
            print("\r#{}/{}".format(i, len(test_dataset)), end="", flush=True)
            items, name, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(i, option_len)
            question_part_id = {"upper": 0, "bottom": 1, "bag": 2, "shoe": 3}.get(question_part)
            images = []
            for option in options:
                new_outfit = items.clone()
                option = option.unsqueeze(0)
                new_outfit = torch.cat((new_outfit, option), 0)
                images.append(new_outfit)
            images = torch.stack(images).to(device)
            names = torch.stack([name] * (option_len-1)).to(device)
            # print("names.shape", names.shape)
            _, _, difference_score, _, _, pos_out, neg_out = model(images, names)

            difference_score_true = (difference_score >= 0)
            difference_score_sum = torch.sum(difference_score_true.float())

            # # 准确率的另外一种计算方式
            # pos_gt_neg = (out_pos >= out_neg).squeeze(1)
            # pos_gt_neg_sum = torch.sum(pos_gt_neg.float())
            # # 或运算
            # mix_true = difference_score_true + pos_gt_neg
            # mix_true_sum = torch.sum(mix_true.float())

            # if int(mix_true_sum) == len(options):
            if int(difference_score_sum) == len(options):
                is_correct.append(True)
            else:
                is_correct.append(False)
            del items, options, images
        print()
        fitb_acc = sum(is_correct) / len(is_correct)
        print("test:{} FitB ACC: {:.4f}".format(epoch + 1, fitb_acc))
        fitb_epochs.append(fitb_acc)
    fitb_epochs = np.array(fitb_epochs)
    fitb_mean = (np.sum(fitb_epochs) - np.max(fitb_epochs) - np.min(fitb_epochs)) / (test_num-2)
    print(f"average fitb acc is {fitb_mean}")
