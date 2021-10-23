import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torch.optim import lr_scheduler
from torchvision import models

from model_MVAE import MultiModuleGenerator
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders

# Leave a comment for this training, and it will be used for name suffix of log and saved model
import argparse
parser = argparse.ArgumentParser(description='Fashion Compatibility Training.')
parser.add_argument('--vse_off', action="store_true")
parser.add_argument('--pe_off', action="store_true")
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--conv_feats', type=str, default="1234")
parser.add_argument('--generator_type', type=str, default="upper")
parser.add_argument('--comment', type=str, default="generator_fuse")
parser.add_argument('--clip', type=int, default=5)
args = parser.parse_args()

print(args)
comment = args.comment
vse_off = args.vse_off
pe_off = args.pe_off
mlp_layers = args.mlp_layers
conv_feats = args.conv_feats
generator_type = args.generator_type
clip = args.clip

# Logger
config_logging(comment)

batch_size = 8
# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
    prepare_dataloaders(batch_size=batch_size)
)

# Device
device = torch.device("cuda:0")

# Model
model = MultiModuleGenerator(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary),
                     vse_off=vse_off, pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats, encoder_path="model_mcn.pth", generator_type=generator_type, device=device)

# Train process
def train(model, device, train_loader, val_loader, comment):
    type_to_id = {'upper': 0, 'bottom': 1, 'bag': 2, 'shoe': 3}
    generator_id = type_to_id[generator_type]
    model = model.to(device)
    criterion = nn.BCELoss()
    get_l2_loss = nn.MSELoss()
    get_l1_loss = nn.L1Loss()
    log_sigmoid = nn.LogSigmoid()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    saver = BestSaver(comment)
    epochs = 50

    for epoch in range(1, epochs + 1):
        logging.info("Train Phase, Epoch: {}".format(epoch))

        total_losses = AverageMeter()
        clf_losses = AverageMeter()
        bpr_losses = AverageMeter()
        l2_losses = AverageMeter()
        l1_losses = AverageMeter()
        kl_losses = AverageMeter()
        # Train phase
        model.train()
        for batch_num, batch in enumerate(train_loader, 1):
            lengths, images, names, set_id, labels = batch
            batch_size, _, _, _, img_size = images.shape
            images = images.to(device)
            generator_target_img = images[:, generator_id, :, :, :]
            # labels 显示样本的组成 单独的项为(set_id)_(index:单张照片的名字),  ['21027552963e099f780f73a5ffde30ed_be9627e0c2e43ee96017e288b03eed96', ..], length = batch_size
            # images.shape [8, 5, 3, 224, 224], [batch_size, item_length, C, H, W ]
            # names is a list with length 40 = 8 * 5, each item of which is a tensor 1-dim like:tensor([772,  68,  72, 208])
            # Forward   前向训练只需要 图像和文本数据
            out_pos, out_neg, low_resolution_img, high_resolution_img, difference_score, z_mean, z_log_var = model(images, names)

            # BCE Loss
            clf_loss_param = 10
            pos_outfit_target = torch.ones(size=[batch_size]).to(device)
            out_pos = out_pos.squeeze(dim=1)
            pos_clf_loss = clf_loss_param * criterion(out_pos, pos_outfit_target)

            neg_outfit_target = torch.zeros(size=[batch_size]).to(device)
            out_neg = out_neg.squeeze(dim=1)
            neg_clf_loss = clf_loss_param * criterion(out_neg, neg_outfit_target)

            # BPR LOSS
            bpr_param = 100
            bpr_loss = bpr_param * torch.sum(-log_sigmoid(difference_score))

            # Generator LOSS
            shape = low_resolution_img.shape
            l1_loss_param = shape[1] * shape[2] * shape[3]
            l2_loss_param = shape[1] * shape[2] * shape[3]

            l2_loss = l1_loss_param * get_l2_loss(low_resolution_img, generator_target_img)
            l1_loss = l2_loss_param * get_l1_loss(high_resolution_img, generator_target_img)

            # KL LOSS
            kl_loss = torch.sum(-0.5*torch.sum(1 + z_log_var-torch.square(z_mean) - torch.exp(z_log_var)))

            # Sum all losses up
            total_loss = pos_clf_loss + neg_clf_loss + bpr_loss + l2_loss + l1_loss + kl_loss

            # Update Recoder
            clf_losses.update(pos_clf_loss.item() + neg_clf_loss.item(), images.shape[0])
            bpr_losses.update(bpr_loss.item(), images.shape[0])
            l2_losses.update(l2_loss.item(), images.shape[0])
            l1_losses.update(l1_loss.item(), images.shape[0])
            kl_losses.update(kl_loss.item(), images.shape[0])
            total_losses.update(total_loss.item(), images.shape[0])
            # Backpropagation
            model.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if batch_num % 50 == 0:
                logging.info(
                    "[{}/{}] #{} clf_loss: {:.4f}, bpr_loss: {:.4f}, l2_loss: {:.4f}, l1_loss: {:.4f}, kl_loss: {:.4f}, total_loss:{:.4f}".format(
                        epoch, epochs, batch_num, clf_losses.val, bpr_losses.val, l2_losses.val, l1_losses.val, kl_losses.val, total_losses.val
                    )
                )
        scheduler.step()
        logging.info("Train Loss (total_loss): {:.4f}".format(total_losses.avg))

        # Valid Phase 验证fitb，取结果好的
        logging.info("Valid Phase, Epoch: {}".format(epoch))
        model.eval()
        clf_losses = AverageMeter()
        clf_diff_acc = AverageMeter()
        outputs = []
        targets = []
        for batch_num, batch in enumerate(val_loader, 1):
            lengths, images, names, set_id, labels = batch
            images = images.to(device)
            batch_size, _, _, _, img_size = images.shape  # 在batch_size = 8 的情况下，最后一个batch的大小是4，多以在这里加一句，在最后一个batch的时候，更改下batch_size的大小
            pos_outfit_target = torch.ones(size=[batch_size]).to(device)
            neg_outfit_target = torch.zeros(size=[batch_size]).to(device)

            with torch.no_grad():
                out_pos, out_neg, low_resolution_img, high_resolution_img, difference_score, z_mean, z_log_var = model(images, names)
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
        logging.info("Valid Loss (clf_loss): {:.4f}".format(clf_losses.avg))
        logging.info("clf_diff_acc: {:.4f}".format(clf_diff_acc.avg))
        outputs = torch.cat(outputs).cpu().data.numpy()
        targets = torch.cat(targets).cpu().data.numpy()
        auc = metrics.roc_auc_score(targets, outputs)
        logging.info("AUC: {:.4f}".format(auc))
        predicts = np.where(outputs > 0.5, 1, 0)
        accuracy = metrics.accuracy_score(predicts, targets)
        logging.info("Accuracy@0.5: {:.4f}".format(accuracy))
        positive_loss = -np.log(outputs[targets==1]).mean()
        logging.info("Positive loss: {:.4f}".format(positive_loss))
        positive_acc = sum(outputs[targets==1]>0.5) / len(outputs)
        logging.info("Positive accuracy: {:.4f}".format(positive_acc))
        # Save best model
        saver.save(auc, clf_diff_acc.avg, model.state_dict(), epoch)
        logging.info("Best AUC is : {:.4f} Best_epoch is {}".format(saver.best_auc, saver.best_auc_epoch))  # 输出已经选择好的最佳模型
        logging.info("Best ACC is : {:.4f} Best_epoch is {}".format(saver.best_acc, saver.best_acc_epoch))  # 输出已经选择好的最佳模型
        logging.info("Best ACC(AUC) is : {:.4f} Best_epoch is {}".format(saver.best_acc_auc, saver.best_acc_auc_epoch))  # 输出已经选择好的最佳模型
if __name__ == "__main__":
    train(model, device, train_loader, val_loader, comment)
