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
parser.add_argument('--target_type', type=str, default="bottom")
parser.add_argument('--comment', type=str, default="wide_deep_attention_fuse")
parser.add_argument('--clip', type=int, default=5)
args = parser.parse_args()

print(args)
comment = args.comment
vse_off = args.vse_off
pe_off = args.pe_off
mlp_layers = args.mlp_layers
conv_feats = args.conv_feats
target_type = args.target_type
clip = args.clip

# Logger
config_logging(comment)

batch_size = 16
# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
    prepare_dataloaders(batch_size=batch_size, target_type=target_type)
)

# Device
device = torch.device("cuda:0")

# Model
model = MultiModuleGenerator(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary),
                     vse_off=vse_off, pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats, target_type=target_type, device=device)

# Train process
def train(model, device, train_loader, val_loader, comment):
    type_to_id = {'upper': 0, 'bottom': 1, 'bag': 2, 'shoe': 3}
    target_id = type_to_id[target_type]
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    saver = BestSaver(comment)
    epochs = 50

    for epoch in range(1, epochs + 1):
        logging.info("Train Phase, Epoch: {}".format(epoch))

        total_losses = AverageMeter()
        clf_losses = AverageMeter()
        vse_losses = AverageMeter()
        # Train phase
        model.train()
        for batch_num, batch in enumerate(train_loader, 1):
            lengths, images, names, set_id, labels, is_compat = batch
            # print(labels)
            # print("is_compat", is_compat)
            batch_size, _, _, _, img_size = images.shape  # [16, 4, 3, 224, 224]
            images = images.to(device)
            # labels 显示样本的组成 单独的项为(set_id)_(index:单张照片的名字),  ['21027552963e099f780f73a5ffde30ed_be9627e0c2e43ee96017e288b03eed96', ..], length = batch_size
            # images.shape [8, 5, 3, 224, 224], [batch_size, item_length, C, H, W ]
            # names is a list with length 40 = 8 * 5, each item of which is a tensor 1-dim like:tensor([772,  68,  72, 208])
            # Forward   前向训练只需要 图像和文本数据
            output, vse_loss = model(images, names)
            # BCE loss
            target = is_compat.float().to(device)
            output = output.squeeze(dim=1)
            clf_loss = criterion(output, target)

            # sum all the loss
            total_loss = clf_loss + vse_loss

            # Update Recoder
            total_losses.update(total_loss.item(), images.shape[0])
            clf_losses.update(clf_loss.item(), images.shape[0])
            vse_losses.update(vse_loss.item(), images.shape[0])

            # Backpropagation
            model.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if batch_num % 50 == 0:
                logging.info(
                    "[{}/{}] #{} clf_loss: {:.4f}, vse_loss: {:.4f},total_loss:{:.4f}".format(
                        epoch, epochs, batch_num, clf_losses.val, vse_losses.val, total_losses.val
                    )
                )
        scheduler.step()
        logging.info("Train Loss (total_loss): {:.4f}".format(total_losses.avg))

        # Valid Phase 验证fitb，取结果好的
        logging.info("Valid Phase, Epoch: {}".format(epoch))
        model.eval()
        clf_losses = AverageMeter()
        outputs = []
        targets = []
        for batch_num, batch in enumerate(val_loader, 1):
            lengths, images, names, set_id, labels, is_compat = batch
            images = images.to(device)
            target = is_compat.float().to(device)
            with torch.no_grad():
                output, _ = model.conpute_compatible_score(images)
                output = output.squeeze(dim=1)
                clf_loss = criterion(output, target)
            clf_losses.update(clf_loss.item(), images.shape[0])
            outputs.append(output)
            targets.append(target)
        logging.info("Valid Loss (clf_loss): {:.4f}".format(clf_losses.avg))
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
        saver.save(auc, accuracy, model.state_dict(), epoch)
        logging.info("Best AUC is : {:.4f} Best_epoch is {}".format(saver.best_auc, saver.best_auc_epoch))  # 输出已经选择好的最佳模型
        logging.info("Best ACC is : {:.4f} Best_epoch is {}".format(saver.best_acc, saver.best_acc_epoch))  # 输出已经选择好的最佳模型
        logging.info("Best ACC(AUC) is : {:.4f} Best_epoch is {}".format(saver.best_acc_auc, saver.best_acc_auc_epoch))  # 输出已经选择好的最佳模型
if __name__ == "__main__":
    train(model, device, train_loader, val_loader, comment)
