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

# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepare_dataloaders(
    root_dir="../../data/images2",
    data_dir="../../data",
    batch_size=12
)

# Load pretrained weights
device = torch.device("cuda:0")
model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary)).to(device)
model.load_state_dict(torch.load("./model_concat_train.pth"))
criterion = nn.BCELoss()

# Compatibility AUC test
model.eval()
test_num = 10
auc_epochs = []
fitb_epochs = []
for epoch in range(test_num):
    total_loss = 0
    outputs = []
    targets = []
    for batch_num, batch in enumerate(test_loader, 1):
        print("\r#{}".format(batch_num), end="", flush=True)
        lengths, images, names, offsets, set_ids, labels, is_compat = batch
        images = images.to(device)
        target = is_compat.float().to(device)
        with torch.no_grad():
            output, _, _ = model._compute_score(images)
            output = output.squeeze(dim=1)
            loss = criterion(output, target)
        total_loss += loss.item()
        outputs.append(output)
        targets.append(target)
    print()
    print("test:{} Test Loss: {:.4f}".format(epoch+1, total_loss / batch_num))
    outputs = torch.cat(outputs).cpu().data.numpy()
    targets = torch.cat(targets).cpu().data.numpy()
    auc = metrics.roc_auc_score(targets, outputs)
    print("test:{} AUC: {:.4f}".format(epoch + 1, auc))
    auc_epochs.append(auc)

    # Fill in the blank evaluation
    is_correct = []
    for i in range(len(test_dataset)):
        print("\r#{}".format(i), end="", flush=True)
        items, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(i)
        question_part = {"upper": 0, "bottom": 1, "shoe": 2, "bag": 3, "accessory": 4}.get(question_part)
        images = [items]

        for option in options:
            new_outfit = items.clone()
            new_outfit[question_part] = option
            images.append(new_outfit)
        images = torch.stack(images).to(device)
        output, _, _ = model._compute_score(images)

        if output.argmax().item() == 0:
            is_correct.append(True)
        else:
            is_correct.append(False)
    print()
    fitb_acc = sum(is_correct) / len(is_correct)
    print("test:{} FitB ACC: {:.4f}".format(epoch+1, fitb_acc))
    fitb_epochs.append(fitb_acc)
auc_epochs = np.array(auc_epochs)
fitb_epochs = np.array(fitb_epochs)
print(f"average compat AUC is {auc_epochs.mean()} average fitb acc is {fitb_epochs.mean()}")
