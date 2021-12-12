"""
| GGNN                                                         |   73.80   |   51.73   |
"""
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

from polyvore_dataset import graph_collate_fn
from tqdm import tqdm, trange
import dgl

# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepare_dataloaders(
    root_dir="../../data/images2",
    data_dir="../../data",
    batch_size=12,
    collate_fn=graph_collate_fn,
    use_mean_img=False,
    num_workers=6
)
def img2graph(items):
    l = items.shape[0]
    g = dgl.DGLGraph()
    g.add_nodes(l)
    edges = list(zip(*itertools.combinations(range(l), 2)))
    g.add_edges(*edges)
    g.ndata['img'] = items
    return g

# Load pretrained weights
device = torch.device("cuda:0")
model = CompatModel(embed_size=512, need_rep=True, vocabulary=len(train_dataset.vocabulary)).to(device)
model.load_state_dict(torch.load("./model_ggnn_train.pth"))
print("Successfully load model weight...")
model.eval()
criterion = nn.BCELoss()
test_num = 10
auc_epochs = []
for epoch in range(test_num):
    # Compatibility AUC test
    total_loss = 0
    outputs = []
    targets = []
    for batch in tqdm(test_loader):
        lengths, batch_g, names, offsets, set_ids, labels, is_compat = batch
        batch_g = batch_g.to(device)
        target = is_compat.float().to(device)
        with torch.no_grad():
            output, _, _ = model._compute_score(batch_g)
            output = output.squeeze(dim=1)
            loss = criterion(output, target)
        total_loss += loss.item()
        outputs.append(output)
        targets.append(target)
    print()
    print("Test Loss: {:.4f}".format(total_loss / len(test_loader)))
    outputs = torch.cat(outputs).cpu().data.numpy()
    targets = torch.cat(targets).cpu().data.numpy()
    auc = metrics.roc_auc_score(targets, outputs)
    print("test:{} AUC: {:.4f}".format(epoch + 1, auc))
    auc_epochs.append(auc)

auc_mean = (np.sum(auc_epochs) - np.max(auc_epochs) - np.min(auc_epochs)) / (test_num-2)
print(f"average compat AUC is {auc_mean}")

for option_len in [4, 5, 6]:
    fitb_epochs = []
    for epoch in range(test_num):
        # # Compatibility AUC test
        # total_loss = 0
        # outputs = []
        # targets = []
        # for batch in tqdm(test_loader):
        #     lengths, batch_g, names, offsets, set_ids, labels, is_compat = batch
        #     batch_g = batch_g.to(device)
        #     target = is_compat.float().to(device)
        #     with torch.no_grad():
        #         output, _, _ = model._compute_score(batch_g)
        #         output = output.squeeze(dim=1)
        #         loss = criterion(output, target)
        #     total_loss += loss.item()
        #     outputs.append(output)
        #     targets.append(target)
        # print()
        # print("Test Loss: {:.4f}".format(total_loss / len(test_loader)))
        # outputs = torch.cat(outputs).cpu().data.numpy()
        # targets = torch.cat(targets).cpu().data.numpy()
        # auc = metrics.roc_auc_score(targets, outputs)
        # print("test:{} AUC: {:.4f}".format(epoch + 1, auc))
        # auc_epochs.append(auc)

        # Fill in the blank evaluation
        is_correct = []
        for i in trange(len(test_dataset)):
            items, exist_parts, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(i,option_len)
            question_part = exist_parts.index(question_part)

            batch_g = [img2graph(items)]
            for option in options:
                new_outfit = items.clone()
                new_outfit[question_part] = option
                new_g = img2graph(new_outfit)
                batch_g.append(new_g)
            batch_g = dgl.batch(batch_g).to(device)
            output, _, _ = model._compute_score(batch_g)

            if output.argmax().item() == 0:
                is_correct.append(True)
            else:
                is_correct.append(False)
        print()
        fitb_acc = sum(is_correct) / len(is_correct)
        print("test:{} FitB ACC: {:.4f}".format(epoch + 1, fitb_acc))
        fitb_epochs.append(fitb_acc)
    fitb_mean = (np.sum(fitb_epochs) - np.max(fitb_epochs) - np.min(fitb_epochs)) / (test_num-2)
    print(f"average fitb acc is {fitb_mean}")



