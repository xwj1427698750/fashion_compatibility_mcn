import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim import lr_scheduler
from model import CompatModel
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders
import argparse

# Train process
def train(model, device, train_loader, val_loader, comment, clip):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
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
            images, names, _, _, is_compat = batch
            images = images.to(device)
            # is_compat is a tensor([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0]) length = batch_size
            # images.shape [16, 4, 3, 224, 224], [batch_size, item_length, C, H, W ]
            # names is a list with length 80 = 16 * 4, each item of which is a tensor 1-dim like: tensor([772,  68,  72, 208])
            # Forward   前向训练只需要 图像和文本数据

            output, vse_loss = model(images, names)

            # BCE Loss
            target = is_compat.float().to(device)
            output = output.squeeze(dim=1)
            clf_loss = criterion(output, target)

            # Sum all losses up
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
                    "[{}/{}] #{} clf_loss: {:.4f}, vse_loss: {:.4f},  total_loss:{:.4f}".format(
                        epoch, epochs, batch_num, clf_losses.val, vse_losses.val, total_losses.val
                    )
                )
        logging.info("Train Loss (clf_loss): {:.4f}".format(clf_losses.avg))
        scheduler.step()

        # Valid Phase
        logging.info("Valid Phase, Epoch: {}".format(epoch))
        model.eval()
        clf_losses = AverageMeter()
        outputs = []
        targets = []
        for batch_num, batch in enumerate(val_loader, 1):
            images, names, _, _, is_compat = batch
            images = images.to(device)
            target = is_compat.float().to(device)
            with torch.no_grad():
                output, _, _ = model.compute_feature_fusion_score(images)
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
        acc = metrics.accuracy_score(predicts, targets)
        logging.info("Accuracy@0.5: {:.4f}".format(acc))

        # Save best model
        saver.save(auc, acc, model.state_dict(), epoch)
        logging.info(
            "Best AUC is : {:.4f} Best_auc_epoch is {}".format(saver.best_auc, saver.best_auc_epoch))  # 输出已经选择好的最佳模型
        logging.info(
            "Best ACC is : {:.4f} Best_acc_epoch is {}".format(saver.best_acc, saver.best_acc_epoch))  # 输出已经选择好的最佳模型


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fashion Compatibility Training.')
    parser.add_argument('--comment', help="模型存储的名字（结合utils.BestSaver给出完整的名字）", type=str,
                        default="MLMSFF_layer_size(256)_multi_layer(4)")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--clip', help="训练过程中梯度更新的大小限制", type=int, default=5)
    parser.add_argument('--vse_off', help="是否关闭vse模块，默认不移除vse模块", action="store_true")
    parser.add_argument('--layer_size', help="多层级特征融合模块中，原有特征被映射到的新维度", type=int, default=256)
    parser.add_argument('--multi_layer', help="取值范围是0-4, 在多层级特征融合模块中，数字i表示前i层特征被用，",
                        type=int, default=4)
    args = parser.parse_args()

    print(args)

    # Logger
    config_logging(args.comment)

    # Dataloader
    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        prepare_dataloaders(batch_size=args.batch_size)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CompatModel(vocabulary=len(train_dataset.vocabulary), vse_off=args.vse_off, layer_size=args.layer_size,
                        multi_layer=args.multi_layer)
    train(model, device, train_loader, val_loader, args.comment, args.clip)

    print(args)  # 方便查找存储的模型
