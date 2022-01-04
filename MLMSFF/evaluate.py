import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders
from model import CompatModel
import argparse


def test(model, device, test_loader, test_dataset, test_num=10):
    criterion = nn.BCELoss()
    model.eval()

    def test_auc():
        # Compatibility AUC test
        auc_epochs = []
        for epoch in range(test_num):
            total_loss = AverageMeter()
            outputs = []
            targets = []
            for batch_num, batch in enumerate(test_loader, 1):
                print("\r#{}/{}".format(batch_num, len(test_loader)), end="", flush=True)
                images, names, _, _, is_compat = batch
                images = images.to(device)
                target = is_compat.float().to(device)
                with torch.no_grad():
                    output, _, _ = model.compute_feature_fusion_score(images)
                    output = output.squeeze(dim=1)
                    loss = criterion(output, target)
                total_loss.update(loss.item(), images.shape[0])
                outputs.append(output)
                targets.append(target)
            print()
            print("test:{} Test Loss: {:.4f}".format(epoch + 1, total_loss.avg))
            outputs = torch.cat(outputs).cpu().data.numpy()
            targets = torch.cat(targets).cpu().data.numpy()
            auc = metrics.roc_auc_score(targets, outputs)
            print("test:{} AUC: {:.4f}".format(epoch + 1, auc))
            auc_epochs.append(auc)
        auc_epochs = np.array(auc_epochs)
        auc_mean = (np.sum(auc_epochs) - np.max(auc_epochs) - np.min(auc_epochs)) / (test_num - 2)  # 去除最大值和最小值，剩下的取平均
        print("AUC mean is {:.4f}".format(auc_mean))

    def test_fitb():
        # Fill in the blank evaluation
        for option_len in [4, 5, 6]:  # 选项分别为4,5,6的fitb任务
            print("option_len=", option_len)
            is_correct = []
            fitb_epochs = []
            for epoch in range(test_num):
                for i in range(len(test_dataset)):
                    print("\r#{}/{}".format(i, len(test_dataset)), end="", flush=True)
                    items, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(
                        i, option_len)
                    question_part = {"upper": 0, "bottom": 1, "shoe": 2, "bag": 3}.get(question_part)
                    images = [items]

                    for option in options:
                        new_outfit = items.clone()
                        new_outfit[question_part] = option
                        images.append(new_outfit)
                    images = torch.stack(images).to(device)
                    output, _, _ = model.compute_feature_fusion_score(images)

                    if output.argmax().item() == 0:
                        is_correct.append(True)
                    else:
                        is_correct.append(False)
                print()
                fitb_acc = sum(is_correct) / len(is_correct)
                print("test:{} FitB ACC: {:.4f}".format(epoch + 1, fitb_acc))
                fitb_epochs.append(fitb_acc)
            fitb_epochs = np.array(fitb_epochs)
            fitb_mean = (np.sum(fitb_epochs) - np.max(fitb_epochs) - np.min(fitb_epochs)) / (test_num - 2)
            print(f"average fitb acc is {fitb_mean}")

    test_auc()
    test_fitb()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fashion Compatibility Evaluation.')
    parser.add_argument('--vse_off', action="store_true")
    parser.add_argument('--model_path', type=str, default="./MLMMSFF_combination_auc.pth")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--layer_size', type=int, default=256)
    parser.add_argument('--multi_layer', type=int, default=4)
    args = parser.parse_args()

    print(args)

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        prepare_dataloaders(batch_size=args.batch_size)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CompatModel(vocabulary=len(train_dataset.vocabulary), vse_off=args.vse_off, layer_size=args.layer_size,
                        multi_layer=args.multi_layer).to(device)
    model.load_state_dict(torch.load(args.model_path))

    test(model, device, test_loader, test_dataset, test_num=10)
