import torch
import torch.nn as nn
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders
from model import CompatModel
import shutil
import os
import argparse

def get_compat_result(model, out_dir='compat_show/', img_source_path='../data/images2/'):
    """
    预测结果分成四类： 搭配的正确预测(TP)，搭配的错误预测的(TN)，不搭配的正确预测的(FP)，不搭配的错误预测的(FN)
    """
    criterion = nn.BCELoss()
    model.eval()
    total_loss = AverageMeter()
    for i, batch in enumerate(test_loader, 1):
        print("\r#{}/{}".format(i, len(test_loader)), end="", flush=True)
        lengths, images, names, offsets, set_ids, labels, is_compat = batch
        images = images.to(device)
        target = is_compat.float().to(device)

        img_ids = []  # 一套单品含有4件
        labels = labels[0]
        for label in labels:
            img_ids.append(label.split('_')[-1])

        with torch.no_grad():
            output, _, _ = model.compute_feature_fusion_score(images)
            output = output.squeeze(dim=1)
            loss = criterion(output, target)
        total_loss.update(loss.item(), images.shape[0])

        sub_dir = 'FN/'
        if is_compat.item() == 1 and output >= 0.5:
            sub_dir = 'TP/'
        elif is_compat.item() == 1 and output < 0.5:
            sub_dir = 'TN/'
        elif is_compat.item() == 0 and output < 0.5:
            sub_dir = 'FP/'

        for idx in range(len(labels)):
            target_path = out_dir + sub_dir + str(i)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            target_path = target_path + "/" + img_ids[idx] + '.jpg'
            shutil.copyfile(img_source_path + img_ids[idx] + '.jpg', target_path)

        with open(out_dir + sub_dir + str(i) + '/' + str(output) + '.txt', 'w', encoding='utf-8') as f:
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fashion Compatibility Evaluation.')
    parser.add_argument('--vse_off', action="store_true")
    parser.add_argument('--model_path', type=str, default="./MLMSFF_layer_size(256)_multi_layer(4)_auc.pth")
    parser.add_argument('--layer_size', type=int, default=256)
    parser.add_argument('--multi_layer', type=int, default=4)
    args = parser.parse_args()

    print(args)

    # Dataloader
    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        prepare_dataloaders(batch_size=1)
    )

    # Load pretrained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CompatModel(vocabulary=len(train_dataset.vocabulary), vse_off=args.vse_off, layer_size=args.layer_size).to(
        device)
    model.load_state_dict(torch.load(args.model_path))

    out_dir = 'compat_show/'
    img_source_path = '../data/images2/'

    get_compat_result(model, out_dir=out_dir, img_source_path=img_source_path)

