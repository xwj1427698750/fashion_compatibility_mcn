import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
from utils import AverageMeter, prepare_dataloaders
from model import MultiModuleFashionCompatPrediction
import argparse

def test(model, device, MLMSFF_off, test_num):
    model.eval()

    def test_acc():
        criterion = nn.BCELoss()
        acc_epochs = []
        for epoch in range(test_num):
            clf_losses = AverageMeter()
            acc_avg = AverageMeter()
            outputs = []
            targets = []
            for batch_num, batch in enumerate(test_loader, 1):
                print("\r#{}/{} ".format(batch_num, len(test_loader)), end="", flush=True)
                images, names, _, _, _ = batch
                images = images.to(device)
                names = names.to(device)
                batch_size, _, _, _, img_size = images.shape
                pos_target = torch.ones(size=[batch_size])
                neg_target = torch.zeros(size=[batch_size])

                with torch.no_grad():
                    _, _, diff_score, _, _, pos_out, neg_out = model(images, names)
                # pos_out,neg_out :(batch_size, 1)   diff_score :(batch_size,)
                if not MLMSFF_off:

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
                diff_score = diff_score
                diff_score_true = (diff_score >= 0)
                diff_score_sum = torch.sum(diff_score_true.float())
                diff_acc = diff_score_sum / diff_score.shape[0]
                acc_avg.update(diff_acc.item(), 1)
            print("\nTest Loss acc_avg: {:.4f}".format(acc_avg.avg))

            if not MLMSFF_off:
                outputs = torch.cat(outputs).cpu().data.numpy()
                targets = torch.cat(targets).cpu().data.numpy()
                auc = metrics.roc_auc_score(targets, outputs)
                print("test:{} AUC: {:.4f}".format(epoch + 1, auc))
            acc_epochs.append(acc_avg.avg)

        acc_epochs = np.array(acc_epochs)
        acc_mean = (np.sum(acc_epochs) - np.max(acc_epochs) - np.min(acc_epochs)) / (test_num - 2)
        print(f"average acc is {acc_mean}")

    def test_fitb():
        for option_len in [4, 5, 6]:  # fitb任务的不同选项
            print("option_len=", option_len)
            fitb_epochs = []
            for epoch in range(test_num):
                is_correct = []
                for i, _ in enumerate(test_dataset):
                    print("\r#{}/{}".format(i, len(test_dataset)), end="", flush=True)
                    items, name, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(
                        i, option_len)
                    images = []
                    for option in options:
                        new_outfit = items.clone()
                        option = option.unsqueeze(0)
                        new_outfit = torch.cat((new_outfit, option), 0)
                        images.append(new_outfit)
                    images = torch.stack(images).to(device)
                    names = torch.stack([name] * (option_len - 1)).to(device)

                    _, _, diff_score, _, _, pos_out, neg_out = model(images, names)

                    diff_score_true = (diff_score >= 0)
                    diff_score_sum = torch.sum(diff_score_true.float())

                    if int(diff_score_sum) == len(options):
                        is_correct.append(True)
                    else:
                        is_correct.append(False)
                    del items, options, images

                fitb_acc = sum(is_correct) / len(is_correct)
                fitb_epochs.append(fitb_acc)
                print("\ntest:{} FitB ACC: {:.4f}".format(epoch + 1, fitb_acc))

            fitb_epochs = np.array(fitb_epochs)
            fitb_mean = (np.sum(fitb_epochs) - np.max(fitb_epochs) - np.min(fitb_epochs)) / (test_num - 2)
            print(f"average fitb acc is {fitb_mean}")

    test_acc()
    test_fitb()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMFCP evaluate.')
    parser.add_argument('--model_path', type=str,
                        default="./MMFCP_atten_head(4)_feature_size(96)_input_off(F)_generator_off(F)_mlmsff_off(F)_acc.pth")
    parser.add_argument('--attention_heads', help="attention_heads表示自注意力机制的头数，0表示没有使用自注意力机制", type=int, default=4)
    parser.add_argument('--feature_size', help="层级特征的维度", type=int, default=96)
    parser.add_argument('--enc_desc_off', help="移除多层级特征交互模块中正负样本与给定的正样本的的文本特征交互计算，默认移除", action="store_false")
    parser.add_argument('--input_off', help="移除多层级特征交互模块中正负样本与给定的多件单品之间的特征交互计算，默认不移除", action="store_true")
    parser.add_argument('--generator_off', help="移除多层级特征交互模块中正负样本与生成单品之间的特征交互计算，默认不移除", action="store_true")
    parser.add_argument('--MLMSFF_off', help="移除MLMSFF模块，默认不移除", action="store_true")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gen_type', help="生成单品的类型是随机的还是固定的，None为随机类型，可选的类型有['upper', 'bottom', 'bag', 'shoe']",
                        type=str, default=None)
    parser.add_argument('--test_num', help="测试的次数:默认测试10次，然后去除最大值和最小值，取剩余的平均值", type=int, default=10)
    args = parser.parse_args()
    print(args)

    # 设置生成单品的类型，随机类型还是指定类型 # 只能选择type_to_id中的四类
    type_to_id = {'upper': 0, 'bottom': 1, 'bag': 2, 'shoe': 3}
    change_id = None  # 不指定生成单品的类型，随机选择类型
    if args.gen_type is not None:  # 指定生成单品的类型
        change_id = type_to_id[args.gen_type]

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        prepare_dataloaders(batch_size=args.batch_size, change_id=change_id)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModuleFashionCompatPrediction(vocab_len=len(train_dataset.vocabulary),
                                               attention_heads=args.attention_heads,
                                               feature_size=args.feature_size, device=device,
                                               enc_desc_off=args.enc_desc_off,
                                               input_off=args.input_off, generator_off=args.generator_off).to(device)
    model.load_state_dict(torch.load(args.model_path))

    test(model, device, args.MLMSFF_off, args.test_num)
