import logging
import torch
import torch.nn as nn
from sklearn import metrics
from model import MultiModuleFashionCompatPrediction
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders
import argparse

def train(model, device, train_loader, val_loader, comment, clip, epochs=100, MLMSFF_off=False):
    generator_id = 3  # 每一个组合都是随机选择生成类型，然后通过调换位置，使得第4件为目标正单品（下标为3）
    model = model.to(device)
    criterion = nn.BCELoss()
    get_l2_loss = nn.MSELoss()
    get_l1_loss = nn.L1Loss()
    log_sigmoid = nn.LogSigmoid()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    saver = BestSaver(comment)

    for epoch in range(1, epochs + 1):
        logging.info("Train Phase, Epoch: {}".format(epoch))

        total_losses = AverageMeter()
        clf_losses = AverageMeter()
        bpr_losses = AverageMeter()
        l2_losses = AverageMeter()
        l1_losses = AverageMeter()
        kl_losses = AverageMeter()

        model.train()
        for batch_num, batch in enumerate(train_loader, 1):
            images, names, _, _, _ = batch
            # images[batch_size, 5, 3, 128, 128]  names [batch_size, 1245]
            # 前3件为给定的输入单品，第四件为目标单品(正样本)，第5件为负样本
            batch_size, _, _, _, img_size = images.shape
            images = images.to(device)
            names = names.to(device)
            generator_target_img = images[:, generator_id, :, :, :]  # 正样本单品图像，也是生成图像的ground truth

            # Forward   前向训练只需要图像, 文本数据(可按照需要添加，默认没有使用)
            low_resolution_img, high_resolution_img, diff_score, z_mean, z_log_var, pos_out, neg_out = model(images,
                                                                                                             names)
            # low/high_resolution_img (batch_size, 3, 128, 128) diff_score:(batch_size,), z_mean/log_var:(batch_size, feature_size)
            # pos/neg_out:(batch_size, 1)
            # BCE Loss
            clf_loss = torch.tensor(0)
            if not MLMSFF_off:
                clf_loss_param = 1
                pos_target = torch.ones(batch_size)
                pos_out = pos_out.squeeze(dim=1)

                neg_target = torch.zeros(batch_size)
                neg_out = neg_out.squeeze(dim=1)

                output = torch.cat((pos_out, neg_out))
                target = torch.cat((pos_target, neg_target)).to(device)
                clf_loss = clf_loss_param * criterion(output, target)

            # BPR LOSS
            bpr_param = 1
            bpr_loss = bpr_param * torch.sum(-log_sigmoid(diff_score))

            # Generator LOSS
            l1_loss_param = 100
            l2_loss_param = 100

            l1_loss = l1_loss_param * get_l1_loss(high_resolution_img, generator_target_img)
            l2_loss = l2_loss_param * get_l2_loss(low_resolution_img, generator_target_img)

            # KL LOSS
            kl_loss = torch.sum(-0.5 * torch.sum(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)))

            # Sum all losses up
            total_loss = clf_loss + bpr_loss + l2_loss + l1_loss + kl_loss

            # Update Recoder
            clf_losses.update(clf_loss.item(), images.shape[0])
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
                    "[{}/{}] #{} clf_loss:{:.4f}, bpr_loss: {:.4f}, l2_loss: {:.4f}, l1_loss: {:.4f}, kl_loss: {:.4f}, total_loss:{:.4f}".format(
                        epoch, epochs, batch_num, clf_losses.avg, bpr_losses.avg, l2_losses.avg, l1_losses.avg,
                        kl_losses.avg, total_losses.avg
                    )
                )
        logging.info("Train Loss (total_loss): {:.4f}".format(total_losses.avg))

        logging.info("Valid Phase, Epoch: {}".format(epoch))
        model.eval()
        clf_losses = AverageMeter()
        acc_avg = AverageMeter()

        outputs = []
        targets = []
        for batch_num, batch in enumerate(val_loader, 1):
            images, names, _, _, _ = batch
            images = images.to(device)
            names = names.to(device)
            batch_size, _, _, _, img_size = images.shape  # 在batch_size = 8 的情况下，最后一个batch的大小是4，多以在这里加一句，在最后一个batch的时候，更改下batch_size的大小
            pos_target = torch.ones(size=[batch_size])
            neg_target = torch.zeros(size=[batch_size])

            with torch.no_grad():
                low_resolution_img, high_resolution_img, diff_score, z_mean, z_log_var, pos_out, neg_out = model(
                    images, names)
            # out_pos:(batch_size, 1)  low_resolution_img:(batch_size, 3, 224, 224)   difference_score: (batch_size, 1)
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
            difference_score_true = (diff_score >= 0)
            difference_score_sum = torch.sum(difference_score_true.float())
            diff_acc = difference_score_sum / diff_score.shape[0]
            acc_avg.update(diff_acc.item(), 1)

        logging.info("Valid Loss:")
        logging.info("acc avg: {:.4f}".format(acc_avg.avg))

        if not MLMSFF_off:
            logging.info("clf_loss: {:.4f}".format(clf_losses.avg))
            outputs = torch.cat(outputs).cpu().data.numpy()
            targets = torch.cat(targets).cpu().data.numpy()
            auc = metrics.roc_auc_score(targets, outputs)
            logging.info("AUC: {:.4f}".format(auc))

        saver.save(acc=acc_avg.avg, data=model.state_dict(), epoch=epoch)
        logging.info("Best acc is : {:.4f} Best_epoch is {}".format(saver.best_acc, saver.best_acc_epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMFCP Training.')
    parser.add_argument('--comment', type=str,
                        default="MMFCP_atten_head(4)_feature_size(96)_input_off(F)_generator_off(F)_mlmsff_off(F)")
    parser.add_argument('--clip', type=int, default=5)
    parser.add_argument('--attention_heads', help="attention_heads表示自注意力机制的头数，0表示没有使用自注意力机制", type=int, default=4)
    parser.add_argument('--feature_size', help="层级特征的维度", type=int, default=96)
    parser.add_argument('--enc_desc_off', help="移除多层级特征交互模块中正负样本与给定的正样本的的文本特征交互计算，默认移除", type=bool, default=True)
    parser.add_argument('--input_off', help="移除多层级特征交互模块中正负样本与给定的多件单品之间的特征交互计算，默认不移除", type=bool, default=False)
    parser.add_argument('--generator_off', help="移除多层级特征交互模块中正负样本与生成单品之间的特征交互计算，默认不移除", type=bool, default=False)
    parser.add_argument('--MLMSFF_off', help="移除MLMSFF模块，默认不移除", type=bool, default=False)
    parser.add_argument('--epochs', help="训练的轮数", type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gen_type', help="生成单品的类型是随机的还是固定的，None为随机类型，可选的类型有['upper', 'bottom', 'bag', 'shoe']",
                        type=str, default=None)
    args = parser.parse_args()
    print(args)

    # Logger
    config_logging(args.comment)

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
                                               input_off=args.input_off, generator_off=args.generator_off)
    train(model, device, train_loader, val_loader, args.comment, clip=args.clip, epochs=args.epochs,
          MLMSFF_off=args.MLMSFF_off)

    print(args)  # 方便查看保存的模型
