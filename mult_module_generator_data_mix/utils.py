import logging

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset

from polyvore_dataset import CategoryDataset, collate_fn


class AverageMeter(object):
    """Computes and stores the average and current value.

    >>> acc = AverageMeter()
    >>> acc.update(0.6)
    >>> acc.update(0.8)
    >>> print(acc.avg)
    0.7
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val  #* n
        self.count += n
        self.avg = self.sum / self.count


class BestSaver(object):
    """Save pytorch model with best performance. Template for save_path is:

        model_{save_path}_{comment}.pth

    >>> comment = "v1"
    >>> saver = BestSaver(comment)
    >>> auc = 0.6
    >>> saver.save(0.6, model.state_dict())
    """
    def __init__(self, comment=None):
        # Get current executing script name
        import __main__, os
        exe_fname=os.path.basename(__main__.__file__)

        auc_save_path = "model_{}".format(exe_fname.split(".")[0])
        if comment is not None and str(comment):
            auc_save_path = auc_save_path + "_" + str(comment)

        auc_save_path = auc_save_path + ".pth"
        self.auc_save_path = auc_save_path
        self.best_auc = float('-inf')
        self.best_auc_epoch = 0  # 取得最佳成绩的轮次

        # ------------------------ 根据diff_ACC的任务来保存优秀的模型判断 ---------------------------
        diff_acc_save_path = "model_diff_acc_{}".format(exe_fname.split(".")[0])
        if comment is not None and str(comment):
            diff_acc_save_path = diff_acc_save_path + "_" + str(comment)

        diff_acc_save_path = diff_acc_save_path + ".pth"
        self.diff_acc_save_path = diff_acc_save_path
        self.best_diff_acc = float('-inf')
        self.best_diff_acc_epoch = 0  # 取得最佳成绩的轮次

        # ------------------------根据pos_gt_neg_acc的任务来保存优秀的模型判断 ---------------------------
        pos_neg_save_path = "model_pos_neg_acc_{}".format(exe_fname.split(".")[0])
        if comment is not None and str(comment):
            pos_neg_save_path = pos_neg_save_path + "_" + str(comment)

        pos_neg_save_path = pos_neg_save_path + ".pth"
        self.pos_neg_save_path = pos_neg_save_path
        self.best_pos_neg_acc = float('-inf')
        self.best_pos_neg_acc_epoch = 0  # 取得最佳成绩的轮次

        # ------------------------根据mix_acc的任务来保存优秀的模型判断---------------------------
        mix_save_path = "mix_acc_{}".format(exe_fname.split(".")[0])
        if comment is not None and str(comment):
            mix_save_path = mix_save_path + "_" + str(comment)

        mix_save_path = mix_save_path + ".pth"
        self.mix_save_path = mix_save_path
        self.best_mix_acc = float('-inf')
        self.best_mix_acc_epoch = 0  # 取得最佳成绩的轮次

    def save(self, auc, diff_acc, pos_neg_acc, mix_acc, data, epoch):
        if auc > self.best_auc:
            self.best_auc = auc
            self.best_auc_epoch = epoch
            torch.save(data, self.auc_save_path)
            logging.info("Saved best model to {}".format(self.auc_save_path))

        # ------------------------ 根据ACC的结果来保存优秀的模型判断 ---------------------------
        if diff_acc >= self.best_diff_acc:
            self.best_diff_acc = diff_acc
            self.best_diff_acc_epoch = epoch
            torch.save(data, self.diff_acc_save_path)
            logging.info("Saved diff acc best model to {}".format(self.diff_acc_save_path))
            # ------------------------ 根据ACC(AUC)的结果来保存优秀的模型判断 ---------------------------
        if pos_neg_acc >= self.best_pos_neg_acc:
            self.best_pos_neg_acc = pos_neg_acc
            self.best_pos_neg_acc_epoch = epoch
            torch.save(data, self.pos_neg_save_path)
            logging.info("Saved pos_neg_acc best model to {}".format(self.pos_neg_save_path))
        # ------------------------根据ACC(AUC)的结果来保存优秀的模型判断---------------------------
        if mix_acc >= self.best_mix_acc:
            self.best_mix_acc = mix_acc
            self.best_mix_acc_epoch = epoch
            torch.save(data, self.mix_save_path)
            logging.info("Saved mix acc best model to {}".format(self.mix_save_path))


def config_logging(comment=None):
    """Configure logging for training log. The format is 

        `log_{log_fname}_{comment}.log`

    .g. for `train.py`, the log_fname is `log_train.log`.
    Use `logging.info(...)` to record running log.

    Args:
        comment (any): Append comment for log_fname
    """

    # Get current executing script name
    import __main__, os
    exe_fname=os.path.basename(__main__.__file__)
    log_fname = "log_{}".format(exe_fname.split(".")[0])

    if comment is not None and str(comment):
        log_fname = log_fname + "_" + str(comment)

    log_fname = log_fname + ".log"
    log_format = "%(asctime)s [%(levelname)-5.5s] %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_fname), logging.StreamHandler()]
    )

def prepare_dataloaders(root_dir="../data/images2/", data_dir="../data/", batch_size=16, generator_type="upper",
                        img_size=224, use_mean_img=True, neg_samples=True,
                        num_workers=1, collate_fn=collate_fn):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
        ]
    )
    train_dataset = CategoryDataset(
        root_dir=root_dir,
        data_dir=data_dir,
        transform=transform,
        use_mean_img=use_mean_img,
        data_file="train.json",
        neg_samples=neg_samples,
        generator_type=generator_type,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )
    val_dataset = CategoryDataset(
        root_dir=root_dir,
        data_dir=data_dir,
        transform=transform,
        use_mean_img=use_mean_img,
        data_file="valid.json",
        neg_samples=neg_samples,
        generator_type=generator_type,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    test_dataset = CategoryDataset(
        root_dir=root_dir,
        data_dir=data_dir,
        transform=transform,
        use_mean_img=use_mean_img,
        data_file="test.json",
        neg_samples=neg_samples,
        generator_type=generator_type,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader
