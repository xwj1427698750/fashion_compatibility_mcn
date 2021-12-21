import torch
from utils import prepare_dataloaders
from model import MultiModuleFashionCompatPrediction
import shutil
import os
import matplotlib.pyplot as plt
import argparse

def get_generatr_show(model, device, gen_type=None):
    """
    在测试集中，展示根据多件单品生成的一件单品的图像, 新建一个目录generator_show，每一个套装存储在一个文件夹里，given开头的图片表示
    输入的3件单品，low_rec.jpg和high_rec.jpg表示生成的图片，high_rec.jpg比low_rec.jpg更清晰，option开头的图片是fitb任务给出的
    选项，target.jpg是目标生成的图片。true.txt表示fitb任务选择正确，false.txt表示选择错误, txt文件内容是正负样本的diff_score得分。
    Args:
        model: 模型
        device: gpu
        gen_type: 生成单品的类型，默认为None, 表示生成的类型是随机的，可以选择的类型是['upper', 'bottom', 'bag', 'shoe']
    """
    type_to_id = {'upper': 0, 'bottom': 1, 'bag': 2, 'shoe': 3}
    gen_id = None  # 不指定生成单品的类型，随机选择类型
    if gen_type is not None:  # 指定生成单品的类型
        gen_id = type_to_id[gen_type]

    model.eval()
    option_len = 4
    out_dir = 'generator_show/'
    img_source_path = '../data/images2/'
    for i, _ in enumerate(test_dataset):
        print("\r#{}/{}".format(i+1, len(test_dataset)), end="", flush=True)
        items, name, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(i, option_len, gen_id)

        # 给定的图像名
        given_img_ids = []  # 含有4件，前3件为给定图像，最后一件是目标单品
        for label in labels:
            given_img_ids.append(label.split('_')[-1])
        option_img_ids = []  # 含有3件， 为候选单品
        for label in option_labels:
            option_img_ids.append(label.split('_')[-1])

        if not os.path.exists(out_dir + str(i)):
            os.makedirs(out_dir + str(i))

        # 将给定查询单品和候选单品放入文件夹中
        for idx in range(len(given_img_ids)):
            if idx < 3:
                target_path = out_dir + str(i) + '/given_' + str(idx) + '.jpg'
            else:
                target_path = out_dir + str(i) + '/target.jpg'
            shutil.copyfile(img_source_path + given_img_ids[idx] + '.jpg', target_path)

        for idx in range(len(option_img_ids)):
            target_path = out_dir + str(i) + '/option_' + str(idx) + '.jpg'
            shutil.copyfile(img_source_path + option_img_ids[idx] + '.jpg', target_path)

        images = []
        for option in options:
            new_outfit = items.clone()
            option = option.unsqueeze(0)
            new_outfit = torch.cat((new_outfit, option), 0)
            images.append(new_outfit)
        images = torch.stack(images).to(device)
        names = torch.stack([name] * (option_len-1)).to(device)

        low_resolution_img, high_resolution_img, diff_score, z_mean, z_log_var, pos_out, neg_out = model(images, names)
        # low_resolution_img.shape torch.Size([3, 3, 128, 128])
        low_resolution_img0 = low_resolution_img[0].permute([1, 2, 0]).cpu().data.numpy()
        plt.imsave(out_dir + str(i) + '/low_rec.jpg', low_resolution_img0)
        high_resolution_img0 = high_resolution_img[0].permute([1, 2, 0]).cpu().data.numpy()
        plt.imsave(out_dir + str(i) + '/high_rec.jpg', high_resolution_img0)

        diff_score_true = (diff_score >= 0)
        diff_score_sum = torch.sum(diff_score_true.float())
        is_true = 'false'
        if int(diff_score_sum) == len(options):
            is_true = 'true'
        with open(out_dir + str(i) + '/' + is_true + '.txt', 'w', encoding='utf-8') as f:
            for score in diff_score:
                f.write(str(score.cpu().data.item()))
                f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMFCP generator show.')
    parser.add_argument('--model_path', type=str,
                        default="./MMFCP_atten_head(4)_feature_size(96)_input_off(F)_generator_off(F)_mlmsff_off(F)_acc.pth")
    parser.add_argument('--attention_heads', type=int, default=4)
    parser.add_argument('--feature_size', type=int, default=96)
    parser.add_argument('--enc_desc_off', type=bool, default=True)
    parser.add_argument('--input_off', type=bool, default=False)
    parser.add_argument('--generator_off', type=bool, default=False)
    parser.add_argument('--MLMSFF_off', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        prepare_dataloaders()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModuleFashionCompatPrediction(vocab_len=len(train_dataset.vocabulary), attention_heads=args.attention_heads,
                                               feature_size=args.feature_size, device=device, enc_desc_off=args.enc_desc_off,
                                               input_off=args.input_off, generator_off=args.generator_off).to(device)

    model.load_state_dict(torch.load(args.model_path))

    get_generatr_show(model, device)




