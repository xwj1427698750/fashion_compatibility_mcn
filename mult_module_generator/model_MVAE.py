import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from model import CompatModel

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    """
    反卷积的输出的张量尺寸计算方式
    img_size = [H, W]
    """
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape


# 隐变量的特征融合模块
class LatentFeatureFuse(nn.Module):
    def __init__(self, drop=0.5):
        super(LatentFeatureFuse, self).__init__()
        # Global average pooling layer
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        layer_lens = [256, 512, 1024, 2048]
        self.layer_fcs = nn.ModuleList()
        for i in range(len(layer_lens)):
            net = nn.Sequential(
                nn.Linear(layer_lens[i] * 3, layer_lens[i]),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(layer_lens[i], 256),
                nn.ReLU(),
            )
            self.layer_fcs.append(net)
    def forward(self, reps_pos, generator_id, outfit_num=4):
        """
        reps_pos是一个长度为4的列表，shape是[4, batch_size*outfit_num, 256, 56, 56]
        """
        batch_size, item_num, _, _, _ = reps_pos[0].shape  # 获取每一层级特征的尺寸信息
        input_features = []
        ground_truth_features = []
        for i in range(outfit_num):
            # 将generator_id对应的张量与最后一件单品的张量进行交换，保证生成模型输入是连续在一起的
            # 下面是in_place操作，反向传播会出错
            # generator_item = reps_pos[i][:, generator_id, :, :, :]
            # swap_item_id = 3
            # swap_item = reps_pos[i][:, swap_item_id, :, :, :]
            # reps_pos[i][:, generator_id, :, :, :] = swap_item
            # reps_pos[i][:, swap_item_id, :, :, :] = generator_item
            rep_li = reps_pos[i]  # (batch_size*outfit_num, 256, 56, 56)
            input_feature = torch.cat((rep_li[:, :generator_id, :, :, :], rep_li[:, generator_id+1:, :, :, :]), 1)
            input_features.append(input_feature)
            ground_truth_features.append(rep_li[:, generator_id, :, :, :])

        # 多尺度融合模块
        out_features = []  # 最后结果是[4, 16, 256]
        for i, rep_li in enumerate(input_features):
            shape = rep_li.shape
            rep_li = rep_li.reshape(batch_size*(outfit_num - 1), shape[-3], shape[-2], shape[-1])
            rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, -1)  # 这里是融合3件，也就是 outfit_num - 1
            # rep_l1 (batch_size, 3*256), rep_l2 (batch_size,3*512), rep_l3 (batch_size,3*1024), rep_l4 (batch_size,3*2048)
            rep_out = self.layer_fcs[i](rep_li)  # 输出都是 (batch,256)
            out_features.append(rep_out)
        fuse_feature = torch.cat(out_features, 1)  # (batch_size, 256*4)
        return fuse_feature, out_features


#  高斯变化
class Transformer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Transformer, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, fuse_feature, device):
        """
        fuse_feature :[batch_size, input_size(256]
        """
        z_mean = self.fc1(fuse_feature)
        z_log_var = self.fc2(fuse_feature)
        epsilon = torch.randn(size=z_mean.shape).to(device)
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon
        return z, z_mean, z_log_var


class SRResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(SRResNetBlock, self).__init__()
        self.same_shape = same_shape
        stride = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return out+x


# 生成器模块
class Generator(nn.Module):
    def __init__(self, input_size, feature_output_size):
        super(Generator, self).__init__()
        self.seq = nn.Sequential(  # 输入(batch_size, input_size), 输出(batch_size, 1024)
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.deconv1 = nn.Sequential(
            # 输入(batch_size, 1024, 1, 1), 输出(batch_size, 512, 4, 4) , 计算方式参见 convtrans2D_output_size
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 输入(batch_size, 512, 4, 4), 输出(batch_size, 512, 8, 8)
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # 输出张量尺寸[batch_size, feature_output_size]
        self.get_feature1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (batch_size, 512, 1, 1)
            nn.Flatten(start_dim=1),
            nn.Linear(512, feature_output_size),
            nn.Sigmoid()
        )

        self.deconv2 = nn.Sequential(
            # 输入(batch_size, 512, 8, 8), 输出(batch_size, 256, 14, 14)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 输入(batch_size, 256, 14, 14), 输出(batch_size, 128, 28, 28)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 输入(batch_size, 128, 28, 28), 输出张量尺寸[batch_size, feature_output_size]
        self.get_feature2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),  # (batch_size, 128, 14, 14)
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1),  # (batch_size, 16, 14, 14)
            nn.Flatten(start_dim=1),
            nn.Linear(16*14*14, feature_output_size),
            nn.Sigmoid()
        )
        self.deconv3 = nn.Sequential(
            # 输入(batch_size, 128, 28, 28), 输出(batch_size, 64, 56, 56)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 输入(batch_size, 64, 56, 56), 输出(batch_size, 32, 112, 112)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 输入(batch_size, 32, 112, 112)， 输出张量尺寸[batch_size, 256]
        self.get_feature3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),  # (batch_size, 32, 56, 56)
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),  # (batch_size, 1, 56, 56)
            nn.Flatten(start_dim=1),
            nn.Linear(56*56, feature_output_size),
            nn.Sigmoid()
        )
        self.deconv4 = nn.Sequential(
            # 输入(batch_size, 32, 112, 112), 输出(batch_size, 16, 224, 224)
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 输入(batch_size, 16, 224, 224), 输出(batch_size, 3, 224, 224)
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
        )
        # 输入(batch_size, 3, 224, 224),输出张量尺寸[batch_size, 256]
        self.get_feature4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),  # (batch_size, 3, 56, 56)
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1),  # (batch_size, 1, 56, 56)
            nn.Flatten(start_dim=1),
            nn.Linear(56*56, feature_output_size),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        # 输入是(batch_size, 3, 224, 224), 输出是(batch_size, 32, 224, 224)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 输入与输出都是(batch_size, 32, 224, 224)
        self.conv2 = nn.Sequential(
            SRResNetBlock(32, 32),
            SRResNetBlock(32, 32),
            SRResNetBlock(32, 32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 输入是(batch_size, 32, 224, 224), 输出是(batch_size, 3, 224, 224)
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, fuse_feature):

        dec = self.seq(fuse_feature)  # [batch_size,1024]
        dec = torch.reshape(dec, (dec.shape[0], dec.shape[1], 1, 1))

        dec = self.deconv1(dec)
        feature1 = self.get_feature1(dec)
        dec = self.deconv2(dec)
        feature2 = self.get_feature2(dec)
        dec = self.deconv3(dec)
        feature3 = self.get_feature3(dec)
        low_resolution_img = self.deconv4(dec)  # 低分辨率
        feature4 = self.get_feature4(low_resolution_img)

        low_to_mid = self.conv1(low_resolution_img)
        mid_to_high = self.conv2(low_to_mid)
        high_resolution_img = self.conv3(low_to_mid + mid_to_high)  # 高分辨率
        return low_resolution_img, high_resolution_img, [feature1, feature2, feature3, feature4]  # features : [4, batch_size, 256]


class LayerOut(nn.Module):
    """
    输入单件单品在resnet模型输出的4层张量
    输出单件单品在resnet模型输出的4层张量，尺寸是[batch_size, feature_output_size]
    """
    def __init__(self, feature_output_size, drop=0.5):
        super(LayerOut, self).__init__()
        # 输入[batch_size,256,56,56], 输出[batch_size, feature_output_size]
        self.get_feature1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(256, feature_output_size),
            nn.Dropout(drop),
            nn.Sigmoid()
        )
        # [batch_size, 512, 28, 28], 输出[batch_size, feature_output_size]
        self.get_feature2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(512, feature_output_size),
            nn.Dropout(drop),
            nn.Sigmoid()
        )
        # [batch_size, 1024, 14, 14], 输出[batch_size, feature_output_size]
        self.get_feature3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(1024, feature_output_size),
            nn.Dropout(drop),
            nn.Sigmoid()
        )
        # [batch_size, 2048, 7, 7], 输出[batch_size, feature_output_size]
        self.get_feature4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, feature_output_size),
            nn.Dropout(drop),
            nn.Sigmoid()
        )

    def forward(self, layer_reps):
        """
            layer_reps : [batch_size,1,256,56,56],[batch_size,1,512,28,28],[batch_size,1,1024,14,14],[batch_size,1, 2048, 7, 7]
        """
        batch_size, *other = layer_reps[0].shape
        for i in range(len(layer_reps)):
            shape = layer_reps[i].shape
            layer_reps[i] = torch.reshape(layer_reps[i], (batch_size, shape[-3], shape[-2], shape[-1]))
        feature1 = self.get_feature1(layer_reps[0])
        feature2 = self.get_feature2(layer_reps[1])
        feature3 = self.get_feature3(layer_reps[2])
        feature4 = self.get_feature4(layer_reps[3])
        return [feature1, feature2, feature3, feature4]


class MultiModuleGenerator(nn.Module):
    def __init__(self, embed_size=1000, need_rep=True, vocabulary=None,
                 vse_off=False, pe_off=False, mlp_layers=2, conv_feats="1234", encoder_path="model_mcn.pth", generator_type="upper", device=torch.device("cuda:0")):
        """ The Multi-Module-Generator for fashion item generator.
        Args:
            embed_size: the output embedding size of the cnn model, default 1000.
            need_rep: whether to output representation of the layer before last fc
                layer, whose size is 2048. This representation can be used for
                compute the Visual Sementic Embedding (VSE) loss.
            vocabulary: the counts of words in the polyvore dataset.
            vse_off: whether use visual semantic embedding.
            pe_off: whether use projected embedding.
            mlp_layers: number of mlp layers used in the last predictor part.
            conv_feats: decide which layer of conv features are used for comparision.
        """
        super(MultiModuleGenerator, self).__init__()
        self.type_to_id = {'upper': 0, 'bottom': 1, 'bag': 2, 'shoe': 3}
        self.generator_type = generator_type
        self.device = device

        self.encoder = CompatModel(embed_size=embed_size, need_rep=need_rep, vocabulary=vocabulary, vse_off=vse_off,
                                   pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats)
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.latent_feature_fuse = LatentFeatureFuse()
        self.transformer = Transformer(input_size=256*4, output_size=256)
        self.generator = Generator(input_size=256*4+256, feature_output_size=256)
        self.get_layer_out = LayerOut(feature_output_size=256)
        to_init_modules = [self.latent_feature_fuse, self.latent_feature_fuse, self.transformer, self.generator,
                           self.get_layer_out]
        for net in to_init_modules:
            for m in net.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    # 返回一个列表，包含generator_id对应单品的每一层特征
    def get_layer_features(self, reps, generator_id):
        """
        reps_neg: 含有套装中generator_id位置对应的单品的层级特征， 一共有4层， reps前4个的特征维度 [batch_size,4,256,56,56],[batch_size,4,512,28,28],[batch_size,4,1024,14,14],[batch_size,4, 2048, 7, 7]
        generator_id,
        return 返回一个列表，包含generator_id对应单品的每一层特征  [batch_size,1,256,56,56],[batch_size,1,512,28,28],[batch_size,1,1024,14,14],[batch_size,1, 2048, 7, 7]
        """
        batch_size, outfit_num, _, _, _ = reps[0].shape  # 获取每一层级特征的尺寸信息
        layer_reps = []
        for i in range(4):  # 一共有4层
            layer_reps.append(reps[i][:, generator_id, :, :, :])
        return layer_reps

    # 层级特征交互得分计算
    def get_layer_feature_score(self, input_fuse_features, generator_layer_features, pos_layer_features,
                                neg_layer_features):
        """
        pos_layer_features与input_fuse_features, generator_layer_features的交互得分大于(>)
        neg_layer_features与input_fuse_features, generator_layer_features
        每一个输入的维度是[4, batch_size, feature_output_size(256)]
        """
        batch_size = input_fuse_features[0].shape[0]
        score_pos = torch.zeros(size=(batch_size, 1)).to(self.device)
        score_neg = torch.zeros(size=(batch_size, 1)).to(self.device)
        for layer_index in range(len(input_fuse_features)):
            score_pos = score_pos + torch.sum(input_fuse_features[layer_index] * pos_layer_features[layer_index], dim=1, keepdim=True)  # [batch_size, 1]
            score_pos = score_pos + torch.sum(generator_layer_features[layer_index] * pos_layer_features[layer_index], dim=1, keepdim=True)  # [batch_size, 1]

            score_neg = score_neg + torch.sum(input_fuse_features[layer_index] * neg_layer_features[layer_index], dim=1, keepdim=True)  # [batch_size, 1]
            score_neg = score_neg + torch.sum(generator_layer_features[layer_index] * neg_layer_features[layer_index], dim=1, keepdim=True)  # [batch_size, 1]

        difference_score = score_pos - score_neg
        return difference_score  # (batch_size, 1)

    def forward(self, images, names):
        """
        Args:
            images: Outfit images with shape (N, T, C, H, W)
            names: Description words of each item in outfit

        Return:
            out: Compatibility score
            vse_loss: Visual Semantic Loss
            tmasks_loss: mask loss to encourage a sparse mask
            features_loss: regularize the feature vector to be normal
        """
        batch_size, item_num, _, _, img_size = images.shape  # (8, 5, 3, 224, 224) item_num = 5,有4个正的,1个负的

        outfit_pos = images[:, :4, :, :, :]  # 正样本 (8, 4, 3, 224, 224)

        image_neg = images[:, 4:5, :, :, :]  # 负样本 (8, 1, 3, 224, 224)
        generator_id = self.type_to_id[self.generator_type]
        outfit_neg = torch.cat((images[:, :generator_id, :, :, :], image_neg, images[:, generator_id+1: 4, :, :, :]), dim=1)

        # 获得正负套装搭配的概率
        # reps_pos包含5层的特征，rep_l1, rep_l2, rep_l3, rep_l4, rep_last_2th
        # [batch_size*outfit_num,256,56,56],[batch_size*outfit_num,512,28,28],[batch_size*outfit_num,1024,14,14],[batch_size*outfit_num, 2048, 7, 7]，_, 我们用到了前面4个
        out_pos, features_pos, _, _, reps_pos = self.encoder._compute_feature_fusion_score(outfit_pos)
        out_neg, features_neg, _, _, reps_neg = self.encoder._compute_feature_fusion_score(outfit_neg)

        for i in range(4):
            shape = reps_pos[i].shape
            reps_pos[i] = torch.reshape(reps_pos[i], (batch_size, item_num-1, -1, shape[-2], shape[-1]))
            reps_neg[i] = torch.reshape(reps_neg[i], (batch_size, item_num-1, -1, shape[-2], shape[-1]))

        # 输入图片特征融合模块
        # reps_pos是一个长度2为5的列表，shape是[5, batch_size*outfit_num, 256, 56, 56],但是只使用其中的前4个
        # input_fuse_features, 尺寸是# (batch_size, 256*4) out_features是tensor的列表, 尺寸是 [4, batch_size, 256]
        input_fuse_features, out_features = self.latent_feature_fuse(reps_pos[0:4], generator_id)

        z, z_mean, z_log_var = self.transformer(input_fuse_features, self.device)

        # 生成模块
        generator_input = torch.cat((z, input_fuse_features), 1)  # generator_input : [16, 256 + 1024]
        low_resolution_img, high_resolution_img, generator_layer_features = self.generator(generator_input)
        # [batch_size, 3, 224,224], [batch_size, 3,224,224], generator_layer_features: [4, batch_size, 256]

        # 获得正负单品的多层级特征表示
        neg_layer_reps = self.get_layer_features(reps_neg, generator_id)
        pos_layer_reps = self.get_layer_features(reps_pos, generator_id)

        neg_layer_out = self.get_layer_out(neg_layer_reps)
        pos_layer_out = self.get_layer_out(pos_layer_reps)

        # 正样本与负样本的层级特征交互得分差 difference_score (batch_size, 1)
        difference_score = self.get_layer_feature_score(out_features, generator_layer_features, pos_layer_out, neg_layer_out)

        return out_pos, out_neg, low_resolution_img, high_resolution_img, difference_score, z_mean, z_log_var


if __name__ == "__main__":

    device = torch.device("cpu:0")
    model = CompatModel(embed_size=1000, vocabulary=1000, need_rep=True).to(device)
    images = torch.ones([16, 4, 3, 224, 224])
    names = [torch.ones([5]) for _ in range(80)]
    output, vse_loss, tmasks_loss, features_loss = model(images, names)
