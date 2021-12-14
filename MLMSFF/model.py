import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from resnet import resnet50

class CompatModel(nn.Module):
    def __init__(self, embed_size=1024, vocabulary=None, vse_off=False, layer_feature_size=64):
        """The Multi-Layered Comparison Network (MCN) for outfit compatibility prediction and diagnosis.
        Args:
            embed_size: the output embedding size of the cnn model, default 1000.
            vocabulary: the counts of words in the polyvore dataset.
            vse_off: whether use visual semantic embedding.
            layer_feature_size： 多层级特征融合模块，融合后的每一层级的特征维度
        """
        super(CompatModel, self).__init__()
        self.vse_off = vse_off
        self.embedding_size = embed_size

        cnn = resnet50(pretrained=True, need_rep=True)
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)  # 更换原先resnet的最后一层
        self.cnn = cnn

        # Semantic embedding model
        self.sem_embedding = nn.Embedding(vocabulary, embed_size)
        # Visual embedding model
        self.image_embedding = nn.Linear(2048, embed_size)
        # Global average pooling layer
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.sigmoid = nn.Sigmoid()
        # 多尺度融合层每一层的卷积核
        self.filter_sizes = [2, 3, 4]
        rep_weight = 9  # 套装的个数，正常是4，如果将套装复制然后拼接了一次，就可以得到8了 ,又加了第一件，为了能够获得所有单品的成对的组合，以及三件的组合

        self.layer_convs_1 = nn.ModuleList()  # 4 x 3 , 一共有4层, 每一层有3个卷积核
        self.layer_convs_2 = nn.ModuleList()  # 4 x 3 , 一共有4层, 每一层有3个卷积核
        for i in range(4):
            multi_convs1 = nn.ModuleList()
            multi_convs2 = nn.ModuleList()
            for size in self.filter_sizes:
                # 维度为0上的kernel尺寸是rep_weight - size + 1， 希望将单个分支卷积之后的特征池化为1
                # 维度为1上的kernel尺寸是rep_weight//2 - size + 2， 这个尺寸的设置目的是希望不同size大小的分支经过池化后的尺寸接近
                pool_kernel_size = (rep_weight - size + 1, rep_weight // 2 - size + 2)
                conv_net1 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size, stride=(1, size)),
                    nn.BatchNorm2d(1),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=pool_kernel_size),
                    nn.Flatten(),
                )
                multi_convs1.append(conv_net1)

                conv_net2 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size, stride=(1, size)),
                    nn.BatchNorm2d(1),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=pool_kernel_size),
                    nn.Flatten(),
                )
                multi_convs2.append(conv_net2)

            self.layer_convs_1.append(multi_convs1)
            self.layer_convs_2.append(multi_convs2)

        self.layer_convs_fcs = nn.ModuleList()
        fashion_item_rep_len = [0, 256, 512, 1024, 2048]  # 从resnet抽取的4层特征的维度（0是为了代码编写方便添加的）
        fcs_output_size = [0, layer_feature_size, layer_feature_size, layer_feature_size, layer_feature_size]

        for i in range(1, len(fashion_item_rep_len)):
            rep_len = fashion_item_rep_len[i]
            input_size = 0
            for size in self.filter_sizes:
                stride = size
                # 卷积操作后张量的尺寸计算公式: (W-F) // S + 1
                wi = (rep_len - size) // stride + 1
                hi = rep_weight - size + 1
                # 池化操作后张量的尺寸计算公式: (W-F) // S + 1
                pool_stride = rep_weight // 2 - size + 2
                wi = wi // pool_stride  # 由 (wi - pool_stride) // pool_stride + 1 简化而来
                hi = 1                  # 由 hi = (hi - hi) // hi + 1 简化得来
                input_size = input_size + hi * wi
            input_size = input_size * 2 + fcs_output_size[i - 1]
            output_size = fcs_output_size[i]

            linear1 = nn.Linear(input_size, output_size)
            linear2 = nn.Linear(output_size, output_size)
            multi_scale_fc = nn.Sequential(linear1, nn.ReLU(), linear2)
            self.layer_convs_fcs.append(multi_scale_fc)

        self.multi_layer_predictor = nn.Linear(layer_feature_size, 1)

    def forward(self, images, names):
        """
        Args:
            images: Outfit images with shape (N, T, C, H, W)
            names: Description words of each item in outfit
        Return:
            out: Compatibility score
            vse_loss: Visual Semantic Loss
        """

        out, features, rep = self.compute_feature_fusion_score(images)

        if self.vse_off:
            vse_loss = torch.tensor(0.)
        else:
            vse_loss = self._compute_vse_loss(names, rep)

        return out, vse_loss

    def _compute_vse_loss(self, names, rep):
        """ Visual semantice loss which map both visual embedding and semantic embedding 
        into a common space.

        Reference: 
        https://github.com/xthan/polyvore/blob/e0ca93b0671491564b4316982d4bfe7da17b6238/polyvore/polyvore_model_bi.py#L362
        """
        # Normalized Semantic Embedding
        padded_names = rnn_utils.pad_sequence(names, batch_first=True).to(rep.device)
        mask = torch.gt(padded_names, 0)
        cap_mask = torch.ge(mask.sum(dim=1), 2)
        semb = self.sem_embedding(padded_names)
        semb = semb * (mask.unsqueeze(dim=2)).float()
        word_lengths = mask.sum(dim=1)
        word_lengths = torch.where(
            word_lengths == 0,
            (torch.ones(semb.shape[0]).float() * 0.1).to(rep.device),
            word_lengths.float(),
        )
        semb = semb.sum(dim=1) / word_lengths.unsqueeze(dim=1)
        semb = F.normalize(semb, dim=1)

        # Normalized Visual Embedding
        vemb = F.normalize(self.image_embedding(rep), dim=1)

        # VSE Loss
        semb = torch.masked_select(semb, cap_mask.unsqueeze(dim=1))
        vemb = torch.masked_select(vemb, cap_mask.unsqueeze(dim=1))
        semb = semb.reshape([-1, self.embedding_size])
        vemb = vemb.reshape([-1, self.embedding_size])
        scores = torch.matmul(semb, vemb.transpose(0, 1))
        diagnoal = scores.diag().unsqueeze(dim=1)
        cost_s = torch.clamp(0.2 - diagnoal + scores, min=0, max=1e6)  # 0.2 is margin
        cost_im = torch.clamp(0.2 - diagnoal.transpose(0, 1) + scores, min=0, max=1e6)
        cost_s = cost_s - torch.diag(cost_s.diag())
        cost_im = cost_im - torch.diag(cost_im.diag())
        vse_loss = cost_s.sum() + cost_im.sum()
        vse_loss = vse_loss / (semb.shape[0] ** 2)

        return vse_loss


    def compute_feature_fusion_score(self, images, activate=True):
        """
        Extract feature vectors from input images.
        Return:
            out: the compatibility score
            features: the visual embedding of the images, we use 1000-d in all experiments
            rep_last_2th: the representations of the second last year, which is 2048-d for resnet-50 backend
        """
        batch_size, item_num, _, _, img_size = images.shape
        images = torch.reshape(images, (-1, 3, img_size, img_size))  # (batch_size*item_num->16*4, 3, 224, 224)

        features, *rep = self.cnn(images)
        rep_l1, rep_l2, rep_l3, rep_l4, rep_last_2th = rep  # rep_last_2th是resnet去除最后的分类器后的特征表示
        # [64,256,56,56],[64,512,28,28],[64,1024,14,14],[64, 2048, 7, 7]

        #  多尺度特征融合
        rep_list = [rep_l1, rep_l2, rep_l3, rep_l4]
        multi_scale_concats = []

        for i, rep_li in enumerate(rep_list):
            # 多尺度特征融合模块
            # 对原有的特征进行组合，不同的卷积核大小对应不同的分支
            rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, 1, item_num, -1)
            # rep_l1 (16,1,4,256), rep_l2 (16,1,4,512), rep_l3 (16,1,4,1024), rep_l4 (16,1,4,2048)
            shape = rep_li.shape
            rep_li_copy = torch.cat((rep_li[:, :, 0, :], rep_li[:, :, 2, :], rep_li[:, :, 1, :], rep_li[:, :, 3, :],
                                     rep_li[:, :, 0, :]), 2).reshape(shape[0], shape[1], shape[2] + 1, shape[3])
            rep_li_double = torch.cat((rep_li, rep_li_copy), 2)  # (16,1,9,256), rep_l2 (16,1,9,512), rep_l3 (16,1,9,1024), rep_l4 (16,1,9,2048)

            # 采用不同尺寸的卷积核来处理特征，一共有两组
            multi_scale_li_feature = [layer_i_convs_scale(rep_li_double) for layer_i_convs_scale in self.layer_convs_1[i]]  # 2x2, 3x3, 4x4  3个尺寸的卷积核作用后的结果
            cat_feature1 = torch.cat(multi_scale_li_feature, 1)
            # cat_feature [16 x (21 + 14 + 16)], [16 x (42 + 28 + 32)], [16 x (85 + 57 + 64)], [16 x (170 + 114 + 128 )]

            multi_scale_li_feature2 = [layer_i_convs_scale(rep_li_double) for layer_i_convs_scale in self.layer_convs_2[i]]  # 2x2, 3x3, 4x4  3个尺寸的卷积核作用后的结果
            cat_feature2 = torch.cat(multi_scale_li_feature2, 1)
            # cat_feature [16 x (21 + 14 + 16)], [16 x (42 + 28 + 32)], [16 x (85 + 57 + 64)], [16 x (170 + 114 + 128 )]

            # 将两组的特征拼接起来
            cat_feature_fuse = torch.cat((cat_feature1, cat_feature2), 1)
            multi_scale_concats.append(cat_feature_fuse)  # [16, 3x255 + 2x254 + 1x253], [16, 3*511 + 2*510 + 1*509], [16, 3*1023 + 2*1022 + 1*1021], [16, 3*2047 + 2*2046 + 1*2045]

        # 多层级特征融合模块
        layer1_to_2 = F.relu(self.layer_convs_fcs[0](multi_scale_concats[0]))              # [16, 64]
        layer2_concat_layer1 = torch.cat((layer1_to_2, multi_scale_concats[1]), 1)
        layer2_to_3 = F.relu(self.layer_convs_fcs[1](layer2_concat_layer1) + layer1_to_2)     # [16, 64]
        layer3_concat_layer2 = torch.cat((layer2_to_3, multi_scale_concats[2]), 1)
        layer3_to_4 = F.relu(self.layer_convs_fcs[2](layer3_concat_layer2) + layer2_to_3)    # [16, 64]
        layer4_concat_layer3 = torch.cat((layer3_to_4, multi_scale_concats[3]), 1)
        layer4_to_out = F.relu(self.layer_convs_fcs[3](layer4_concat_layer3) + layer3_to_4)  # [16, 64]

        out = self.multi_layer_predictor(layer4_to_out)
        if activate:
            out = self.sigmoid(out)

        return out, features, rep_last_2th

if __name__ == "__main__":

    device = torch.device("cpu:0")
    model = CompatModel(embed_size=1000, vocabulary=1000).to(device)
    images = torch.ones([16, 4, 3, 224, 224])
    names = [torch.ones([5]) for _ in range(80)]
    output, vse_loss, tmasks_loss, features_loss = model(images, names)
