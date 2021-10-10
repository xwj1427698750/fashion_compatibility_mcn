import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models

from resnet import resnet50

class CompatModel(nn.Module):
    def __init__(self, embed_size=1000, need_rep=False, vocabulary=None,
                 vse_off=False, pe_off=False, mlp_layers=2, conv_feats="1234",):
        """The Multi-Layered Comparison Network (MCN) for outfit compatibility prediction and diagnosis.
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
        super(CompatModel, self).__init__()
        self.vse_off = vse_off
        self.pe_off = pe_off
        self.mlp_layers = mlp_layers
        self.conv_feats = conv_feats

        cnn = resnet50(pretrained=True, need_rep=need_rep)
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)  # 更换原先resnet的最后一层
        self.cnn = cnn
        self.need_rep = need_rep
        self.num_rela = 10 * len(conv_feats)
        self.bn = nn.BatchNorm1d(self.num_rela)  # 4x4 relationship matrix have 16 elements, 其中10个元素是不重复的

        # Define predictor part
        if self.mlp_layers > 0:                  #套装搭配预测模块的多层感知机
            predictor = []
            for _ in range(self.mlp_layers-1):
                linear = nn.Linear(self.num_rela, self.num_rela)
                nn.init.xavier_uniform_(linear.weight)
                nn.init.constant_(linear.bias, 0)
                predictor.append(linear)
                predictor.append(nn.ReLU())
            linear = nn.Linear(self.num_rela, 1)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0)
            predictor.append(linear)
            self.predictor = nn.Sequential(*predictor)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(cnn.fc.weight)
        nn.init.constant_(cnn.fc.bias, 0)

        # # Type specified masks
        # # l1, l2, l3 is the masks for feature maps for the beginning layers
        # # not suffix one is for the last layer
        # # 10是comparsion matrix中不重复的元素个数
        # self.masks = nn.Embedding(10, embed_size)
        # self.masks.weight.data.normal_(0.9, 0.7)
        # self.masks_l1 = nn.Embedding(10, 256)
        # self.masks_l1.weight.data.normal_(0.9, 0.7)
        # self.masks_l2 = nn.Embedding(10, 512)
        # self.masks_l2.weight.data.normal_(0.9, 0.7)
        # self.masks_l3 = nn.Embedding(10, 1024)
        # self.masks_l3.weight.data.normal_(0.9, 0.7)

        # Semantic embedding model
        self.sem_embedding = nn.Embedding(vocabulary, 1000)
        # Visual embedding model
        self.image_embedding = nn.Linear(2048, 1000)

        # Global average pooling layer
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        # 多尺度融合层每一层的卷积核
        filter_sizes = [2, 3, 4]
        self.layer_convs = nn.ModuleList()  # 4 x 3 , 一共有4层, 每一层有3个卷积核
        for i in range(4):
            self.layer_convs.append(nn.ModuleList([nn.Conv2d(1, 1, size, (1, size*size)) for size in filter_sizes]))  #  1:in_channel, 1: out_channel, size : 卷积核大小, stride:(1,size*2)  2x2, 3x3, 4x4
        # stride = size * size
        # rep_len: 256
        # size = 2, w = 64, h = 3
        # size = 3, w = 29, h = 2
        # size = 4, w = 16, h = 1
        # rep_len: 512
        # size = 2, w = 128, h = 3
        # size = 3, w = 57, h = 2
        # size = 4, w = 32, h = 1
        # rep_len: 1024
        # size = 2, w = 256, h = 3
        # size = 3, w = 114, h = 2
        # size = 4, w = 64, h = 1
        # rep_len: 2048
        # size = 2, w = 512, h = 3
        # size = 3, w = 228, h = 2
        # size = 4, w = 128, h = 1

        # self.layer_convs_fc1 = nn.Linear(3*64 + 2*29 + 1*16, 256/2)
        #
        # self.layer_convs_fc2 = nn.Linear(3*128 + 2*57 + 1*32 + 256/2, 512/2)
        #
        # self.layer_convs_fc3 = nn.Linear(3*256 + 2*114 + 1*64 + 512/2, 1024/2)
        #
        # self.layer_convs_fc4 = nn.Linear(3*512 + 2*228 + 1*128 + 1024/2, 2048/2)
        self.layer_convs_fcs = nn.ModuleList()
        fashion_item_rep_len = [0, 256, 512, 1024, 2048]
        for i in range(1, len(fashion_item_rep_len)):
            rep_len = fashion_item_rep_len[i]
            input_size = 0
            for size in filter_sizes:
                stride = size * size
                wi = (rep_len - size) // stride + 1
                hi = (4 - size) + 1
                input_size = input_size + hi * wi
            input_size = input_size + fashion_item_rep_len[i-1] // 2
            self.layer_convs_fcs.append(nn.Linear(input_size, rep_len // 2))
            nn.init.xavier_uniform_(self.layer_convs_fcs[-1].weight)
            nn.init.constant_(self.layer_convs_fcs[-1].bias, 0)
        self.multi_layer_predictor = nn.Linear(1024, 1)

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
        if self.need_rep:
            out, features, tmasks, rep = self._compute_feature_fusion_score(images)
        else:
            out, features, tmasks = self._compute_feature_fusion_score(images)

        if self.vse_off:
            vse_loss = torch.tensor(0.)
        else:
            vse_loss = self._compute_vse_loss(names, rep)
        if self.pe_off:
            tmasks_loss, features_loss = torch.tensor(0.), torch.tensor(0.)
        else:
            tmasks_loss, features_loss = self._compute_type_repr_loss(tmasks, features)

        return out, vse_loss, tmasks_loss, features_loss

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
        semb = semb.reshape([-1, 1000])
        vemb = vemb.reshape([-1, 1000])
        scores = torch.matmul(semb, vemb.transpose(0, 1))
        diagnoal = scores.diag().unsqueeze(dim=1)
        cost_s = torch.clamp(0.2 - diagnoal + scores, min=0, max=1e6)  # 0.2 is margin
        cost_im = torch.clamp(0.2 - diagnoal.transpose(0, 1) + scores, min=0, max=1e6)
        cost_s = cost_s - torch.diag(cost_s.diag())
        cost_im = cost_im - torch.diag(cost_im.diag())
        vse_loss = cost_s.sum() + cost_im.sum()
        vse_loss = vse_loss / (semb.shape[0] ** 2)

        return vse_loss

    def _compute_type_repr_loss(self, tmasks, features):
        """ Here adopt two losses to improve the type-spcified represetations.
        `tmasks_loss` expect the masks to be sparse and `features_loss` regularize
        the feature vector to be a unit vector.

        Reference:
        Conditional Similarity Networks: https://arxiv.org/abs/1603.07810
        """
        # Type embedding loss
        # tmasks_loss = tmasks.norm(1) / len(tmasks)

        features_loss = features.norm(2) / np.sqrt((features.shape[0] * features.shape[1]))
        return torch.tensor(0.), features_loss

    def _compute_score(self, images, activate=True):
        """Extract feature vectors from input images.
        Return:
            out: the compatibility score
            features: the visual embedding of the images, we use 1000-d in all experiments
            masks: the mask for type-specified embedding
            rep: the representations of the second last year, which is 2048-d for resnet-50 backend
        """
        batch_size, item_num, _, _, img_size = images.shape
        images = torch.reshape(images, (-1, 3, img_size, img_size))  # (batch_size*item_num->16*4, 3, 224, 224)
        if self.need_rep:
            features, *rep = self.cnn(images)
            rep_l1, rep_l2, rep_l3, rep_l4, rep = rep  # 左侧的rep是倒数第二层的特征
            # [80,256,56,56],[80,512,28,28],[80,1024,14,14]
        else:
            features = self.cnn(images)  # (batch_size * item_num -> 16*4, 1000)

        relations = []
        features = features.reshape(batch_size, item_num, -1)  # (batch_size->16, 4, 1000)
        masks = F.relu(self.masks.weight)

        masks_weight = [masks]  # 函数需要返回的所有mask权值列表

        # Comparison matrix
        if "4" in self.conv_feats:
            for mi, (i, j) in enumerate(itertools.combinations_with_replacement([0, 1, 2, 3], 2)):
                # 一共有10轮的循环 (i,j)->(0,0),(0,1),..,(1,1),(1,2),...(2,2),....,(3,3)
                if self.pe_off:
                    left = F.normalize(features[:, i:i+1, :], dim=-1)  # (batch_size->16, 1, 1000)
                    right = F.normalize(features[:, j:j+1, :], dim=-1)
                else:
                    left = F.normalize(masks[mi] * features[:, i:i+1, :], dim=-1) # (batch_size->16, 1, 1000)
                    right = F.normalize(masks[mi] * features[:, j:j+1, :], dim=-1)
                rela = torch.matmul(left, right.transpose(1, 2)).squeeze() # (batch_size->16)
                relations.append(rela) # （10,16）

        # Comparision at Multi-Layered representations
        rep_list = []
        masks_list = []
        if "1" in self.conv_feats:
            rep_list.append(rep_l1); masks_list.append(self.masks_l1)
        if "2" in self.conv_feats:
            rep_list.append(rep_l2); masks_list.append(self.masks_l2)
        if "3" in self.conv_feats:
            rep_list.append(rep_l3); masks_list.append(self.masks_l3)
        for rep_li, masks_li in zip(rep_list, masks_list):
            rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, item_num, -1)
            # rep_l1 (16,4,256), rep_l2 (16,4,512), rep_l3 (16,4,1024)
            masks_li = F.relu(masks_li.weight)

            masks_weight.append(masks_li)

            # Enumerate all pairwise combination among the outfit then compare their features
            for mi, (i, j) in enumerate(itertools.combinations_with_replacement([0, 1, 2, 3], 2)):
                if self.pe_off: # 这个分支需要对比源代码判断一下
                    left = F.normalize(masks_li[mi] * rep_li[:, i:i+1, :], dim=-1)  # (16, 1, rep_li.shape[-1])
                    right = F.normalize(masks_li[mi] * rep_li[:, j:j+1, :], dim=-1)
                else:
                    left = F.normalize(masks_li[mi] * rep_li[:, i:i+1, :], dim=-1)  # (16, 1, 1000)
                    right = F.normalize(masks_li[mi] * rep_li[:, j:j+1, :], dim=-1)
                rela = torch.matmul(left, right.transpose(1, 2)).squeeze()  # (16)
                relations.append(rela)
        # relations 是个列表，10*4个元素，每个元素size是torch.Size([16]) [10*4,16]
        if batch_size == 1: # Inference during evaluation, which input one sample
            relations = torch.stack(relations).unsqueeze(0)
        else:
            relations = torch.stack(relations, dim=1)  # stack之后 torch.Size([16, 10*4])
        relations = self.bn(relations)  # torch.Size([16, 10*4])

        # Predictor
        if self.mlp_layers == 0:
            out = relations.mean(dim=-1, keepdim=True)
        else:
            out = self.predictor(relations) #torch.Size([16, 1])

        if activate:
            out = self.sigmoid(out)
        if self.need_rep:
            return out, features, masks_weight, rep
        else:
            return out, features, masks_weight

    def _compute_feature_fusion_score(self, images, activate=True):
        """Extract feature vectors from input images.
                Return:
                    out: the compatibility score
                    features: the visual embedding of the images, we use 1000-d in all experiments
                    masks: the mask for type-specified embedding
                    rep: the representations of the second last year, which is 2048-d for resnet-50 backend
                """
        batch_size, item_num, _, _, img_size = images.shape
        images = torch.reshape(images, (-1, 3, img_size, img_size))  # (batch_size*item_num->16*4, 3, 224, 224)
        if self.need_rep:
            features, *rep = self.cnn(images)
            rep_l1, rep_l2, rep_l3, rep_l4, rep = rep  # 左侧的rep是倒数第二层的特征
            # [64,256,56,56],[64,512,28,28],[64,1024,14,14],[64, 2048, 7, 7]
        else:
            features = self.cnn(images)  # (batch_size * item_num -> 16*4, 1000)

        # # ------------------------------- mcn的比较模块 -----------------------------------------
        # relations = []
        # features = features.reshape(batch_size, item_num, -1)  # (batch_size->16, 4, 1000)
        # masks = F.relu(self.masks.weight)
        #
        # masks_weight = [masks]  # 函数需要返回的所有mask权值列表
        #
        # # Comparison matrix
        # if "4" in self.conv_feats:
        #     for mi, (i, j) in enumerate(itertools.combinations_with_replacement([0, 1, 2, 3], 2)):
        #         # 一共有10轮的循环 (i,j)->(0,0),(0,1),..,(1,1),(1,2),...(2,2),....,(3,3)
        #         if self.pe_off:
        #             left = F.normalize(features[:, i:i + 1, :], dim=-1)  # (batch_size->16, 1, 1000)
        #             right = F.normalize(features[:, j:j + 1, :], dim=-1)
        #         else:
        #             left = F.normalize(masks[mi] * features[:, i:i + 1, :], dim=-1)  # (batch_size->16, 1, 1000)
        #             right = F.normalize(masks[mi] * features[:, j:j + 1, :], dim=-1)
        #         rela = torch.matmul(left, right.transpose(1, 2)).squeeze()  # (batch_size->16)
        #         relations.append(rela)  # （10,16）
        #
        # # Comparision at Multi-Layered representations
        # rep_list = []
        # masks_list = []
        # if "1" in self.conv_feats:
        #     rep_list.append(rep_l1);
        #     masks_list.append(self.masks_l1)
        # if "2" in self.conv_feats:
        #     rep_list.append(rep_l2);
        #     masks_list.append(self.masks_l2)
        # if "3" in self.conv_feats:
        #     rep_list.append(rep_l3);
        #     masks_list.append(self.masks_l3)
        # for rep_li, masks_li in zip(rep_list, masks_list):
        #     rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, item_num, -1)
        #     # rep_l1 (16,4,256), rep_l2 (16,4,512), rep_l3 (16,4,1024)
        #     masks_li = F.relu(masks_li.weight)
        #
        #     masks_weight.append(masks_li)
        #
        #     # Enumerate all pairwise combination among the outfit then compare their features
        #     for mi, (i, j) in enumerate(itertools.combinations_with_replacement([0, 1, 2, 3], 2)):
        #         if self.pe_off:  # 这个分支需要对比源代码判断一下
        #             left = F.normalize(masks_li[mi] * rep_li[:, i:i + 1, :], dim=-1)  # (16, 1, rep_li.shape[-1])
        #             right = F.normalize(masks_li[mi] * rep_li[:, j:j + 1, :], dim=-1)
        #         else:
        #             left = F.normalize(masks_li[mi] * rep_li[:, i:i + 1, :], dim=-1)  # (16, 1, 1000)
        #             right = F.normalize(masks_li[mi] * rep_li[:, j:j + 1, :], dim=-1)
        #         rela = torch.matmul(left, right.transpose(1, 2)).squeeze()  # (16)
        #         relations.append(rela)
        # # relations 是个列表，10*4个元素，每个元素size是torch.Size([16]) [10*4,16]
        # if batch_size == 1:  # Inference during evaluation, which input one sample
        #     relations = torch.stack(relations).unsqueeze(0)
        # else:
        #     relations = torch.stack(relations, dim=1)  # stack之后 torch.Size([16, 10*4])
        # relations = self.bn(relations)  # torch.Size([16, 10*4])

        #  ------------------------------------ 新添加的模块 --------------------------------------------
        #  多尺度特征融合
        rep_list = []
        if "1" in self.conv_feats:
            rep_list.append(rep_l1)
        if "2" in self.conv_feats:
            rep_list.append(rep_l2)
        if "3" in self.conv_feats:
            rep_list.append(rep_l3)
        if "4" in self.conv_feats:
            rep_list.append(rep_l4)
        self.multi_scale_concats = []
        for i, rep_li in enumerate(rep_list):
            rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, 1, item_num, -1)
            # rep_l1 (16,1,4,256), rep_l2 (16,1,4,512), rep_l3 (16,1,4,1024), rep_l4 (16,1,4,2048)
            multi_scale_li_feature = [layer_i_convs_scale(rep_li).squeeze().reshape(batch_size, -1) for layer_i_convs_scale in self.layer_convs[i]]  # 2x2, 3x3, 4x4  3个尺寸的卷积核作用后的结果
            # 2x2 ---> [16, 3 x 255],  [16, 3 x 511], [16, 3 x 1023], [16, 3 x 2047]
            # 3x3 ---> [16, 2 x 254],  [16, 2 x 510], [16, 2 x 1022], [16, 2 x 2046]
            # 4x4 ---> [16, 1 x 253],  [16, 1 x 509], [16, 1 x 1021], [16, 1 x 2045]
            self.multi_scale_concats.append(torch.cat(multi_scale_li_feature, 1))  # [16, 3x255 + 2x254 + 1x253], [16, 3*511 + 2*510 + 1*509], [16, 3*1023 + 2*1022 + 1*1021], [16, 3*2047 + 2*2046 + 1*2045]

        # 多层级特征融合
        layer1_to_2 = self.layer_convs_fcs[0](self.multi_scale_concats[0])  # [16, 256/2]
        layer1_to_2 = F.relu(layer1_to_2)
        layer2_concat_layer1 = torch.cat((layer1_to_2, self.multi_scale_concats[1]), 1)
        layer2_to_3 = self.layer_convs_fcs[1](layer2_concat_layer1)    # [16, 512/2]
        layer2_to_3 = F.relu(layer2_to_3)
        layer3_concat_layer2 = torch.cat((layer2_to_3, self.multi_scale_concats[2]), 1)
        layer3_to_4 = self.layer_convs_fcs[2](layer3_concat_layer2)    # [16, 1024/2]
        layer3_to_4 = F.relu(layer3_to_4)
        layer4_concat_layer3 = torch.cat((layer3_to_4, self.multi_scale_concats[3]), 1)
        layer4_to_out = self.layer_convs_fcs[3](layer4_concat_layer3)  # [16, 2048/2]
        layer4_to_out = F.relu(layer4_to_out)
        # 预测
        out = self.multi_layer_predictor(layer4_to_out)
        if activate:
            out = self.sigmoid(out)
        if self.need_rep:
            return out, features, _, rep
        else:
            return out, features, _


        # # Predictor
        # if self.mlp_layers == 0:
        #     out = relations.mean(dim=-1, keepdim=True)
        # else:
        #     out = self.predictor(relations)  # torch.Size([16, 1])
        #
        # if activate:
        #     out = self.sigmoid(out)
        # if self.need_rep:
        #     return out, features, masks, rep
        # else:
        #     return out, features, masks

if __name__ == "__main__":

    device = torch.device("cpu:0")
    model = CompatModel(embed_size=1000, vocabulary=1000, need_rep=True).to(device)
    images = torch.ones([16, 4, 3, 224, 224])
    names = [torch.ones([5]) for _ in range(80)]
    output, vse_loss, tmasks_loss, features_loss = model(images, names)
