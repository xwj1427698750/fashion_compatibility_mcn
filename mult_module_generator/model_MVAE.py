import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import math

from model import CompatModel
from resnet import resnet50, resnet34



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob=0.5, attention_dropout_prob=0.5):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        self.query = nn.Sequential(
            nn.Linear(input_size, self.all_head_size),
            nn.ReLU(),
        )
        self.key = nn.Sequential(
            nn.Linear(input_size, self.all_head_size),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(input_size, self.all_head_size),
            nn.ReLU(),
        )

        self.attn_dropout = nn.Dropout(attention_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        """
        input_tensor's shape = (batch, n, input_size) = (batch_size, 3, 256|512|1024|2048)
        输出# (batch,n, hidden_size) 输出的维度
        """
        # mixed_xxx_layer'shape = (batch, n, all_head_size)
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # xxx_layer'shape = (batch, num_attention_heads, n, attention_head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores'shape = (batch, num_attention_heads, n, n) 最后的的维度(每一行)的内容是时尚单品1对其他单品的attention得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)

        # context_layer'shape = (batch,num_attention_heads,n,attention_head_size)
        # 这里是attention得分和value加权求得的均值
        context_layer = torch.matmul(attention_probs, value_layer)

        # 变换context_layer维度，为了后面将各头得到的结果拼接。这里的contiguous()是将tensor的内存变成连续的，为后面的view()做准备。
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 将各注意力头的结果拼接起来，context_layer：(batch,n,all_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        out = self.LayerNorm(hidden_states)
        return out  # (batch,n, all_head_size) 输出的维度



class GetInputAndQuery(nn.Module):
    def __init__(self):
        super(GetInputAndQuery, self).__init__()
        # Global average pooling layer
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, reps_pos, generator_id, outfit_num=4):
        """
            输入是resnet每一层的输出:eg:(batch_size*outfit_num, 256, 56, 56)，分离input(3件单品)和query项(一件单品)每一层的特征
        """
        input_tensors = []
        query_tensors = []
        batch_item_num, _, _, _ = reps_pos[0].shape  # 获取每一层级特征的尺寸信息
        batch_size = batch_item_num // outfit_num
        for i in range(outfit_num):
            # 将generator_id对应的张量与最后一件单品的张量进行交换，保证生成模型输入是连续在一起的
            # 下面是in_place操作，反向传播会出错
            # generator_item = reps_pos[i][:, generator_id, :, :, :]
            # swap_item_id = 3
            # swap_item = reps_pos[i][:, swap_item_id, :, :, :]
            # reps_pos[i][:, generator_id, :, :, :] = swap_item
            # reps_pos[i][:, swap_item_id, :, :, :] = generator_item

            rep_li = reps_pos[i]  # (batch_size*outfit_num, 256, 56, 56)
            rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, outfit_num, -1)  # avgpool2d的输入尺寸有要求只能是4个维度的，不能是
            input_feature = torch.cat((rep_li[:, :generator_id, :], rep_li[:, generator_id + 1:, :]), 1)  # (batch, 3 , 256)
            input_tensors.append(input_feature)
            query_tensors.append(rep_li[:, generator_id, :].unsqueeze(1))  # a list of 4 tensors with shape (batch, 1 , 256)
            # input_tensors：a list of  4 tensors with shape (batch, 3, 256/512/1024/2048), query_tensors： a list of 4 tensors with shape (batch, 1, 256/512/1024/2048)
        return input_tensors, query_tensors

class SelfAttenFeatureFuse(nn.Module):
    def __init__(self, num_attention_heads, input_size=256, hidden_size=256, hidden_dropout_prob=0, rep_lens=[256, 512, 1024, 2048]):
        """
        num_attention_heads：表示采用的多头注意力机制的数量
        hidden_size： 单个时尚单品最终输出的维度
        hidden_dropout_prob： nn.Drop()中的概率， 0表示去除元素的概率
        rep_lens : forward函数中的input_tensors中单品的特征对应的维度列表
        """
        super(SelfAttenFeatureFuse, self).__init__()
        # attention模块
        self.attentions = nn.ModuleList()
        self.rep_lens = rep_lens
        for rep_len in self.rep_lens:
            self.attentions.append(SelfAttention(num_attention_heads, input_size, hidden_size, hidden_dropout_prob))

    def forward(self, input_tensors):
        """
        输出每一个层级经过attention加权之后的特征
        """
        layer_attention_out = []
        for i in range(len(self.rep_lens)):
            layer_i_output = self.attentions[i](input_tensors[i])  # 输出(batch, 3,hidden_size)
            layer_attention_out.append(layer_i_output)
        return layer_attention_out  # layer_attention_out输出4个[batch, 3, hidden_size]

class multi_layer_fuse(nn.Module):
    def __init__(self, hidden_size):
        super(multi_layer_fuse, self).__init__()
        # 多层级特征融合模块
        self.layer_attention_fcs = nn.ModuleList()
        self.flatten = nn.Flatten()
        for i in range(len(self.rep_lens)):
            input_size = hidden_size * 3  # 3件单品
            output_size = hidden_size * 3
            # print("input_size = {}, output_size = {}".format(input_size, output_size))
            linear = nn.Linear(input_size, output_size)
            fc = nn.Sequential(linear, nn.ReLU())
            self.layer_attention_fcs.append(fc)

    def forward(self, layer_attention_out):
        """
        layer_attention_out 每一个层级attention的后的特征, 4个[batch, 3, 256]
        """
        # 更改每一个层级特征的尺寸
        for i in range(len(layer_attention_out)):
            layer_attention_out[i] = self.flatten(layer_attention_out[i])

        out0 = self.layer_attention_fcs[0](layer_attention_out[0]) + layer_attention_out[0]
        layer1_cat_out0 = out0 + layer_attention_out[1]
        out1 = self.layer_attention_fcs[1](layer1_cat_out0) + layer1_cat_out0

        layer2_cat_out1 = out1 + layer_attention_out[2]
        out2 = self.layer_attention_fcs[2](layer2_cat_out1) + layer2_cat_out1

        layer3_cat_out2 = out2 + layer_attention_out[3]
        out3 = self.layer_attention_fcs[3](layer3_cat_out2) + layer3_cat_out2  # 输出(batch, 3 x 256)
        return out3  # out3 输出(batch, 3 x 256)


class MultiModuleGenerator(nn.Module):
    def __init__(self, embed_size=1000, need_rep=True, vocabulary=None,
                 vse_off=False, pe_off=False, mlp_layers=2, conv_feats="1234", target_type="upper", device=torch.device("cuda:0"), mfb_drop=0):
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
        self.target_type = target_type
        self.device = device
        self.conv_feats = conv_feats
        self.vse_off = vse_off
        self.need_rep = need_rep
        # Semantic embedding model
        self.sem_embedding = nn.Embedding(vocabulary, 1000)
        # Visual embedding model
        self.image_embedding = nn.Linear(2048, 1000)

        cnn = resnet50(pretrained=True, need_rep=need_rep)
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)  # 更换原先resnet的最后一层
        self.encoder = cnn
        self.get_input_and_query = GetInputAndQuery()
        hidden_size = 256
        self.get_attention_feature = SelfAttenFeatureFuse(num_attention_heads=1, input_size=hidden_size, hidden_size=hidden_size, hidden_dropout_prob=0.5)
        # 层级attention的时候，增加了对query的映射
        self.query_mapping = nn.ModuleList()
        rep_lens = [256, 512, 1024, 2048]
        for i in range(4):
            fc = nn.Sequential(
                nn.Linear(rep_lens[i], hidden_size),
                nn.ReLU(),
            )
            self.query_mapping.append(fc)
        self.input_mapping = nn.ModuleList()
        # 层级attention的时候，增加了对input的映射
        for i in range(4):
            fc = nn.Sequential(
                nn.Linear(rep_lens[i], hidden_size),
                nn.ReLU(),
            )
            self.input_mapping.append(fc)
        # MFB模块
        self.mfb_drop = nn.Dropout(mfb_drop)

        self.layer_norms = nn.ModuleList()
        for i in range(len(rep_lens)):
            self.layer_norms.append(LayerNorm(hidden_size))
        # self.layer_norms2 = nn.ModuleList()
        # for i in range(len(rep_lens)):
        #     self.layer_norms2.append(LayerNorm(hidden_size))
        layer_fuse_out_size = 32
        self.get_layer_attention_fuse = SelfAttenFeatureFuse(num_attention_heads=1, input_size=hidden_size, hidden_size=layer_fuse_out_size, hidden_dropout_prob=0.5)
        self.predictor = nn.Sequential(
            nn.Linear(3*4 + 4*3*layer_fuse_out_size, 1),
            nn.Sigmoid()
        )

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
    def get_layer_feature_score(self, input_fuse_features, pos_layer_features, neg_layer_features):
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
            score_neg = score_neg + torch.sum(input_fuse_features[layer_index] * neg_layer_features[layer_index], dim=1, keepdim=True)  # [batch_size, 1]

        difference_score = score_pos - score_neg
        return difference_score  # (batch_size, 1)

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

    def conpute_compatible_score(self, images):
        """
            Args:
                images: Outfit images with shape (N, T, C, H, W)
                names: Description words of each item in outfit

            Return:
                out: Compatibility score
                vse_loss: Visual Semantic Loss
            """
        batch_size, item_num, _, _, img_size = images.shape
        images = torch.reshape(images, (-1, 3, img_size, img_size))  # (batch_size*item_num->16*4, 3, 224, 224)
        if self.need_rep:
            features, *rep = self.encoder(images)
            rep_l1, rep_l2, rep_l3, rep_l4, rep_last_2th = rep  # 左侧的rep是倒数第二层的特征
            # [64,256,56,56],[64,512,28,28],[64,1024,14,14],[64, 2048, 7, 7]
        else:
            features = self.encoder(images)  # (batch_size * item_num -> 16*4, 1000)

        rep_list = []
        if "1" in self.conv_feats:
            rep_list.append(rep_l1)
        if "2" in self.conv_feats:
            rep_list.append(rep_l2)
        if "3" in self.conv_feats:
            rep_list.append(rep_l3)
        if "4" in self.conv_feats:
            rep_list.append(rep_l4)

        target_id = self.type_to_id[self.target_type]
        input_tensors, query_tensors = self.get_input_and_query(rep_list, target_id)  # input_tensors 4x(batch, 3, 256|512|...), query_tensors 4x(batch, 1, 256|512|...)


        # 计算wide部分， query对应的单品与input对应的3件单品的每一个层进行点积运算, 3x4 = 12
        wide_scores = []  # 处理完后是4 (batch, 3,1)
        wide_out = []
        wide_param = 0.1  # 减轻wide输出对结果的影响
        for i in range(len(input_tensors)):
            # 增加了一层额外的空间映射
            query_li = self.query_mapping[i](query_tensors[i])
            input_li = self.input_mapping[i](input_tensors[i])
            score = torch.matmul(input_li, query_li.transpose(1, 2))  # (batch, 3,1)
            wide_scores.append(F.normalize(score, dim=1))
            wide_out.append(score.squeeze(2))  #(batch, 3,1)-->(16,3)
        wide_out = torch.cat(wide_out, 1) * wide_param  # (batch, 3x4)

        #  计算deep部分 + MFB模块
        input_mfb_tensors = []
        for i in range(len(input_tensors)):
            query_li = self.query_mapping[i](query_tensors[i])
            input_li = self.input_mapping[i](input_tensors[i])
            input_li = torch.mul(input_li, query_li)
            input_li = self.mfb_drop(input_li)
            input_li = torch.sqrt(F.relu(input_li)) - torch.sqrt(F.relu(-input_li))
            input_mfb_tensors.append(self.layer_norms[i](input_li))
        layer_attention_out = self.get_attention_feature(input_mfb_tensors)  # layer_attention_out输出4x(batch, 3, 256)

        for i in range(len(layer_attention_out)):
            layer_attention_out[i] = layer_attention_out[i] * wide_scores[i]  # (batch, 3, 256) * (batch, 3, 1)

        # 计算层级attention
        layer_attention_fuse_list = self.get_layer_attention_fuse(layer_attention_out)  # (batch, 3, 32)
        for i in range(len(layer_attention_fuse_list)):
            layer_attention_fuse_list[i] = layer_attention_fuse_list[i].reshape(batch_size, -1)
        deep_out = torch.cat(layer_attention_fuse_list, 1)  # (batch, 4*3*32)
        deep_out = F.normalize(deep_out, dim=1)
        out = torch.cat((wide_out, deep_out), 1)
        out = self.predictor(out)
        return out, rep_last_2th

    def forward(self, images, names):
        out_prob, rep = self.conpute_compatible_score(images)
        if self.vse_off:
            vse_loss = torch.tensor(0.)
        else:
            vse_loss = self._compute_vse_loss(names, rep)
        return out_prob, vse_loss

if __name__ == "__main__":

    device = torch.device("cpu:0")
    model = CompatModel(embed_size=1000, vocabulary=1000, need_rep=True).to(device)
    images = torch.ones([16, 4, 3, 224, 224])
    names = [torch.ones([5]) for _ in range(80)]
    output, vse_loss, tmasks_loss, features_loss = model(images, names)
