import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
from scipy.special import comb


# 标准化特征，用于SelfAttention中
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


# 自注意力机制组件
class SelfAttention(nn.Module):
    """
    代码实现参考连接：https://blog.csdn.net/beilizhang/article/details/115282604
    """

    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob=0.5,
                 attention_dropout_prob=0.5):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        query_linear = nn.Linear(input_size, self.all_head_size)
        self.query = nn.Sequential(
            query_linear,
        )
        key_linear = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Sequential(
            key_linear,
        )
        value_linear = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Sequential(
            value_linear,
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
        input_tensor's shape = (batch, n, input_size) = (batch_size, 3, feature_size)
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
        # attention_scores'shape = (batch, num_attention_heads, n, n)
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
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        out = self.LayerNorm(hidden_states)
        return out  # (batch,n, all_head_size) 输出的维度


# 特征融合模块中的自注意力模块，用于融合多件单品的特征
class AttentionFeatureFuse(nn.Module):
    def __init__(self, attention_heads=1, input_size=96, hidden_size=96, hidden_dropout_prob=0.5):
        super(AttentionFeatureFuse, self).__init__()
        self.attention = SelfAttention(attention_heads, input_size, hidden_size, hidden_dropout_prob)

    def forward(self, last_features_input, outfit_num=4):
        """
        features_input'shape (batch, 3, feature_size)
        """
        out = self.attention(last_features_input)  # 输出(batch, 3, hidden_size)
        return out


# 超分模块，用于增强图像的清晰度， 用于Generator中
class SRResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(SRResNetBlock, self).__init__()
        self.same_shape = same_shape
        stride = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1,
                               stride=stride)
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
        return out + x


# 图像编码模块
class Encoder(nn.Module):
    def __init__(self, output_size=96, drop=0.5):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),  # batch_size*64*64*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),  # batch_size*128*32*32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),  # batch_size*256*16*16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),  # batch_size*512*8*8
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.get_feature1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [batch_size,512,1,1]
            nn.Flatten(),  # [batch_size,512]
            nn.Linear(512, output_size),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),  # batch_size*512*4*4
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.get_feature2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [batch_size,512,1,1]
            nn.Flatten(),  # [batch_size,512]
            nn.Linear(512, output_size),
            nn.Sigmoid(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=1, padding=0),  # batch_size*1024*1*1
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.get_feature3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, output_size),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Sequential(
            nn.Linear(1024, output_size),
            nn.Sigmoid(),
        )

    def forward(self, img):
        enc1 = self.conv1(img)  # batch_size*512*8*8
        feature1 = self.get_feature1(enc1)  # [batch_size,output_size]
        enc2 = self.conv2(enc1)  # batch_size*512*4*4
        feature2 = self.get_feature2(enc2)  # [batch_size,output_size]
        enc3 = self.conv3(enc2)  # batch_size*1024*1*1
        feature3 = self.get_feature3(enc3)  # [batch_size,output_size]
        enc_drop = self.dropout(enc3.flatten(start_dim=1))  # batch_size*1024
        enc = self.fc(enc_drop)  # [batch_size,output_size]
        return enc, feature3, feature2, feature1, [enc1, enc2, enc3]


# 将3件单品的特征融合在一起
class FeatureFusion(nn.Module):
    def __init__(self, input_size=96, output_size=96):
        super(FeatureFusion, self).__init__()
        self.getFeatureFuse = nn.Sequential(
            nn.Linear(input_size * 3, output_size),
            nn.Sigmoid()
        )

    def forward(self, enc_x):  # enc ( batch_size, 3, output_size)
        enc_x = enc_x.reshape((enc_x.shape[0], -1))  # (batch_size, 3*feature_size) 前3件为输入单品
        return self.getFeatureFuse(enc_x)  # (batch_size, output_size)


#  变分转换模块
class Transformer(nn.Module):
    def __init__(self, input_size=96, output_size=96):
        super(Transformer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size * 2, output_size),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(input_size * 2, output_size),
        )

    def forward(self, enc_x, enc_desc, device):
        """
        enc_x, enc_desc :[batch_size, output_size]
        """
        fuse_feature = torch.cat((enc_x, enc_desc), 1)
        z_mean = self.fc1(fuse_feature)
        z_log_var = self.fc2(fuse_feature)
        epsilon = torch.randn(size=z_mean.shape).to(device)
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon
        return z, z_mean, z_log_var


# 生成器模块
class Generator(nn.Module):
    def __init__(self, input_size=96, output_size=96, drop=0.5):
        super(Generator, self).__init__()
        self.seq = nn.Sequential(  # 输入(batch_size, input_size), 输出(batch_size, 1024)
            nn.Linear(input_size * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.get_feature1 = nn.Sequential(
            nn.Linear(1024, output_size),
            nn.Sigmoid(),
        )
        self.deconv1 = nn.Sequential(
            # 输入(batch_size, 1024, 1, 1), 输出(batch_size, 512, 4, 4) , 计算方式参见 convtrans2D_output_size
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.get_feature2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [batch_size,512,1,1]
            nn.Flatten(),  # [batch_size,512]
            nn.Linear(512, output_size),
            nn.Sigmoid(),
        )
        self.deconv2 = nn.Sequential(
            # 输入(batch_size, 512, 4, 4), 输出(batch_size, 512, 8, 8)
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.get_feature3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [batch_size,512,1,1]
            nn.Flatten(),  # [batch_size,512]
            nn.Linear(512, output_size),
            nn.Sigmoid(),
        )

        self.deconv3 = nn.Sequential(
            # 输入(batch_size, 512, 8, 8), 输出(batch_size, 256, 16, 16)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 输入(batch_size, 256, 16, 16), 输出(batch_size, 128, 32, 32)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 输入(batch_size, 128, 32, 32), 输出(batch_size, 64, 64, 64)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 输入(batch_size, 64, 64, 64), 输出(batch_size, 3, 128, 128)
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        # 输入是(batch_size, 3, 128, 128), 输出是(batch_size, 32, 128, 128)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 输入与输出都是(batch_size, 32, 128, 128)
        self.conv5 = nn.Sequential(
            SRResNetBlock(32, 32),
            SRResNetBlock(32, 32),
            SRResNetBlock(32, 32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 输入是(batch_size, 32, 128, 128), 输出是(batch_size, 3, 128, 128)
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, fuse_feature):  # fuse_feature的尺寸与 z, enc_x, enc_desc有关
        dec = self.seq(fuse_feature)  # [batch_size, 1024]
        feature1 = self.get_feature1(dec)
        dec = torch.reshape(dec, (dec.shape[0], dec.shape[1], 1, 1))  # batch_size*1024*1*1
        dec = self.deconv1(dec)  # batch_size*512*4*4

        feature2 = self.get_feature2(dec)
        dec = self.deconv2(dec)  # batch_size*512*8*8

        feature3 = self.get_feature3(dec)
        low_resolution_img = self.deconv3(dec)  # batch_size*3*128*128

        low_to_mid = self.conv4(low_resolution_img)
        mid_to_high = self.conv5(low_to_mid)
        high_resolution_img = self.conv6(low_to_mid + mid_to_high)  # batch_size*3*128*128
        features = [feature1, feature2, feature3]  # features : [3, batch_size, output_size]
        return low_resolution_img, high_resolution_img, features


# 套装搭配判断模块，第一个工作
class MLMSFF(nn.Module):
    def __init__(self, layer_size=256, outfit_items=4):
        super(MLMSFF, self).__init__()
        # 多尺度融合层每一层的卷积核
        self.outfit_items = outfit_items
        self.filter_sizes = [2, 3, 4]
        layer_len = [512, 512, 1024]  # encoder对应的卷积神经网络输出的3层特征的维度

        self.layer_convs_1 = nn.ModuleList()  # 3 x 3 , 一共有3层, 每一层有3个卷积核
        self.layer_convs_2 = nn.ModuleList()  # 3 x 3 , 一共有3层, 每一层有3个卷积核
        for i in range(len(layer_len)):
            multi_convs1 = nn.ModuleList()
            multi_convs2 = nn.ModuleList()
            for size in self.filter_sizes:
                # 维度为0上的kernel尺寸是comb(outfit_items, size)， 希望将单个分支卷积之后的特征池化为1
                # 维度为1上的kernel尺寸是outfit_items - size + 1， 这个尺寸的设置目的是希望不同size大小的分支经过池化后的尺寸接近
                size_comb = int(comb(outfit_items, size))  # 计算不同尺寸卷积核对应的组合数 C(4, 2), C(4, 3), C(4, 4),
                pool_kernel_size = (size_comb, outfit_items - size + 1)

                conv_net1 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size, stride=(size, size)),
                    nn.BatchNorm2d(1),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=pool_kernel_size),
                    nn.Flatten(),
                )
                multi_convs1.append(conv_net1)

                conv_net2 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size, stride=(size, size)),
                    nn.BatchNorm2d(1),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=pool_kernel_size),
                    nn.Flatten(),
                )
                multi_convs2.append(conv_net2)
            self.layer_convs_1.append(multi_convs1)
            self.layer_convs_2.append(multi_convs2)

        # 多尺度融合模块
        self.layer_convs_fcs = nn.ModuleList()
        fashion_item_rep_len = [0, 512, 512, 1024]
        fcs_output_size = [0, layer_size, layer_size, layer_size]
        for i in range(1, len(fashion_item_rep_len)):
            rep_len = fashion_item_rep_len[i]
            input_size = 0
            for size in self.filter_sizes:
                stride = size
                # 卷积操作后张量的尺寸计算公式: (W-F) // S + 1
                wi = (rep_len - size) // stride + 1
                hi = int(comb(self.outfit_items, size))  # 组合数的个数
                # 池化操作后张量的尺寸
                pool_stride = self.outfit_items - size + 1
                wi = wi // pool_stride  # 由 (wi - pool_stride) // pool_stride + 1 简化而来
                hi = 1  # 由 hi = (hi - hi) // hi + 1 简化得来
                input_size += hi * wi  # input_size : 3个分支处理过的特征转换成一维向量，然后拼接在一起的长度

            input_size = input_size * 2 + fcs_output_size[i - 1]
            output_size = fcs_output_size[i]

            linear = nn.Linear(input_size, output_size)
            linear2 = nn.Linear(output_size, output_size)
            multi_scale_fc = nn.Sequential(linear, nn.ReLU(), linear2)
            self.layer_convs_fcs.append(multi_scale_fc)

        self.multi_layer_predictor = nn.Linear(layer_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rep_list):
        """
        rep_list为正样本套装或者负样本套装
        """
        # 多尺度融合
        multi_scale_concats = []
        for layer, rep_li in enumerate(rep_list):
            shape = rep_li.shape
            rep_li = rep_li.reshape(shape[0], 1, shape[1], -1)  # (batch_size, 1, 4, 512|512|1024)
            multi_scale_li_feature1 = []
            multi_scale_li_feature2 = []

            # 对原有的特征进行组合，不同的卷积核大小对应不同的分支
            for idx, size in enumerate(self.filter_sizes):
                rep_li_combs = []
                for index_tuple in itertools.combinations(range(4), size):
                    # index_tuple:索引组合， size=2 形如(0,1).., size=3, 形如(0,1,2), size=4,为（0,1,2,3)
                    combs_list = [rep_li[:, :, i, :] for i in index_tuple]
                    combs = torch.stack(combs_list, 2)
                    rep_li_combs.append(combs)

                rep_li_combs = torch.cat(rep_li_combs, 2)
                # 在不同size和不同层级的情况下，rep_li_combinations的尺寸
                # C(4,2) 6种组合，每个组合含有2件单品特征， 6*2 = 12
                # C(4,3) 4种组合，每个组合含有3件单品特征， 4*3 = 12
                # C(4,4) 1种组合，每个组合含有4件单品特征， 4*1 = 4
                # rep_l1_combs:
                # size = 2 --> (batch,1,12,512),  size = 3 --> (batch,1,12,512),  size = 4 --> (batch,1,4,512)
                # rep_l2_combs:
                # size = 2 --> (batch,1,12,512),  size = 3 --> (batch,1,12,512),  size = 4 --> (batch,1,4,512)
                # rep_l3_combs:
                # size = 2 --> (batch,1,12,1024), size = 3 --> (batch,1,12,1024), size = 4 --> (batch,1,4,1024)

                multi_scale_li_feature1.append(self.layer_convs_1[layer][idx](rep_li_combs))
                multi_scale_li_feature2.append(self.layer_convs_2[layer][idx](rep_li_combs))

            # 不同特征层的cat_feature的特征尺度 rep_l1: [batch, 298], rep_l2: [batch, 298], rep_l3: [batch, 596]
            cat_feature1 = torch.cat(multi_scale_li_feature1, 1)
            cat_feature2 = torch.cat(multi_scale_li_feature2, 1)

            # 将两组的特征拼接起来，尺寸变双倍 rep_l1: [batch, 298*2], rep_l2: [batch, 298*2], rep_l3: [batch, 596*2]
            cat_feature_fuse = torch.cat((cat_feature1, cat_feature2), 1)
            multi_scale_concats.append(cat_feature_fuse)

        # 多层级特征融合
        layer1_to_2 = F.relu(self.layer_convs_fcs[0](multi_scale_concats[0]))  # [batch, layer_size]
        layer2_concat_layer1 = torch.cat((layer1_to_2, multi_scale_concats[1]), 1)
        layer2_to_3 = F.relu(self.layer_convs_fcs[1](layer2_concat_layer1) + layer1_to_2)  # [batch, layer_size]
        layer3_concat_layer2 = torch.cat((layer2_to_3, multi_scale_concats[2]), 1)
        layer3_to_out = F.relu(self.layer_convs_fcs[2](layer3_concat_layer2) + layer2_to_3)  # [batch, layer_size]

        out = self.multi_layer_predictor(layer3_to_out)
        out = self.sigmoid(out)
        return out


# MMFCP模型整体结构
class MultiModuleFashionCompatPrediction(nn.Module):
    def __init__(self, vocab_len=None, attention_heads=0, feature_size=96, device=None, enc_desc_off=True,
                 input_off=False, generator_off=False):
        """
        Args:
            vocab_len: the counts of words in the polyvore dataset.
        """
        super(MultiModuleFashionCompatPrediction, self).__init__()
        self.type_to_id = {'upper': 0, 'bottom': 1, 'bag': 2, 'shoe': 3}
        self.attention_heads = attention_heads
        self.device = device
        self.input_off = input_off
        self.enc_desc_off = enc_desc_off
        self.generator_off = generator_off

        self.get_desc_embedding = nn.Sequential(
            nn.Linear(in_features=vocab_len, out_features=feature_size),  # desc_len:1245
            nn.Sigmoid(),
        )
        self.encoder = Encoder(output_size=feature_size, drop=0.5)

        # 特征融合模块中的自注意力模块
        if self.attention_heads > 0:
            self.attention_fuse = AttentionFeatureFuse(attention_heads=attention_heads, input_size=feature_size,
                                                       hidden_size=feature_size)

        self.get_feature_fuse = FeatureFusion(input_size=feature_size, output_size=feature_size)

        self.transformer = Transformer(input_size=feature_size, output_size=feature_size)

        self.generator = Generator(input_size=feature_size, output_size=feature_size)

        # Global average pooling layer
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.get_compat_prob = MLMSFF()

        # 多层级特征交互模块: 目标单品与给定单品之间的交互空间转换矩阵
        self.layers = nn.ModuleList()
        for i in range(3):
            net = nn.Sequential(
                nn.Linear(feature_size, feature_size),
                nn.ReLU(),
            )
            self.layers.append(net)

    # 多层级特征交互得分计算
    def get_layer_feature_score(self, enc, feature_x1, feature_x2, feature_x3, feature_yp, feature_yn, enc_desc,
                                feature_yg):
        """
        Args:
        enc: 所有单品经过卷积神经网络最后的输出  (item_num, batch_size, feature_size)
        feature_x1:单品1经过卷积神经网络的3层输出  [3, batch_size, feature_size]
        feature_x2:单品2经过卷积神经网络的3层输出  [3, batch_size, feature_size]
        feature_x3:单品3经过卷积神经网络的3层输出  [3, batch_size, feature_size]
        feature_yp:单品4经过卷积神经网络的3层输出  [3, batch_size, feature_size] 目标正样本
        feature_yn:单品5经过卷积神经网络的3层输出  [3, batch_size, feature_size] 目标负样本
        feature_yg:生成单品经过卷积神经网络的3层输出  [3, batch_size, feature_size]
        enc_x 前3件输入单品融合后的特征   batch_size, feature_size
        enc_desc: 文本特征              batch_size, feature_size
        """
        enc_yp, enc_yn = enc[3], enc[4]  # enc_yp 表示目标单品正样本最后一层特征， enc_yn 表示目标单品负样本最后一层特征，尺寸都是batch_size, feature_size

        score_pos = torch.tensor(0)
        score_neg = torch.tensor(0)
        if not self.enc_desc_off:
            score_pos = score_pos + torch.sum(enc_yp * enc_desc, dim=1)  # 尺寸是(batch_size,)
            score_neg = score_neg + torch.sum(enc_yn * enc_desc, dim=1)

        for i in range(len(feature_yg)):  # 共3层的特征

            if not self.generator_off:  # 多层级特征交互模块中正负样本与生成单品之间的特征交互计算
                score_pos = score_pos + torch.sum(feature_yp[i] * feature_yg[i], dim=1)  # [batch_size,]
                score_neg = score_neg + torch.sum(feature_yn[i] * feature_yg[i], dim=1)  # [batch_size,]

            if not self.input_off:  # 多层级特征交互模块中正负样本与给定的多件单品之间的特征交互计算
                score_pos = score_pos + torch.sum(feature_yp[i] * self.layers[i](feature_x1[i]), dim=1)  # [batch_size,]
                score_pos = score_pos + torch.sum(feature_yp[i] * self.layers[i](feature_x2[i]), dim=1)  # [batch_size,]
                score_pos = score_pos + torch.sum(feature_yp[i] * self.layers[i](feature_x3[i]), dim=1)  # [batch_size,]

                score_neg = score_neg + torch.sum(feature_yn[i] * self.layers[i](feature_x1[i]), dim=1)  # [batch_size,]
                score_neg = score_neg + torch.sum(feature_yn[i] * self.layers[i](feature_x2[i]), dim=1)  # [batch_size,]
                score_neg = score_neg + torch.sum(feature_yn[i] * self.layers[i](feature_x3[i]), dim=1)  # [batch_size,]

        diff_score = score_pos - score_neg
        return diff_score  # (batch_size)

    # 获得正负样本的特征
    def get_pos_neg_outfit(self, rep_list, batch_size=64, item_num=5):
        """
        输入：rep_list含有3层的特征,尺寸为 (batch_size:64 * item_num:5, 512|512|1024, x, x)
        输出： pos_outfit[(batch_size:64 ,item_num:4, 512|512|1024)]*3, neg_outfit[(batch_size:64 ,item_num:4, 512|512|1024)*3]
        """
        pos_outfit, neg_outfit = [], []
        for i, rep_li in enumerate(rep_list):
            rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, item_num, -1)
            # rep_li reshap之后 (batch_size:64 , item_num:5, 512|512|1024)
            pos_outfit.append(rep_li[:, :4, :])  # 前4件为正样本
            neg_outfit.append(torch.cat((rep_li[:, :3, :], rep_li[:, 4:, :]), 1))
        return pos_outfit, neg_outfit

    def forward(self, images, names):
        """
        Args:
            images: Outfit images with shape (batch_size, item_num, C, H, W)
            names: Description words of each item in outfit
        Return:
            low_resolution_img:低画质的图片 (batch_size,3,128,128)
            high_resolution_img:高画质的图片 (batch_size,3,128,128)
            diff_score：多层级特征交互模块的中正样本与负样本的得分差 sp-sn (batch_size,)
            z_mean: 变分转换模块的中间变量，计算kl损失函数需要用到    (batch_size, feature_size)
            z_log_var:变分转换模块的中间变量，计算kl损失函数需要用到  (batch_size, feature_size)
            pos_out： 正样本与输入的3件单品组成套装的搭配概率 (batch_size, 1)
            neg_out： 负样本与输入的3件单品组成套装的搭配概率(batch_size, 1)
        """
        batch_size, item_num, _, _, img_size = images.shape  # (batch_size, 5, 3, 128, 128) item_num = 5,有4个正的,1个负的 第4件为目标正单品，第5件为目标负单品
        images = torch.reshape(images, (-1, 3, img_size, img_size))  # (batch_size*item_num, 3, 128, 128)

        enc_desc = self.get_desc_embedding(names)  # 文本描述的embedding表示 (batch_size, feature_size)

        enc, layer3, layer2, layer1, rep_list = self.encoder(images)  # (batch_size*item_num(64*5), feature_size)

        # enc以及layer1-3 张量尺寸的前后变化 (batch_size, item_num, feature_size) --> (item_num, batch_size, feature_size)
        enc = torch.reshape(enc, (batch_size, item_num, -1)).transpose(0, 1)
        layer3 = torch.reshape(layer3, (batch_size, item_num, -1)).transpose(0, 1)
        layer2 = torch.reshape(layer2, (batch_size, item_num, -1)).transpose(0, 1)
        layer1 = torch.reshape(layer1, (batch_size, item_num, -1)).transpose(0, 1)

        feature_x1 = [layer3[0], layer2[0], layer1[0]]  # 第一件单品的前3层特征, 每一层特征都是(batch_size, feature_size)
        feature_x2 = [layer3[1], layer2[1], layer1[1]]  # 第二件单品的前3层特征, 同上
        feature_x3 = [layer3[2], layer2[2], layer1[2]]  # 第三件单品的前3层特征, 同上
        feature_yp = [layer3[3], layer2[3], layer1[3]]  # 第四件单品的前3层特征(正样本), 同上
        feature_yn = [layer3[4], layer2[4], layer1[4]]  # 第五件单品的前3层特征(负样本), 同上

        enc_x = torch.stack((enc[0], enc[1], enc[2]), 1)  # enc_x :(batch_size, 3, feature_size)
        # 根据self.attention_heads判断是否增加attention 模块，前后的特征尺寸一致
        if self.attention_heads > 0:
            enc_x = self.attention_fuse(enc_x)  # enc_x :(batch_size, 3, feature_size)

        enc_x = self.get_feature_fuse(enc_x)  # 将3件单品的特征融合在一起, 输出的enc_x(batch_size, feature_size)

        # 变分转换 z, z_mean,z_log_var (batch_size, feature_size)
        z, z_mean, z_log_var = self.transformer(enc_x, enc_desc, self.device)

        # 生成部分
        fuse_feature = torch.cat((z, enc_x, enc_desc), 1)  # fuse_feature: (batch_size, feature_size*2)
        low_resolution_img, high_resolution_img, feature_yg = self.generator(fuse_feature)
        # (batch,3,128,128), (batch,3,128,128), feature_yg为[feature1, feature2, feature3]，每个feature是(batch_size, feature_size)

        diff_score = self.get_layer_feature_score(enc, feature_x1, feature_x2, feature_x3, feature_yp, feature_yn,
                                                  enc_desc, feature_yg)  # diff_score :(batch_size,)

        # mlmsff 套装搭配模型
        pos_outfit, neg_outfit = self.get_pos_neg_outfit(rep_list, batch_size=batch_size)
        pos_out = self.get_compat_prob(pos_outfit)  # (batch_size, 1)
        neg_out = self.get_compat_prob(neg_outfit)  # (batch_size, 1)

        return low_resolution_img, high_resolution_img, diff_score, z_mean, z_log_var, pos_out, neg_out

