import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import math

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    """
    反卷积的输出的张量尺寸计算方式
    img_size = [H, W]
    """
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape

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
        query_linear = nn.Linear(input_size, self.all_head_size)
        nn.init.xavier_uniform_(query_linear.weight)
        nn.init.constant_(query_linear.bias, 0)
        self.query = nn.Sequential(
            query_linear,
            # nn.ReLU(),
        )
        key_linear = nn.Linear(input_size, self.all_head_size)
        nn.init.xavier_uniform_(key_linear.weight)
        nn.init.constant_(key_linear.bias, 0)
        self.key = nn.Sequential(
            key_linear,
            # nn.ReLU(),
        )
        value_linear = nn.Linear(input_size, self.all_head_size)
        nn.init.xavier_uniform_(value_linear.weight)
        nn.init.constant_(value_linear.bias, 0)
        self.value = nn.Sequential(
            value_linear,
            # nn.ReLU(),
        )

        self.attn_dropout = nn.Dropout(attention_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        """
        input_tensor's shape = (batch, n, input_size) = (batch_size, 3, 100)
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
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        out = self.LayerNorm(hidden_states)
        return out  # (batch,n, all_head_size) 输出的维度

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

class AttentionFeatureFuse(nn.Module):
    def __init__(self, num_attention_heads=1, input_size=100, hidden_size=100, hidden_dropout_prob=0.5):
        super(AttentionFeatureFuse, self).__init__()
        self.last_attention = SelfAttention(num_attention_heads, input_size, hidden_size, hidden_dropout_prob)

    def forward(self, last_features_input, outfit_num=4):
        """
        features_input'shape (batch, 3, 100)
        """
        out = self.last_attention(last_features_input)  # 输出(batch, 3, hidden_size)
        return out

class LastFeatureFuse(nn.Module):
    def __init__(self, drop=0.5):
        super(LastFeatureFuse, self).__init__()
        # Global average pooling layer
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.last_fuse = nn.Sequential(
            nn.Linear(1000*3, 1024),
            nn.Dropout(drop),
            nn.LeakyReLU()
        )

    def forward(self, last_features_input, outfit_num=4):
        """
        features_input'shape (batch, 3, 1000)
        """
        shape = last_features_input.shape
        last_features = last_features_input.reshape(shape[0], -1)  # (batch,3000)
        out = self.last_fuse(last_features)
        return out


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
        return out + x

class Encoder(nn.Module):
    def __init__(self, output_size=100, drop=0.5):
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
            nn.AdaptiveAvgPool2d(1),   # [batch_size,512,1,1]
            nn.Flatten(),              # [batch_size,512]
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
        enc1 = self.conv1(img)
        feature1 = self.get_feature1(enc1)
        enc2 = self.conv2(enc1)
        feature2 = self.get_feature2(enc2)
        enc3 = self.conv3(enc2)
        feature3 = self.get_feature3(enc3)
        enc_drop = self.dropout(enc3.flatten(start_dim=1))
        out = self.fc(enc_drop)
        return out, feature3, feature2, feature1, [enc1, enc2, enc3]   # 3到1， 离原始图像越近, encx = [batch_size,512|512|1024,1,1]

class FeatureFusion(nn.Module):
    def __init__(self, item_input_size=100, output_size=100):
        super(FeatureFusion, self).__init__()
        self.getFeatureFuse = nn.Sequential(
            nn.Linear(item_input_size*3, output_size),
            nn.Sigmoid()
        )
    def forward(self, enc_x): # enc ( batch_size:16, 3, 100)
        enc_x = enc_x.reshape((enc_x.shape[0], -1))  # batch_size:16, 3*100 前3件为输入单品
        return self.getFeatureFuse(enc_x)  # batch_size, 100


#  高斯变化
class Transformer(nn.Module):
    def __init__(self, item_input_size=100, output_size=100):
        super(Transformer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(item_input_size*2, output_size),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(item_input_size*2, output_size),
        )

    def forward(self, enc_x, enc_desc, device):
        """
        enc_x, enc_desc :[batch_size, 100]
        """
        fuse_feature = torch.cat((enc_x, enc_desc), 1)
        z_mean = self.fc1(fuse_feature)
        z_log_var = self.fc2(fuse_feature)
        epsilon = torch.randn(size=z_mean.shape).to(device)
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon
        return z, z_mean, z_log_var

# 生成器模块
class Generator(nn.Module):
    def __init__(self, item_input_size=100, output_size=100, drop=0.5):
        super(Generator, self).__init__()
        self.seq = nn.Sequential(  # 输入(batch_size, input_size), 输出(batch_size, 1024)
            nn.Linear(item_input_size*3, 1024),
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
            nn.Flatten(),   # [batch_size,512]
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
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, fuse_feature):  # fuse_feature的尺寸与 z, enc_x, enc_desc有关
        dec = self.seq(fuse_feature)  # [batch_size, 1024]
        feature1 = self.get_feature1(dec)
        dec = torch.reshape(dec, (dec.shape[0], dec.shape[1], 1, 1))  # batch_size*1024*1*1
        dec = self.deconv1(dec)  # 512*4*4

        feature2 = self.get_feature2(dec)
        dec = self.deconv2(dec)  # 512*8*8

        feature3 = self.get_feature3(dec)
        low_resolution_img = self.deconv3(dec)  # 3*128*128 低分辨率

        low_to_mid = self.conv4(low_resolution_img)
        mid_to_high = self.conv5(low_to_mid)
        high_resolution_img = self.conv6(low_to_mid + mid_to_high)  # 高分辨率
        return low_resolution_img, high_resolution_img, [feature1, feature2, feature3]  # features : [3 batch_size, 100] # 1到3 离原始图像越近


class MLMSFF(nn.Module):
    def __init__(self):
        super(MLMSFF, self).__init__()
        # 多尺度融合层每一层的卷积核
        self.filter_sizes = [2, 3, 4]
        layer_feature_weights = [512, 512, 1024]
        rep_weight = 9  # 套装的个数，正常是4，如果将套装复制然后拼接了一次，就可以得到8了 ,又加了第一件，为了能够获得所有的22组合，33组合
        self.layer_deep1_convs = nn.ModuleList()  # 3 x 3 , 一共有3层, 每一层有3个卷积核
        self.layer_deep2_convs = nn.ModuleList()  # 3 x 3 , 一共有3层, 每一层有3个卷积核
        for i in range(len(layer_feature_weights)):
            multi_convs = nn.ModuleList()
            multi_convs2 = nn.ModuleList()
            for size in self.filter_sizes:
                conv_net = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size, stride=(1, size * size)),
                    nn.BatchNorm2d(1),
                    nn.LeakyReLU(),
                    nn.AvgPool2d(kernel_size=(rep_weight - size + 1, rep_weight // 2 - size + 1)),
                    nn.Flatten(),
                )
                multi_convs.append(conv_net)

                conv_net2 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size, stride=(1, size * size)),
                    nn.BatchNorm2d(1),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=(rep_weight - size + 1, rep_weight // 2 - size + 1)),
                    nn.Flatten(),
                )
                multi_convs2.append(conv_net2)
            self.layer_deep1_convs.append(multi_convs)
            self.layer_deep2_convs.append(multi_convs2)

        # 多尺度融合模块
        self.layer_convs_fcs = nn.ModuleList()
        fashion_item_rep_len = [0, 512, 512, 1024]
        fcs_output_size = [0, 64, 64, 64]
        for i in range(1, len(fashion_item_rep_len)):
            rep_len = fashion_item_rep_len[i]
            input_size = 0
            for size in self.filter_sizes:
                stride = size * size
                wi = (rep_len - size) // stride + 1
                hi = (4 - size) + 1
                # 卷积之后的池化操作, 对张量产生的影响
                wi = wi // hi
                hi = 1
                input_size = input_size + hi * wi
            input_size = input_size * 2 + fcs_output_size[i - 1]
            output_size = fcs_output_size[i]

            linear = nn.Linear(input_size, output_size)
            linear2 = nn.Linear(output_size, output_size)
            multi_scale_fc = nn.Sequential(linear, nn.ReLU(), linear2)
            self.layer_convs_fcs.append(multi_scale_fc)

        self.multi_layer_predictor = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # 网络参数初始化
        to_init_net = [self.multi_layer_predictor, self.layer_convs_fcs, self.layer_deep1_convs, self.layer_deep2_convs]
        for net in to_init_net:
            for m in net.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, rep_list):
        """
        rep_list为正样本套装或者负样本套装
        """
        # 多尺度融合
        multi_scale_concats = []
        for i, rep_li in enumerate(rep_list):
            shape = rep_li.shape
            rep_li = rep_li.reshape(shape[0], 1, shape[1], -1)  # (64, 1, 4, 512|512|1024)
            shape = rep_li.shape
            rep_li_copy = torch.cat((rep_li[:, :, 0, :], rep_li[:, :, 2, :], rep_li[:, :, 1, :], rep_li[:, :, 3, :],
                                     rep_li[:, :, 0, :]), 2).reshape(shape[0], shape[1], shape[2] + 1, shape[3])
            rep_li_double = torch.cat((rep_li, rep_li_copy), 2)  # (16,1,9,512), rep_l2 (16,1,9,512), rep_l3 (16,1,9,1024)

            multi_scale_li_feature = [layer_i_convs_scale(rep_li_double) for layer_i_convs_scale in self.layer_deep1_convs[i]]  # 2x2, 3x3, 4x4  3个尺寸的卷积核作用后的结果
            cat_feature1 = torch.cat(multi_scale_li_feature, 1)
            # cat_feature [[batch_size ,(42 + 28 + 32)], [batch_size , (42 + 28 + 32)], [batch_size ,(85 + 57 + 64)])]

            multi_scale_li_feature2 = [layer_i_convs_scale(rep_li_double) for layer_i_convs_scale in self.layer_deep2_convs[i]]  # 2x2, 3x3, 4x4  3个尺寸的卷积核作用后的结果
            cat_feature2 = torch.cat(multi_scale_li_feature2, 1)
            # cat_feature [[batch_size ,(42 + 28 + 32)*2], [batch_size ,(42 + 28 + 32)*2], [batch_size ,(85 + 57 + 64)*2])]

            cat_feature_fuse = torch.cat((cat_feature1, cat_feature2), 1)
            multi_scale_concats.append(cat_feature_fuse)

        # 多层级特征融合
        layer1_to_2 = F.relu(self.layer_convs_fcs[0](multi_scale_concats[0]))  # [16, 64]
        layer2_concat_layer1 = torch.cat((layer1_to_2, multi_scale_concats[1]), 1)
        layer2_to_3 = F.relu(self.layer_convs_fcs[1](layer2_concat_layer1) + layer1_to_2)  # [16, 64]
        layer3_concat_layer2 = torch.cat((layer2_to_3, multi_scale_concats[2]), 1)
        layer3_to_out = F.relu(self.layer_convs_fcs[2](layer3_concat_layer2) + layer2_to_3)  # [16, 64]

        out = self.multi_layer_predictor(layer3_to_out)
        return out

class MultiModuleGenerator(nn.Module):
    def __init__(self, vocabulary=None, num_attention_heads=0, device=torch.device("cuda:0"), feature_size=100):
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
        # feature_size = 100
        self.type_to_id = {'upper': 0, 'bottom': 1, 'bag': 2, 'shoe': 3}
        self.device = device
        self.get_desc_embedding = nn.Sequential(
            nn.Linear(in_features=vocabulary, out_features=feature_size),  # desc_len:1245
            nn.Sigmoid(),
        )
        self.encoder = Encoder(output_size=feature_size, drop=0.5)

        self.num_attention_heads = num_attention_heads
        if self.num_attention_heads > 0:
            self.attention_fuse = AttentionFeatureFuse(num_attention_heads=num_attention_heads, input_size=feature_size, hidden_size=feature_size)

        self.get_feature_fuse = FeatureFusion(item_input_size=feature_size, output_size=feature_size)

        self.transformer = Transformer(item_input_size=feature_size, output_size=feature_size)

        self.generator = Generator(item_input_size=feature_size, output_size=feature_size)

        # Global average pooling layer
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.get_compat_prob = MLMSFF()

        # 多层级特征交互模块目标单品与给定单品之间的交互空间转换矩阵
        self.layers = nn.ModuleList()
        for i in range(3):
            net = nn.Sequential(
                nn.Linear(feature_size, feature_size),
                nn.ReLU(),
                # LayerNorm(hidden_size=feature_size),
            )
            self.layers.append(net)

        to_init_modules = [self.get_desc_embedding, self.encoder, self.get_feature_fuse, self.transformer, self.generator]
        if num_attention_heads > 0:
            to_init_modules.append(self.attention_fuse)
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
    def get_layer_feature_score(self, enc, feature_x1, feature_x2, feature_x3, feature_yp, feature_yn, enc_x, enc_desc, feature_yg):
        """
        enc: 所有单品经过卷积神经网络最后的输出  (item_num, batch_size, 100)
        feature_x1:单品1经过卷积神经网络的3层输出  [3, batch_size, 100]
        feature_x2:单品2经过卷积神经网络的3层输出  [3, batch_size, 100]
        feature_x3:单品3经过卷积神经网络的3层输出  [3, batch_size, 100]
        feature_yp:单品4经过卷积神经网络的3层输出  [3, batch_size, 100] 目标正样本
        feature_yn:单品5经过卷积神经网络的3层输出  [3, batch_size, 100] 目标负样本
        feature_yg:生成单品经过卷积神经网络的3层输出  [3, batch_size, 100]
        enc_x 前3件输入单品融合后的特征   batch_size, 100
        enc_desc: 文本特征                batch_size, 100
        """
        enc_yp, enc_yn = enc[3], enc[4]  # enc_yp 表示目标单品正样本最后一层特征， enc_yn 表示目标单品负样本最后一层特征，尺寸都是batch_size, 100

        score_pos = torch.tensor(0) #+ torch.sum(enc_yp * enc_desc, dim=1) + torch.sum(enc_yp * enc_x, dim=1)  #   #尺寸是(batch_size,)
        score_neg = torch.tensor(0) #+ torch.sum(enc_yn * enc_desc, dim=1) + torch.sum(enc_yn * enc_x, dim=1)  #
        for i in range(len(feature_yg)):    # 共3层的特征
            score_pos = score_pos + torch.sum(feature_yp[i] * feature_yg[i], dim=1)  # [batch_size,]
            score_pos = score_pos + torch.sum(feature_yp[i] * self.layers[i](feature_x1[i]), dim=1)  # [batch_size,]
            score_pos = score_pos + torch.sum(feature_yp[i] * self.layers[i](feature_x2[i]), dim=1)  # [batch_size,]
            score_pos = score_pos + torch.sum(feature_yp[i] * self.layers[i](feature_x3[i]), dim=1)  # [batch_size,]

            score_neg = score_neg + torch.sum(feature_yn[i] * feature_yg[i], dim=1) # [batch_size,]
            score_neg = score_neg + torch.sum(feature_yn[i] * self.layers[i](feature_x1[i]), dim=1)  # [batch_size,]
            score_neg = score_neg + torch.sum(feature_yn[i] * self.layers[i](feature_x2[i]), dim=1)  # [batch_size,]
            score_neg = score_neg + torch.sum(feature_yn[i] * self.layers[i](feature_x3[i]), dim=1)  # [batch_size,]


        difference_score = score_pos - score_neg
        return difference_score  # (batch_size)

    # 获得正负样本的特征
    def get_pos_neg_outfit(self, rep_list, batch_size=64, item_num=5,):
        """
        输入：rep_list含有3层的特征,尺寸为 (batch_size:64 * item_num:5, 512|512|1024, x, x)
        输出： pos_outfit[(batch_size:64 ,item_num:4, 512|512|1024)]*3, neg_outfit[(batch_size:64 ,item_num:4, 512|512|1024)*3]
        """
        pos_outfit, neg_outfit = [], []
        for i, rep_li in enumerate(rep_list):
            rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, item_num, -1)  # (batch_size:64 , item_num:5, 512|512|1024)
            pos_outfit.append(rep_li[:, :4, :])  # 前4件为正样本
            neg_outfit.append(torch.cat((rep_li[:, :3, :], rep_li[:, 4:, :]), 1))
        return pos_outfit, neg_outfit


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
        batch_size, item_num, _, _, img_size = images.shape  # (64, 5, 3, 128, 128) item_num = 5,有4个正的,1个负的 第4件为目标正单品，第5件为目标负单品
        images = torch.reshape(images, (-1, 3, img_size, img_size))  # (batch_size*item_num->16*5, 3, 128, 128)
        # outfit_pos = images[:, :4, :, :, :]  # 正样本 (8, 4, 3, 224, 224)
        #
        # image_neg = images[:, 4:5, :, :, :]  # 负样本 (8, 1, 3, 224, 224)
        # generator_id = 3  # 第4件为目标正单品，下标为3
        # outfit_neg = torch.cat((images[:, :3, :, :, :], image_neg), dim=1)

        enc_desc = self.get_desc_embedding(names)  # (batch_size, 100)

        enc, layer3, layer2, layer1, rep_list = self.encoder(images)  # (batch_size*item_num->16*5, 100)
        enc = torch.reshape(enc, (batch_size, item_num, -1)).transpose(0, 1)  # (batch_size, item_num, 100) ---> (item_num, batch_size, 100)
        layer3 = torch.reshape(layer3, (batch_size, item_num, -1)).transpose(0, 1)  # 含有单个位置，所有batch的张量
        layer2 = torch.reshape(layer2, (batch_size, item_num, -1)).transpose(0, 1)
        layer1 = torch.reshape(layer1, (batch_size, item_num, -1)).transpose(0, 1)

        feature_x1 = [layer3[0], layer2[0], layer1[0]]   # 第一件单品的前3层特征, 从左到右远离图像
        feature_x2 = [layer3[1], layer2[1], layer1[1]]   # 第二件单品的前3层特征
        feature_x3 = [layer3[2], layer2[2], layer1[2]]   # 第三件单品的前3层特征
        feature_yp = [layer3[3], layer2[3], layer1[3]]   # 第四件单品的前3层特征 正样本
        feature_yn = [layer3[4], layer2[4], layer1[4]]   # 第五件单品的前3层特征 负样本

        enc_x = torch.stack((enc[0], enc[1], enc[2]), 1)  # (batch_size, 3, 100)
        # attention 模块
        if self.num_attention_heads > 0:
            enc_x = self.attention_fuse(enc_x)

        enc_x = self.get_feature_fuse(enc_x)  # 获得3件输入的最后一层特征
        # 变分转换
        z, z_mean, z_log_var = self.transformer(enc_x, enc_desc, self.device)

        # 生成部分
        fuse_feature = torch.cat((z, enc_x, enc_desc), 1)  # batch_size, 100
        low_resolution_img, high_resolution_img, feature_yg = self.generator(fuse_feature) # gen_features = [feature1, feature2, feature3]

        difference_pn = self.get_layer_feature_score(enc, feature_x1, feature_x2, feature_x3, feature_yp, feature_yn, enc_x, enc_desc, feature_yg)  # (batch_size,)

        # mlmsff 分类模型
        pos_outfit, neg_outfit = self.get_pos_neg_outfit(rep_list, batch_size=batch_size)
        pos_out = self.get_compat_prob(pos_outfit)
        neg_out = self.get_compat_prob(neg_outfit)


        # (3,128,128), (3,128,128), (batch_size,), (batch_size, 100), (batch_size, 100)
        return low_resolution_img, high_resolution_img, difference_pn, z_mean, z_log_var, pos_out, neg_out

if __name__ == "__main__":

    device = torch.device("cpu:0")
    # model = CompatModel(embed_size=1000, vocabulary=1000, need_rep=True).to(device)
    # images = torch.ones([16, 4, 3, 224, 224])
    # names = [torch.ones([5]) for _ in range(80)]
    # output, vse_loss, tmasks_loss, features_loss = model(images, names)
