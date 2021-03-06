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
        query_linear = nn.Linear(input_size, self.all_head_size)
        nn.init.xavier_uniform_(query_linear.weight)
        nn.init.constant_(query_linear.bias, 0)
        self.query = nn.Sequential(
            query_linear,
            nn.ReLU(),
        )
        key_linear = nn.Linear(input_size, self.all_head_size)
        nn.init.xavier_uniform_(key_linear.weight)
        nn.init.constant_(key_linear.bias, 0)
        self.key = nn.Sequential(
            key_linear,
            nn.ReLU(),
        )
        value_linear = nn.Linear(input_size, self.all_head_size)
        nn.init.xavier_uniform_(value_linear.weight)
        nn.init.constant_(value_linear.bias, 0)
        self.value = nn.Sequential(
            value_linear,
            nn.ReLU(),
        )

        self.attn_dropout = nn.Dropout(attention_dropout_prob)

        # ??????self-attention ???????????????????????? LayerNorm ??????
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
        input_tensor's shape = (batch, n, input_size) = (batch_size, 3, 256|512|1024|2048)
        ??????# (batch,n, hidden_size) ???????????????
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
        # attention_scores'shape = (batch, num_attention_heads, n, n) ??????????????????(?????????)????????????????????????1??????????????????attention??????
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
        # ?????????attention?????????value?????????????????????
        context_layer = torch.matmul(attention_probs, value_layer)

        # ??????context_layer???????????????????????????????????????????????????????????????contiguous()??????tensor???????????????????????????????????????view()????????????
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # ??????????????????????????????????????????context_layer???(batch,n,all_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        out = self.LayerNorm(hidden_states)
        return out  # (batch,n, all_head_size) ???????????????



class GetInputAndQuery(nn.Module):
    def __init__(self):
        super(GetInputAndQuery, self).__init__()
        # Global average pooling layer
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, reps_pos, generator_id, outfit_num=4):
        """
            ?????????resnet??????????????????eg:(batch_size*outfit_num, 256, 56, 56)?????????input(3?????????)???query???(1??????,1??????)??????????????????
        """
        input_tensors = []
        pos_query_tensors = []
        neg_query_tensors = []
        batch_item_num, _, _, _ = reps_pos[0].shape  # ???????????????????????????????????????
        batch_size = batch_item_num // (outfit_num + 1)
        for i in range(outfit_num):
            # ???generator_id?????????????????????????????????????????????????????????????????????????????????????????????????????????
            # ?????????in_place??????????????????????????????
            # generator_item = reps_pos[i][:, generator_id, :, :, :]
            # swap_item_id = 3
            # swap_item = reps_pos[i][:, swap_item_id, :, :, :]
            # reps_pos[i][:, generator_id, :, :, :] = swap_item
            # reps_pos[i][:, swap_item_id, :, :, :] = generator_item

            rep_li = reps_pos[i]  # (batch_size*(outfit_num+1), 256, 56, 56)
            rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, outfit_num + 1, -1)  # avgpool2d?????????????????????????????????4????????????
            input_feature = torch.cat((rep_li[:, :generator_id, :], rep_li[:, generator_id + 1:outfit_num, :]), 1)  # (batch, 3 , 256)
            input_tensors.append(input_feature)
            pos_query_tensors.append(rep_li[:, generator_id, :].unsqueeze(1))  # a list of 4 tensors with shape (batch, 1 , 256)
            neg_query_tensors.append(rep_li[:, outfit_num, :].unsqueeze(1))    # a list of 4 tensors with shape (batch, 1 , 256) ????????????????????????
            # input_tensors???a list of  4 tensors with shape (batch, 3, 256/512/1024/2048), query_tensors??? a list of 4 tensors with shape (batch, 1, 256/512/1024/2048)
        return input_tensors, pos_query_tensors, neg_query_tensors

class GetPosAndNegOutfit(nn.Module):
    def __init__(self):
        super(GetPosAndNegOutfit, self).__init__()
        # Global average pooling layer
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, reps, generator_id, outfit_num=4, layer_lens=4):
        """
            ?????????resnet??????????????????eg:(batch_size*outfit_num, 256, 56, 56)????????????pos_outfit???neg_outfit??????????????????
        """
        input_tensors = []
        pos_outfit_tensors = []
        neg_outfit_tensors = []
        batch_item_num, _, _, _ = reps[0].shape  # ???????????????????????????????????????
        batch_size = batch_item_num // (outfit_num + 1)
        for i in range(layer_lens):  # ?????????4??????????????????????????????
            # ???generator_id?????????????????????????????????????????????????????????????????????????????????????????????????????????
            # ?????????in_place??????????????????????????????
            # generator_item = reps_pos[i][:, generator_id, :, :, :]
            # swap_item_id = 3
            # swap_item = reps_pos[i][:, swap_item_id, :, :, :]
            # reps_pos[i][:, generator_id, :, :, :] = swap_item
            # reps_pos[i][:, swap_item_id, :, :, :] = generator_item

            rep_li = reps[i]  # (batch_size*(outfit_num+1), 256, 56, 56)
            rep_li = self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, outfit_num + 1, -1)  # avgpool2d?????????????????????????????????4????????????
            pos_outfit_tensors.append(rep_li[:, :layer_lens, :])
            neg_outfit = torch.cat((rep_li[:, :generator_id, :], rep_li[:, layer_lens, :].unsqueeze(1), rep_li[:, generator_id+1:outfit_num, :]), 1)
            neg_outfit_tensors.append(neg_outfit)    # a list of 4 tensors with shape (batch, 4 , 256) ????????????????????????
            # input_tensors???a list of  4 tensors with shape (batch, 4, 256/512/1024/2048), query_tensors??? a list of 4 tensors with shape (batch, 4, 256/512/1024/2048)
        return pos_outfit_tensors, neg_outfit_tensors
class SelfAttenFeatureFuse(nn.Module):
    def __init__(self, num_attention_heads, input_size=256, hidden_size=256, hidden_dropout_prob=0, rep_lens=[256, 512, 1024, 2048]):
        """
        num_attention_heads????????????????????????????????????????????????
        hidden_size??? ???????????????????????????????????????
        hidden_dropout_prob??? nn.Drop()??????????????? 0???????????????????????????
        rep_lens : forward????????????input_tensors???????????????????????????????????????
        """
        super(SelfAttenFeatureFuse, self).__init__()
        # attention??????
        self.attentions = nn.ModuleList()
        self.rep_lens = rep_lens
        for rep_len in self.rep_lens:
            self.attentions.append(SelfAttention(num_attention_heads, rep_len, hidden_size, hidden_dropout_prob))

    def forward(self, input_tensors):
        """
        ???????????????????????????attention?????????????????????
        """
        layer_attention_out = []
        for i in range(len(self.rep_lens)):
            layer_i_output = self.attentions[i](input_tensors[i])  # ??????(batch, 3,hidden_size)
            layer_attention_out.append(layer_i_output)
        return layer_attention_out  # layer_attention_out??????4???[batch, 3, hidden_size]

class multi_layer_fuse(nn.Module):
    def __init__(self, hidden_size):
        super(multi_layer_fuse, self).__init__()
        # ???????????????????????????
        self.layer_attention_fcs = nn.ModuleList()
        self.flatten = nn.Flatten()
        for i in range(len(self.rep_lens)):
            input_size = hidden_size * 3  # 3?????????
            output_size = hidden_size * 3
            # print("input_size = {}, output_size = {}".format(input_size, output_size))
            linear = nn.Linear(input_size, output_size)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0)
            fc = nn.Sequential(linear, nn.ReLU())
            self.layer_attention_fcs.append(fc)

    def forward(self, layer_attention_out):
        """
        layer_attention_out ???????????????attention???????????????, 4???[batch, 3, 256]
        """
        # ????????????????????????????????????
        for i in range(len(layer_attention_out)):
            layer_attention_out[i] = self.flatten(layer_attention_out[i])

        out0 = self.layer_attention_fcs[0](layer_attention_out[0]) + layer_attention_out[0]
        layer1_cat_out0 = out0 + layer_attention_out[1]
        out1 = self.layer_attention_fcs[1](layer1_cat_out0) + layer1_cat_out0

        layer2_cat_out1 = out1 + layer_attention_out[2]
        out2 = self.layer_attention_fcs[2](layer2_cat_out1) + layer2_cat_out1

        layer3_cat_out2 = out2 + layer_attention_out[3]
        out3 = self.layer_attention_fcs[3](layer3_cat_out2) + layer3_cat_out2  # ??????(batch, 3 x 256)
        return out3  # out3 ??????(batch, 3 x 256)


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
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)  # ????????????resnet???????????????
        self.encoder = cnn
        self.get_pos_neg_outfit = GetPosAndNegOutfit()
        hidden_size = 256
        self.get_attention_feature = SelfAttenFeatureFuse(num_attention_heads=1, input_size=hidden_size, hidden_size=hidden_size, hidden_dropout_prob=0.5)
        # ??????attention????????????????????????query?????????
        self.query_mapping = nn.ModuleList()
        rep_lens = [256, 512, 1024, 2048]
        for i in range(4):
            fc = nn.Sequential(
                nn.Linear(rep_lens[i], hidden_size),
                nn.ReLU(),
            )
            self.query_mapping.append(fc)
        self.input_mapping = nn.ModuleList()
        # ??????attention????????????????????????input?????????
        for i in range(4):
            fc = nn.Sequential(
                nn.Linear(rep_lens[i], hidden_size),
                nn.ReLU(),
            )
            self.input_mapping.append(fc)
        # MFB??????
        self.mfb_drop = nn.Dropout(mfb_drop)

        self.layer_norms = nn.ModuleList()
        for i in range(len(rep_lens)):
            self.layer_norms.append(LayerNorm(hidden_size))

        layer_fuse_out_size = 32
        self.get_layer_attention_fuse = SelfAttenFeatureFuse(num_attention_heads=1, input_size=hidden_size, hidden_size=layer_fuse_out_size, hidden_dropout_prob=0.5)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size*4, hidden_size*2), nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1), nn.Sigmoid()
        )
        # ??????????????????????????????????????????
        self.get_compat_score = nn.Sequential(
            nn.Linear(hidden_size*3, 1),
            # nn.ReLU(),
            # nn.Linear(hidden_size, 1)
        )
        to_init_net = [self.get_compat_score, self.predictor, self.input_mapping, self.query_mapping, self.encoder.fc]
        for net in to_init_net:
            for m in net.modules():
                if isinstance(m, (nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)

    # ???????????????????????????generator_id??????????????????????????????
    def get_layer_features(self, reps, generator_id):
        """
        reps_neg: ???????????????generator_id??????????????????????????????????????? ?????????4?????? reps???4?????????????????? [batch_size,4,256,56,56],[batch_size,4,512,28,28],[batch_size,4,1024,14,14],[batch_size,4, 2048, 7, 7]
        generator_id,
        return ???????????????????????????generator_id??????????????????????????????  [batch_size,1,256,56,56],[batch_size,1,512,28,28],[batch_size,1,1024,14,14],[batch_size,1, 2048, 7, 7]
        """
        batch_size, outfit_num, _, _, _ = reps[0].shape  # ???????????????????????????????????????
        layer_reps = []
        for i in range(4):  # ?????????4???
            layer_reps.append(reps[i][:, generator_id, :, :, :])
        return layer_reps

    # ??????????????????????????????
    def get_layer_feature_score(self, input_fuse_features, pos_layer_features, neg_layer_features):
        """
        pos_layer_features???input_fuse_features, generator_layer_features?????????????????????(>)
        neg_layer_features???input_fuse_features, generator_layer_features
        ???????????????????????????[4, batch_size, feature_output_size(256)]
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
        images = torch.reshape(images, (-1, 3, img_size, img_size))  # (batch_size*item_num->16*5, 3, 224, 224)
        if self.need_rep:
            features, *rep = self.encoder(images)
            rep_l1, rep_l2, rep_l3, rep_l4, rep_last_2th = rep  # ?????????rep???????????????????????????
            # [batch_size*item_num,256,56,56],[..,512,28,28],[.., 1024,14,14],[.., 2048, 7, 7]
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
        # input_tensors, pos_query_tensors, neg_query_tensors = self.get_input_and_query(rep_list, target_id)  # input_tensors 4x(batch, 3, 256|512|...), query_tensors 4x(batch, 1, 256|512|...)
        pos_outfit_tensors, neg_outfit_tensors = self.get_pos_neg_outfit(rep_list, target_id)  # input_tensors 4x(batch, 3, 256|512|...), query_tensors 4x(batch, 1, 256|512|...)
        pos_layer_attention_out = self.get_attention_feature(pos_outfit_tensors)  # layer_attention_out??????4x(batch, 4, 256)
        neg_layer_attention_out = self.get_attention_feature(neg_outfit_tensors)  # layer_attention_out??????4x(batch, 4, 256)
        pos_query_layer = []
        neg_query_layer = []
        pos_question = []
        neg_question = []

        for i in range(len(pos_layer_attention_out)):
            pos_li = pos_layer_attention_out[i]
            neg_li = neg_layer_attention_out[i]
            pos_query_layer.append(pos_li[:, target_id, :])  # (batch, 256)
            neg_query_layer.append(neg_li[:, target_id, :])  # (batch, 256)

            pos_que = torch.cat((pos_li[:, :target_id, :], pos_li[:, target_id+1:, :]), 1).reshape(batch_size, -1)
            pos_question.append(pos_que)
            neg_que = torch.cat((neg_li[:, :target_id, :], neg_li[:, target_id+1:, :]), 1).reshape(batch_size, -1)  # (batch, 3*256)
            neg_question.append(neg_que)


        pos_query = torch.cat(pos_query_layer, 1)  # (batch, 256*4) ???????????????????????????
        neg_query = torch.cat(neg_query_layer, 1)
        pos_out = self.predictor(pos_query)
        neg_out = self.predictor(neg_query)

        pos_ques_sum = pos_question[0]  # (batch, 3*256)
        neg_ques_sum = neg_question[0]  # (batch, 3*256)
        for i in range(1, len(pos_question)):
            pos_ques_sum = pos_ques_sum + pos_question[i]
            neg_ques_sum = neg_ques_sum + neg_question[i]
        pos_score = self.get_compat_score(pos_ques_sum)
        neg_score = self.get_compat_score(neg_ques_sum)

        diff = pos_score - neg_score
        return pos_out, neg_out, diff, rep_last_2th
    def forward(self, images, names):
        pos_out, neg_out, diff, rep_last_2th = self.conpute_compatible_score(images)
        if self.vse_off:
            vse_loss = torch.tensor(0.)
        else:
            vse_loss = self._compute_vse_loss(names, rep_last_2th)
        return pos_out, neg_out, diff, vse_loss

if __name__ == "__main__":

    device = torch.device("cpu:0")
    model = CompatModel(embed_size=1000, vocabulary=1000, need_rep=True).to(device)
    images = torch.ones([16, 4, 3, 224, 224])
    names = [torch.ones([5]) for _ in range(80)]
    output, vse_loss, tmasks_loss, features_loss = model(images, names)
