import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from model import CompatModel


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.par = 1

    def forward(self, x):
        return x


class MultiFarm(nn.Module):
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
        super(MultiFarm, self).__init__()
        self.encoder = CompatModel(embed_size=embed_size, need_rep=need_rep, vocabulary=vocabulary, vse_off=vse_off,
                                   pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats)

        self.decoder = Decoder()

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


if __name__ == "__main__":

    device = torch.device("cpu:0")
    model = CompatModel(embed_size=1000, vocabulary=1000, need_rep=True).to(device)
    images = torch.ones([16, 4, 3, 224, 224])
    names = [torch.ones([5]) for _ in range(80)]
    output, vse_loss, tmasks_loss, features_loss = model(images, names)
