import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet import MXNetError


class SRResNetBlock(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(SRResNetBlock, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return out + x


class farm_encoder(nn.Block):
    def __init__(self, **kwargs):
        super(farm_encoder, self).__init__(**kwargs)
        self.convs_1 = nn.Sequential()
        self.convs_1.add(
            nn.Conv2D(channels=64, kernel_size=4, strides=2, padding=1),  # 64*64*64
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(channels=128, kernel_size=4, strides=2, padding=1),  # 128*32*32
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(channels=256, kernel_size=4, strides=2, padding=1),  # 256*16*16
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(channels=512, kernel_size=4, strides=2, padding=1),  # 512*8*8
            nn.BatchNorm(),
            nn.Activation('relu')
        )
        self.feature_3 = nn.Sequential()
        self.feature_3.add(
            nn.GlobalAvgPool2D(),  # [batch_size,512,1,1]
            nn.Flatten(),  # [batch_size,512]
            nn.Dense(100, activation='sigmoid')
        )
        self.convs_2 = nn.Sequential()
        self.convs_2.add(
            nn.Conv2D(channels=512, kernel_size=4, strides=2, padding=1),  # 512*4*4
            nn.BatchNorm(),
            nn.Activation('relu')
        )
        self.feature_2 = nn.Sequential()
        self.feature_2.add(
            nn.GlobalAvgPool2D(),  # [batch_size,512,1,1]
            nn.Flatten(),  # [batch_size,512]
            nn.Dense(100, activation='sigmoid')
        )
        self.convs_3 = nn.Sequential()
        self.convs_3.add(
            nn.Conv2D(channels=1024, kernel_size=4, strides=1, padding=0),  # 1024*1*1
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Flatten()  # [batch_size,1024]
        )
        self.feature_1 = nn.Sequential()
        self.feature_1.add(
            nn.Dense(100, activation='sigmoid')
        )
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Dense(100, activation='sigmoid')

    def forward(self, img):
        enc = self.convs_1(img)
        feature_3 = self.feature_3(enc)
        enc = self.convs_2(enc)
        feature_2 = self.feature_2(enc)
        enc = self.convs_3(enc)
        feature_1 = self.feature_1(enc)
        enc = self.dropout(enc)
        enc = self.dense(enc)
        return enc, feature_1, feature_2, feature_3


class farm_transformer(nn.Block):
    def __init__(self, **kwargs):
        super(farm_transformer, self).__init__(**kwargs)
        self.dense_1 = nn.Dense(100)
        self.dense_2 = nn.Dense(100)

    def forward(self, enc_x, enc_desc, ctx):
        con = nd.concat(enc_x, enc_desc, dim=1)
        z_mean = self.dense_1(con)
        z_log_var = self.dense_2(con)
        epsilon = nd.random_normal(loc=0, scale=1, shape=z_mean.shape, ctx=ctx)
        z = z_mean + nd.exp(0.5 * z_log_var) * epsilon
        return z, z_mean, z_log_var


class farm_generator(nn.Block):
    def __init__(self, **kwargs):
        super(farm_generator, self).__init__(**kwargs)
        self.seq = nn.Sequential()
        self.seq.add(
            nn.Dense(1024),
            nn.BatchNorm(),
            nn.Activation('relu')
        )
        self.feature_1 = nn.Sequential()
        self.feature_1.add(
            nn.Dense(100, activation='sigmoid')
        )
        self.convs_1 = nn.Sequential()
        self.convs_1.add(
            nn.Conv2DTranspose(channels=512, kernel_size=4, strides=1, padding=0),  # 512*4*4
            nn.BatchNorm(),
            nn.Activation('relu')
        )
        self.feature_2 = nn.Sequential()
        self.feature_2.add(
            nn.GlobalAvgPool2D(),  # [batch_size,512,1,1]
            nn.Flatten(),  # [batch_size,512]
            nn.Dense(100, activation='sigmoid')
        )
        self.convs_2 = nn.Sequential()
        self.convs_2.add(
            nn.Conv2DTranspose(channels=512, kernel_size=4, strides=2, padding=1),  # 512*8*8
            nn.BatchNorm(),
            nn.Activation('relu')
        )
        self.feature_3 = nn.Sequential()
        self.feature_3.add(
            nn.GlobalAvgPool2D(),  # [batch_size,512,1,1]
            nn.Flatten(),  # [batch_size,512]
            nn.Dense(100, activation='sigmoid')
        )
        self.convs_3 = nn.Sequential()
        self.convs_3.add(
            nn.Conv2DTranspose(channels=256, kernel_size=4, strides=2, padding=1),  # 256*16*16
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2DTranspose(channels=128, kernel_size=4, strides=2, padding=1),  # 128*32*32
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2DTranspose(channels=64, kernel_size=4, strides=2, padding=1),  # 64*64*64
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2DTranspose(channels=3, kernel_size=4, strides=2, padding=1),  # 3*128*128
            nn.Activation('sigmoid')
        )
        self.convs_4 = nn.Sequential()
        self.convs_4.add(
            nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1),  # 32*128*128
            nn.BatchNorm(),
            nn.Activation('relu')
        )
        self.convs_5 = nn.Sequential()
        self.convs_5.add(
            SRResNetBlock(32),
            SRResNetBlock(32),
            SRResNetBlock(32),
            nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1),  # 32*128*128
            nn.BatchNorm(),
            nn.Activation('relu')
        )
        self.convs_6 = nn.Sequential()
        self.convs_6.add(
            nn.Conv2D(channels=3, kernel_size=1, strides=1, padding=0),  # 3*128*128
            nn.Activation('sigmoid')
        )

    def forward(self, z, enc_x, enc_c):
        dec = nd.concat(z, enc_x, enc_c, dim=1)
        dec = self.seq(dec)  # [batch_size,1024]
        feature_1 = self.feature_1(dec)
        dec = dec.reshape(shape=(dec.shape[0], 1024, 1, 1))  # 1024*1*1
        dec = self.convs_1(dec)
        feature_2 = self.feature_2(dec)
        dec = self.convs_2(dec)
        feature_3 = self.feature_3(dec)
        y_rec_low = self.convs_3(dec)
        low_to_high_1 = self.convs_4(y_rec_low)
        low_to_high_2 = self.convs_5(low_to_high_1)
        y_rec_high = self.convs_6(low_to_high_1 + low_to_high_2)
        return y_rec_low, y_rec_high, feature_1, feature_2, feature_3


class farm_recommender(nn.Block):
    def __init__(self, **kwargs):
        super(farm_recommender, self).__init__(**kwargs)

        # feature_x1, feature_x2, feature_x3,feature_yp, feature_yn, feature_yg shape [使用的特征层数, batch_size, embeddinf_size]

    def forward(self, enc_x, feature_x1, feature_x2, feature_x3, enc_desc, enc_yp, enc_yn, feature_yp, feature_yn,feature_yg):
        score_yp = nd.sum(enc_yp * enc_x, axis=1) + nd.sum(enc_yp * enc_desc, axis=1)
        for i in range(len(feature_yp)):
            score_yp = score_yp + nd.sum(feature_yp[i] * feature_yg[i], axis=1)
            score_yp = score_yp + nd.sum(feature_yp[i] * feature_x1[i], axis=1)
            score_yp = score_yp + nd.sum(feature_yp[i] * feature_x2[i], axis=1)
            score_yp = score_yp + nd.sum(feature_yp[i] * feature_x3[i], axis=1)

        score_yn = nd.sum(enc_yn * enc_x, axis=1) + nd.sum(enc_yn * enc_desc, axis=1)
        for i in range(len(feature_yn)):
            score_yn = score_yn + nd.sum(feature_yn[i] * feature_yg[i], axis=1)
            score_yn = score_yn + nd.sum(feature_yn[i] * feature_x1[i], axis=1)
            score_yn = score_yn + nd.sum(feature_yn[i] * feature_x2[i], axis=1)
            score_yn = score_yn + nd.sum(feature_yn[i] * feature_x3[i], axis=1)
        difference_pn = score_yp - score_yn
        return difference_pn


class farm_predictor(nn.Block):
    def __init__(self, **kwargs):
        super(farm_predictor, self).__init__(**kwargs)
        self.layer1 = nn.Dense(30, activation='relu')
        self.layer2 = nn.Dense(1, activation='sigmoid')

        # feature_x1, feature_x2, feature_x3,feature_yp, feature_yn, feature_yg shape [使用的特征层数, batch_size, embedding_size]

    def forward(self, feature_x1, feature_x2, feature_x3, feature_yp, feature_yn):
        outfit_p = [feature_x1, feature_x2, feature_x3, feature_yp]  # (4,3,batch_size,embedding_size)
        outfit_n = [feature_x1, feature_x2, feature_x3, feature_yn]  # (4,3,batch_size,embedding_size)
        # Comparison matrix
        score_list_p = []
        score_list_n = []
        # 一共有10轮的循环 (i,j)->(0,0),(0,1),..,(1,1),(1,2),...(2,2),....,(3,3)
        for idx in range(len(feature_x1)):
            for mi, (i, j) in enumerate(itertools.combinations_with_replacement([0, 1, 2, 3], 2)):
                score_ij_p = outfit_p[i][idx] * outfit_p[j][idx]  # batch_size ,embedding_size (80,100)
                #                 print(f"score_ij_p1.shape{score_ij_p.shape}")
                score_ij_p = score_ij_p.sum(axis=-1, keepdims=True)  # batch_size ,1
                score_list_p.append(score_ij_p)

                score_ij_n = outfit_n[i][idx] * outfit_n[j][idx]  # batch_size ,embedding_size
                score_ij_n = score_ij_n.sum(axis=-1, keepdims=True)  # batch_size ,1
                score_list_n.append(score_ij_n)
        #         print(score_list_p)
        scores_p = nd.concat(*score_list_p)  # batch_size,30
        out_p = self.layer1(scores_p)
        out_p = self.layer2(out_p)  # batch_size,1

        scores_n = nd.concat(*score_list_n)  # batch_size,30
        out_n = self.layer1(scores_n)
        out_n = self.layer2(out_n)

        return out_p, out_n  # batch_size,1


class farm_layer_fusion(nn.Block):
    def __init__(self, **kwargs):
        super(farm_layer_fusion, self).__init__(**kwargs)
        self.low_to_medium = nn.Dense(100, activation='relu')
        self.medium_to_high = nn.Dense(100, activation='relu')
        self.layer1 = nn.Dense(30, activation='relu')
        self.layer2 = nn.Dense(1, activation='sigmoid')

        # feature_x1, feature_x2, feature_x3,feature_yp, feature_yn, feature_yg shape [使用的特征层数, batch_size, embedding_size]

    def forward(self, feature_x1, feature_x2, feature_x3, feature_y):
        low_feature = nd.concat(feature_x1[0], feature_x2[0], feature_x3[0], feature_y[0], dim=1)
        medium_orig = nd.concat(feature_x1[1], feature_x2[1], feature_x3[1], feature_y[1], dim=1)
        high_orig = nd.concat(feature_x1[2], feature_x2[2], feature_x3[2], feature_y[2], dim=1)

        medium_from_low = self.low_to_medium(low_feature)
        medium_new = nd.concat(medium_from_low, medium_orig, dim=1)
        high_from_medium = self.medium_to_high(medium_new)
        high_new = nd.concat(high_from_medium, high_orig, dim=1)

        out = self.layer1(high_new)
        out = self.layer2(out)  # batch_size,1

        return out  # batch_size,1

class feature_fusion(nn.Block):
    def __init__(self, **kwargs):
        super(feature_fusion, self).__init__(**kwargs)
        self.low_to_medium = nn.Dense(100, activation='relu')
        self.medium_to_high = nn.Dense(100, activation='relu')
        self.layer1 = nn.Dense(100, activation='sigmoid')
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, feature_x1, feature_x2, feature_x3):
        low_feature = nd.concat(feature_x1[0], feature_x2[0], feature_x3[0], dim=1)
        medium_orig = nd.concat(feature_x1[1], feature_x2[1], feature_x3[1], dim=1)
        high_orig = nd.concat(feature_x1[2], feature_x2[2], feature_x3[2], dim=1)

        medium_from_low = self.bn1(self.low_to_medium(low_feature))
        medium_new = nd.concat(medium_from_low, medium_orig, dim=1)
        high_from_medium = self.bn2(self.medium_to_high(medium_new))
        high_new = nd.concat(high_from_medium, high_orig, dim=1)

        out = self.layer1(high_new)
        return out  # 保持维度统一，都是100维


class farm(nn.Block):
    def __init__(self, **kwargs):
        super(farm, self).__init__(**kwargs)
        self.description_embedding = nn.Dense(100, activation='sigmoid')
        self.encoder = farm_encoder()
        self.transformer = farm_transformer()
        self.generator = farm_generator()
        self.recommender = farm_recommender()
        self.feature_fusion = feature_fusion()
        #         self.farm_predictor = farm_predictor()
        self.farm_layer_fusion = farm_layer_fusion()

    def forward(self, x_img1, x_img2, x_img3, y_img, negative_img, y_desc, y_desc_drop, phase):
        enc_desc = self.description_embedding(y_desc)
        enc_desc_drop = self.description_embedding(y_desc_drop)
        enc_x1, *feature_x1 = self.encoder(x_img1)  # enc_x_*,(batch_size, 100)
        enc_x2, *feature_x2 = self.encoder(x_img2)
        enc_x3, *feature_x3 = self.encoder(x_img3)
        enc_yp, *feature_yp = self.encoder(y_img)
        enc_yn, *feature_yn = self.encoder(negative_img)
        #         out_p,out_n = self.farm_predictor(feature_x1, feature_x2, feature_x3,feature_yp, feature_yn)
        out_p = self.farm_layer_fusion(feature_x1, feature_x2, feature_x3, feature_yp)
        out_n = self.farm_layer_fusion(feature_x1, feature_x2, feature_x3, feature_yn)
        enc_x = self.feature_fusion(feature_x1, feature_x2, feature_x3)
        z, z_mean, z_log_var = self.transformer(enc_x, enc_desc_drop)
        if phase == 'train':
            y_rec_low, y_rec_high, *feature_yg = self.generator(z, enc_x,
                                                                enc_desc_drop)  # feature_yg =[feature_1_yg, feature_2_yg, feature_3_yg]
        if phase == 'test':
            y_rec_low, y_rec_high, *feature_yg = self.generator(z_mean, enc_x,
                                                                enc_desc_drop)  # feature_yg =[feature_1_yg, feature_2_yg, feature_3_yg]
        difference_pn = self.recommender(enc_x, feature_x1, feature_x2, feature_x3, enc_desc, enc_yp, enc_yn,
                                         feature_yp, feature_yn, feature_yg)
        return y_rec_low, y_rec_high, difference_pn, z_mean, z_log_var, out_p, out_n  # , feature_1_yp, feature_2_yp, feature_3_yp, feature_1_yn, feature_2_yn, feature_3_yn, feature_1_yg, feature_2_yg, feature_3_yg