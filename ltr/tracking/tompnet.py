import math
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.features as clf_features
import ltr.models.backbone as backbones
from ltr import model_constructor
import torch
import ltr.models.transformer.transformer as trans
import ltr.models.transformer.filter_predictor as fp
import ltr.models.transformer.heads as heads

class Mutual_Info(nn.Module):
    def __init__(self,feature_sz,channel):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=(feature_sz ** 2) * channel * 2, out_features=512)
        # self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,z):
        x = self.flatten(x)
        z = self.flatten(z)
        xx = torch.cat([x,z],dim=1)
        xx = self.relu(self.dense1(xx))
        xx = self.relu(self.dense2(xx))
        # xx = self.relu(self.dense3(xx))
        xx = self.sigmoid(self.dense3(xx))
        return xx

class ToMPnet(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))


    # def forward(self, train_imgs, test_imgs, train_bb, test_bb,*args, **kwargs):
    #     """Runs the ToMP network the way it is applied during training.
    #     The forward function is ONLY used for training. Call the individual functions during tracking.
    #     args:
    #         train_imgs:  Train image samples (images, sequences, 3, H, W).
    #         test_imgs:  Test image samples (images, sequences, 3, H, W).
    #         trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
    #         *args, **kwargs:  These are passed to the classifier module.
    #     returns:
    #         test_scores:  Classification scores on the test samples.
    #         bbox_preds:  Predicted bounding box offsets."""
    #
    #     assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
    #
    #     # Extract backbone features(特征提取网络的特征)
    #     train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
    #     test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))
    #
    #     # Classification features（）
    #     train_feat_head = self.get_backbone_head_feat(train_feat)
    #     test_feat_head = self.get_backbone_head_feat(test_feat)
    #
    #     # Run head module
    #     # test_scores, bbox_preds,mutual_loss = self.head(train_feat_head, test_feat_head, train_bb,test_bb, *args, **kwargs)
    #     # test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb,test_bb, *args, **kwargs)
    #     # test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb, *args, **kwargs)
    #     test_scores, bbox_preds,_ = self.head(train_feat_head, test_feat_head, train_bb,time_token=None, *args, **kwargs)
    #
    #     # return test_scores, bbox_preds,mutual_loss
    #     return test_scores, bbox_preds

    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features(特征提取网络的特征)
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features（）
        train_feat_head = self.get_backbone_head_feat(train_feat)
        test_feat_head = self.get_backbone_head_feat(test_feat)

        test_feat_head = test_feat_head.reshape(*test_imgs.shape[:2],*test_feat_head.shape[1:])
        time_token = None
        out_dict=[]
        for i in range(len(test_imgs)):
            out={}
            # test_scores,bbox_preds,time_token,template_feat = self.head(train_feat_head, test_feat_head[i], train_bb,time_token=time_token, *args, **kwargs)
            if time_token is not None:
                test_scores,bbox_preds,time_token = self.head(train_feat_head, test_feat_head[i], train_bb,time_token=time_token.detach(), *args, **kwargs)
            else:
                test_scores, bbox_preds, time_token = self.head(train_feat_head, test_feat_head[i], train_bb,time_token=time_token, *args, **kwargs)
            out['test_scores'] = test_scores
            out['bbox_preds'] = bbox_preds
            # out['template_feature'] = template_feat.detach()
            # out['time_token'] = time_token
            out_dict.append(out)
        return out_dict
        # Run head module
        # test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb, *args, **kwargs)
        # return test_scores, bbox_preds


    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]
        return feat

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def tompnet50(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
              final_conv=True, input_feature_dim=1024,out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
              num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    # backbone_net = backbones.convnextv2_base(pretrained=backbone_pretrained,frozen_layers=frozen_backbone_layers)
    # backbone_net = backbones.convnextv2_tiny(pretrained=backbone_pretrained,frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              input_dim=input_feature_dim,
                                                              out_dim=out_feature_dim)
    # 原始版本
    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                    return_intermediate_dec=True,return_intermediate_enc=True)
    # 修改版本
    # transformer = trans.Transformer_mod(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
    #                                 num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
    #                                     c_model=feature_sz*feature_sz,c_layers=3,c_nhead=9)
    # transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
    #                                 num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
    #                                 return_intermediate_dec=True,return_intermediate_enc=True)


    filter_predictor = fp.FilterPredictor_timetoken(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)
    # filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
    #                                                 use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)
    #
    # global_mutual = Mutual_Info(feature_sz=9,channel=out_feature_dim)

    # head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
    #                   classifier=classifier, bb_regressor=bb_regressor,global_mutual=global_mutual)
    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)
    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net

@model_constructor
def tompnet_convnext(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
              final_conv=True, input_feature_dim=512,out_feature_dim=256, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
              num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    # backbone_net = backbones.convnextv2_base(pretrained=backbone_pretrained,frozen_layers=frozen_backbone_layers)
    backbone_net = backbones.convnextv2_base_adapter(pretrained=backbone_pretrained,frozen_layers=frozen_backbone_layers)
    # backbone_net = backbones.convnextv2_tiny(pretrained=backbone_pretrained,frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              input_dim=input_feature_dim,
                                                              out_dim=out_feature_dim)
    # 原始版本
    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                    return_intermediate_dec=True, return_intermediate_enc=True)

    filter_predictor = fp.FilterPredictor_timetoken(transformer, feature_sz=feature_sz,
                                                    use_test_frame_encoding=use_test_frame_encoding)
    # filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
    #                                                 use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)
    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net
