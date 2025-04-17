import torch
import torch.nn as nn
import torchvision.ops as ops
from collections import OrderedDict
import ltr.models.layers.filter as filter_layer
import math
import numpy as np
from ltr.models.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


def conv_layer(inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1):
    layers = [
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.GroupNorm(1, outplanes),
        nn.ReLU(inplace=True),
    ]
    return layers

class Head(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,global_mutual=None,
                 separate_filters_for_cls_and_bbreg=False):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg
        self.feature_sz = self.filter_predictor.feature_sz
        # self.global_mutual = global_mutual
        # self.prroi_rool = FeaturePool(pooled_height=9,pooled_width=9,spatial_scale=1/16)

    def forward_(self, train_feat, test_feat, train_bb,test_bb, *args, **kwargs):
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        # Train filter
        cls_filter, breg_filter, test_feat_enc,st_feat = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        # 互信息损失计算
        num_frame = train_bb.shape[0]+test_bb.shape[0]
        st_feats = list(st_feat.chunk(num_frame,dim=2))
        st_bboxs = list(train_bb.chunk(num_frame-1,dim=0))
        st_bboxs.append(test_bb)
        mutual_loss_sz1 = self.compute_mutual_loss(t_feat=st_feats[0],s_feat=st_feats[num_frame-1],
                                                   t_bbox=st_bboxs[0],s_bbox=st_bboxs[num_frame-1])
        mutual_loss_sz2 = self.compute_mutual_loss(t_feat=st_feats[1],s_feat=st_feats[num_frame-1],
                                                   t_bbox=st_bboxs[1],s_bbox=st_bboxs[num_frame-1])
        mutual_loss_z1z2 = self.compute_mutual_loss(t_feat=st_feats[0],s_feat=st_feats[1],
                                                    t_bbox=st_bboxs[0],s_bbox=st_bboxs[1])
        mutual_loss = mutual_loss_sz1+mutual_loss_sz2+mutual_loss_z1z2
        # mutual_loss = torch.tensor(0,device='cuda')

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

        return target_scores, bbox_preds,mutual_loss

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        # Train filter
        cls_filter, breg_filter, test_feat_enc,time_token = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

        return target_scores, bbox_preds,time_token

    def forward_ori(self, train_feat, test_feat, train_bb, *args, **kwargs):
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

        return target_scores, bbox_preds

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):
        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).

        weights, test_feat_enc,timetoken = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        cls_weights = bbreg_weights = weights

        return cls_weights, bbreg_weights, test_feat_enc,timetoken

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc,time_token,attn_enc\
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc,time_token,attn_enc

    def compute_mutual_loss(self,t_feat,s_feat,t_bbox,s_bbox):
        """
            feats: output embeddings of the backbone, it can be list[(B, HW1, C),( B, HW2, C)],第一个是模板帧
            bboxs: [(B,4),(B,4)]
        """

        B, C = t_feat.shape[:2]
        sz = self.feature_sz
        template_anno = t_bbox[0]
        search_anno = s_bbox[0]
        opt_feat_z = t_feat.reshape(B,C,self.feature_sz,self.feature_sz)
        opt_feat_s = s_feat.reshape(B,C,self.feature_sz,self.feature_sz)

        x1,y1,w1,h1 = template_anno.transpose(0,1)
        x2,y2 = (((x1+w1)/16).ceil()).clip(max=sz-1),(((y1+h1)/16).ceil()).clip(max=sz-1)
        x1,y1 = ((x1/16).floor()).clip(min=0),((y1/16).floor()).clip(min=0)
        template_anno = torch.stack([x1,y1,x2-x1,y2-y1],dim=1)

        x1, y1, w1, h1 = search_anno.transpose(0, 1)
        x2, y2 = (((x1 + w1) / 16).ceil()).clip(max=sz-1), (((y1 + h1) / 16).ceil()).clip(max=sz-1)
        x1, y1 = ((x1 / 16).floor()).clip(min=0), ((y1 / 16).floor()).clip(min=0)
        search_anno = torch.stack([x1, y1, x2 - x1, y2 - y1], dim=1)

        # opt_feat_mask = torch.zeros_like(opt_feat)
        # opt_feat_x = torch.zeros_like(opt_feat)
        # for i in range(B):
        #     # B is batchsize,C is channel
        #     # op_feat:  (B,C,18,18)
        #     # op_feat_x:  (B,C,18,18)
        #     # anno:     (B,4)
        #
        #     x_t,y_t,w_t,h_t = template_anno[i].cpu().numpy().astype(np.int32)
        #     x_s,y_s,w_s,h_s = search_anno[i].cpu().numpy().astype(np.int32)
        #
        #     h = min([h_t, h_s])
        #     w = min([w_t,w_s])
        #     # 为什么要将模板与搜索区域的token放到相同的位置
        #     opt_feat_x[i, :, y_t:y_t + h, x_t:x_t + w] = opt_feat[i, :, y_s:y_s + h, x_s:x_s + w]
        #     opt_feat_mask[i, :, y_t:y_t + h, x_t:x_t + w] = 1
        #
        # opt_feat_z = opt_feat_z * opt_feat_mask.to(opt_feat_z.device)
        # # z_feat_inter = nn.functional.interpolate(opt_feat_z[0].unsqueeze(0),scale_factor=16,mode='nearest').squeeze(0).mean(dim=0).cpu().detach()

        x = self.prroi_rool(opt_feat_z,template_anno)
        y = self.prroi_rool(opt_feat_s,search_anno)

        x_shuffled = torch.cat([x[1:], x[0].unsqueeze(0)], dim=0)

        # Global mutual information estimation
        global_mutual_M_R_x = self.global_mutual(x, y)  # positive statistic
        global_mutual_M_R_x_prime = self.global_mutual(x_shuffled, y)
        loss = -torch.mean(torch.log(global_mutual_M_R_x+1e-6)+torch.log(1-global_mutual_M_R_x_prime+1e-6))
        return loss

class LinearFilterClassifier(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, feat, filter):
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter
        return filter_layer.apply_filter(feat, filter_proj)


class DenseBoxRegressor(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

        layers = []
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        self.tower = nn.Sequential(*layers)

        self.bbreg_layer = nn.Conv2d(num_channels, 4, kernel_size=3, dilation=1, padding=1)

    def forward(self, feat, filter):
        nf, ns, c, h, w = feat.shape

        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, c)).reshape(filter.shape)
        else:
            filter_proj = filter

        attention = filter_layer.apply_filter(feat, filter_proj) # (nf, ns, h, w)
        if attention.dim() == 4:
            feats_att = attention.unsqueeze(2) * feat  # (nf, ns, c, h, w)
        else:
            feats_att = feat.unsqueeze(2) * attention.unsqueeze(3)

        feats_tower = self.tower(feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1])) # (nf*ns, c, h, w)

        ltrb = torch.exp(self.bbreg_layer(feats_tower)).unsqueeze(0)  # (1, nf*ns, 4, h, w)

        if attention.dim() == 5:
            ltrb = ltrb.reshape(1, feats_att.shape[1], feats_att.shape[2], 4, feats_att.shape[4],
                                feats_att.shape[5])  # (1, nf*ns, num_obj, 4, h, w)

        return ltrb

class Head_trans(nn.Module):
    """
    """
    def __init__(self, filter_predictor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

    def forward(self, feat, *args, **kwargs):
        # assert train_bb.dim() == 3
        #
        # if train_feat.dim() == 5:
        #     train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        # if test_feat.dim() == 5:
        #     test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(feat, *args, **kwargs)

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

        return target_scores, bbox_preds

    def get_filter_and_features(self, feat, *args, **kwargs):
        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(feat, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(feat, *args, **kwargs)
            cls_weights = bbreg_weights = weights

        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, feat, num_gth_frames, *args, **kwargs):
        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(feat, num_gth_frames, *args, **kwargs)

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc


class FeaturePool(nn.Module):
    def __init__(self,pooled_height,pooled_width,spatial_scale,pool_square=False):
        super(FeaturePool,self).__init__()
        self.prroi_pool=PrRoIPool2D(pooled_height=pooled_height,pooled_width=pooled_width,spatial_scale=spatial_scale)
        self.pool_square = pool_square

    def forward(self, feat,bb):
        """Pool the regions in bb.
            args:
                feat:  Input feature maps. Dims (num_samples, feat_dim, H, W).
                bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (num_samples, 4).
            returns:
                pooled_feat:  Pooled features. Dims (num_samples, feat_dim, wH, wW)."""

        #add batch_index to rois
        bb=bb.reshape(-1,4)
        num_images=bb.shape[0]
        batch_index=torch.arange(num_images,dtype=torch.float32).reshape(-1,1).to(bb.device)

        #input bb is in format xywh,convert it to x0y0x1y1 format
        pool_bb=bb.clone()

        if self.pool_square:
            bb_sz=pool_bb[:,2:4].prod(dim=1,keepdim=True).sqrt()
            pool_bb[:,0:2]=pool_bb[:,2:4]/2-bb_sz/2
            pool_bb[:,2:4]=bb_sz

        pool_bb[:,2:4]=pool_bb[:,0:2]+pool_bb[:,:2:4]
        roi1=torch.cat((batch_index,pool_bb),dim=1)

        return self.prroi_pool(feat,roi1)