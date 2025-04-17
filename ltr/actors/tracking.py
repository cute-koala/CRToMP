import random
import cv2
import numpy as np
from . import BaseActor
import torch
import matplotlib.pyplot as plt
import os
from pytracking import dcf
import torch.nn.functional as F
import ltr.data.processing_utils as prutils


class ToMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight


    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g) # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def get_template_label(self,test_feat,test_label,test_ltrb_target):
        test_feat_seq = test_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)  # Nf_te*H*W, Ns, C
        test_label_seq = test_label.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2)  # Nf_tr*H*W,Ns,1
        test_ltrb_target_seq_T = test_ltrb_target.permute(1, 2, 0, 3, 4).flatten(2)  # Ns,4,Nf_tr*H*W

        fg_token = self.net.head.filter_predictor.query_embed_fg.weight.reshape(1, 1, -1).detach()
        train_label_enc = fg_token * test_label_seq

        train_ltrb_target_enc = self.net.head.filter_predictor.box_encoding(test_ltrb_target_seq_T).permute(2, 0, 1).detach()  # Nf_tr*H*H,Ns,C
        test_feat_label = test_feat_seq + train_label_enc + train_ltrb_target_enc
        return test_feat_label.detach()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """

        # Run network
        # target_scores, bbox_preds = self.net(train_imgs=data['train_images'],
        #                                      test_imgs=data['test_images'],
        #                                      train_bb=data['train_anno'],
        #                                      test_bb = data['test_anno'],
        #                                      train_label=data['train_label'],
        #                                      train_ltrb_target=data['train_ltrb_target'])
        # # 可视化
        # self.visualize(train_img=data['train_images'],
        #                test_img = data['test_images'],
        #                train_box= data['train_anno'],
        #                test_box=bbox_preds,
        #                target_scores=target_scores )
        # print(target_scores[0,0])
        # normal
        # loss_giou,ious = self.objective['giou'](bbox_preds,data['test_ltrb_target'],data['test_sample_region'])
        #
        # # Classification losses for the different optimization iterations
        # clf_loss_test = self.objective['test_clf'](target_scores,data['test_label'],data['test_anno'])
        #
        # loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test+mutual_loss
        #
        # if torch.isnan(loss):
        #     raise ValueError('NaN detected in loss')
        #
        # ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], bbox_preds)
        #
        # stats = {'Loss/total': loss.item(),
        #          'Loss/GIoU': loss_giou.item(),
        #          'Loss/Mutual_loss': mutual_loss.item(),
        #          'Loss/weighted_clf_loss_test': self.loss_weight['test_clf']*clf_loss_test.mean().item(),
        #          'ious_pred_center':ious_pred_center.mean().item()}
        # return loss, stats

        # RCR
        # 原本的损失
        # loss_giou_, _ = self.objective['giou2'](bbox_preds.detach(), data['test_ltrb_target'],data['test_sample_region'])
        # clf_loss_test_ = self.objective['test_clf'](target_scores.detach(), data['test_label'], data['test_anno']).mean()
        # loss_ = self.loss_weight['giou'] * loss_giou_ + self.loss_weight['test_clf'] * clf_loss_test_
        #
        # loss_giou,ious = self.objective['giou'](bbox_preds,data['test_ltrb_target'],data['test_sample_region'])
        # # Classification losses for the different optimization iterations
        # clf_loss_test = self.objective['test_clf'](target_scores,data['test_label'],data['test_anno'])
        #
        # loss_reg = ((1-loss_giou)*data['test_sample_region']).squeeze(2) * (1+target_scores.clip(min=0))
        # loss_reg  = loss_reg.sum()/data['test_sample_region'].sum()
        # loss_cls = clf_loss_test * (1+(loss_giou).squeeze(2).clip(min=0))
        # loss_cls = loss_cls.mean()
        #
        # loss = self.loss_weight['giou'] * loss_reg + self.loss_weight['test_clf'] * loss_cls
        #
        # if torch.isnan(loss):
        #     raise ValueError('NaN detected in loss')
        #
        # ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], bbox_preds)
        #
        # stats = {'Loss/total': loss_.item(),
        #          'Loss/GIoU': loss_giou_.item(),
        #          'Loss/weighted_clf_loss': self.loss_weight['test_clf'] * clf_loss_test_.item(),
        #          'ious_pred_center':ious_pred_center.mean().item(),
        #          }
        # return loss, stats

        #              f'{i}_frame_Loss/GIoU': loss_giou_.item(),
        #              f'{i}_frame_Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test_.item(),
        #              f'{i}_frame_Loss/weighted_clf_loss_template': self.loss_weight['test_clf'] * clf_loss_template.item(),

        out = self.net(train_imgs=data['train_images'],
                        test_imgs=data['test_images'],
                        train_bb=data['train_anno'],
                        train_label=data['train_label'],
                        train_ltrb_target=data['train_ltrb_target'])


        total_stats = {}
        total_loss = torch.tensor(0.,dtype=torch.float).cuda()
        print_loss = 0
        for i in range(len(out)):
            bbox_preds = out[i]['bbox_preds']
            target_scores = out[i]['test_scores']
            # 原本的损失
            loss_giou_, _ = self.objective['giou2'](bbox_preds.detach(), data['test_ltrb_target'][[i]],data['test_sample_region'][[i]])
            clf_loss_test_ = self.objective['test_clf'](target_scores.detach(), data['test_label'][[i]], data['test_anno'][[i]]).mean()
            loss_ = self.loss_weight['giou'] * loss_giou_ + self.loss_weight['test_clf'] * clf_loss_test_

            # new
            loss_giou,ious = self.objective['giou'](bbox_preds,data['test_ltrb_target'][i:i+1],data['test_sample_region'][i:i+1])
            clf_loss_test = self.objective['test_clf'](target_scores,data['test_label'][i:i+1],data['test_anno'][i:i+1])

            loss_reg = ((1-loss_giou)*data['test_sample_region'][i:i+1]).squeeze(2) * (1+target_scores.clip(min=0))
            loss_reg  = loss_reg.sum()/data['test_sample_region'][i:i+1].sum()
            loss_cls = clf_loss_test * (1+(loss_giou).squeeze(2).clip(min=0))
            loss_cls = loss_cls.mean()

            loss = self.loss_weight['giou'] * loss_reg + self.loss_weight['test_clf'] * loss_cls

            if torch.isnan(loss):
                raise ValueError('NaN detected in loss')

            total_loss += loss
            print_loss += loss.item()
            ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'][[i]],bbox_preds)

            stats = {f'{i}_frame_Loss/total':loss_.item(),
                     f'{i}_frame_Loss/GIoU': loss_giou_.item(),
                     f'{i}_frame_Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test_.item(),
                     f'{i}_frame_mIoU_pred_center': ious_pred_center.mean().item()}
            total_stats.update(stats)
        total_stats.update({'rcr_loss': print_loss/ len(out)})

        return total_loss, total_stats