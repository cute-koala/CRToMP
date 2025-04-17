import torch.optim as optim
from collections import OrderedDict
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq,ImagenetVID
from ltr.dataset import Lasot_lmdb, Got10k_lmdb, TrackingNet_lmdb, MSCOCOSeq_lmdb,ImagenetVID_lmdb
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import tompnet
import ltr.models.loss as ltr_losses
import ltr.actors.tracking as actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.models.loss.bbr_loss import GIoULoss_new,GIoULoss
from ltr.scheduler import CosineLRScheduler
import torch.nn as nn
import torch
torch._dynamo.config.suppress_errors=True
torch.set_float32_matmul_precision('high')
# TORCH_LOGS="+dynamo"
# TORCHDYNAMO_VERBOSE=1

def run(settings):
    settings.description = 'ToMP50'
    settings.batch_size = 20 #36
    settings.num_workers = 16 #16
    settings.multi_gpu = False

    settings.print_interval = 10
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 1
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 0., 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0., 'test': 0.5}
    settings.hinge_threshold = 0.05
    settings.num_train_frames =1 #1
    settings.num_test_frames =2 #3
    settings.num_encoder_layers = 6
    settings.num_decoder_layers = 6
    # settings.frozen_backbone_layers = ['conv1', 'bn1', 'layer1', 'layer2']
    settings.frozen_backbone_layers = [0,1]
    settings.freeze_backbone_bn_layers = True

    settings.crop_type = 'inside_major'
    settings.max_scale_change = 1.5
    settings.max_gap = 200 #200
    settings.train_samples_per_epoch = 40000 # 40000
    settings.val_samples_per_epoch = 10000 # 10000
    settings.val_epoch_interval = 3
    settings.num_epochs = 300 #372

    settings.weight_giou = 1.0
    settings.weight_clf = 100.0
    settings.normalized_bbreg_coords = True
    settings.center_sampling_radius = 1.0
    settings.use_test_frame_encoding = False  # Set to True to use the same as in the paper but is less stable to train.

    ''' Train datasets '''
    # lasot_train = Lasot(settings.env.lasot_dir, split='train')
    # got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    # coco_train = MSCOCOSeq(settings.env.coco_dir,version='2017')

    ''' lmdb data '''
    lasot_train = Lasot_lmdb(settings.env.lasotlmdb_dir, split='train')
    got10klmdb_train = Got10k_lmdb(settings.env.got10klmdb_dir, split='vottrain')
    trackingnet_train = TrackingNet_lmdb(settings.env.trackingnetlmdb_dir, set_ids=list(range(7)))
    # cocolmdb_train = MSCOCOSeq_lmdb(settings.env.cocolmdb_dir, version='2017')
    imagenetlmdb_vid = ImagenetVID_lmdb(settings.env.imagenetlmdb_dir)


    # Validation datasets
    got10k_val = Got10k_lmdb(settings.env.got10klmdb_dir, split='votval')


    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5)
                                    )

    transform_train = tfm.Transform(
                                    tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)
                                    )

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}

    data_processing_train = processing.LTRBDenseRegressionProcessing(search_area_factor=settings.search_area_factor,
                                                                     output_sz=settings.output_sz,
                                                                     center_jitter_factor=settings.center_jitter_factor,
                                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                                     crop_type=settings.crop_type,
                                                                     max_scale_change=settings.max_scale_change,
                                                                     mode='sequence',
                                                                     label_function_params=label_params,
                                                                     transform=transform_train,
                                                                     joint_transform=transform_joint,
                                                                     use_normalized_coords=settings.normalized_bbreg_coords,
                                                                     center_sampling_radius=settings.center_sampling_radius)

    data_processing_val = processing.LTRBDenseRegressionProcessing(search_area_factor=settings.search_area_factor,
                                                                   output_sz=settings.output_sz,
                                                                   center_jitter_factor=settings.center_jitter_factor,
                                                                   scale_jitter_factor=settings.scale_jitter_factor,
                                                                   crop_type=settings.crop_type,
                                                                   max_scale_change=settings.max_scale_change,
                                                                   mode='sequence',
                                                                   label_function_params=label_params,
                                                                   transform=transform_val,
                                                                   joint_transform=transform_joint,
                                                                   use_normalized_coords=settings.normalized_bbreg_coords,
                                                                   center_sampling_radius=settings.center_sampling_radius)

    # Train sampler and loader
    # dataset_train = sampler.DiMPSampler([lasot_train, got10k_train,trackingnet_train,coco_train], [1, 1,1,1],
    #                                     samples_per_epoch=settings.train_samples_per_epoch, max_gap=settings.max_gap,
    #                                     num_test_frames=settings.num_test_frames, num_train_frames=settings.num_train_frames,
    #                                     processing=data_processing_train)
    dataset_train = sampler.DiMPSampler([got10klmdb_train,trackingnet_train,lasot_train,imagenetlmdb_vid], [1,1,1,1],
                                        samples_per_epoch=settings.train_samples_per_epoch, max_gap=settings.max_gap,
                                        num_test_frames=settings.num_test_frames,
                                        num_train_frames=settings.num_train_frames,
                                        processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    dataset_val = sampler.DiMPSampler([got10k_val], [1], samples_per_epoch=settings.val_samples_per_epoch,
                                      max_gap=settings.max_gap, num_test_frames=settings.num_test_frames,
                                      num_train_frames=settings.num_train_frames, processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=settings.val_epoch_interval, stack_dim=1)

    # Create network and actor
    # net = tompnet.tompnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True, head_feat_blocks=0,
    #                         head_feat_norm=True, final_conv=True, input_feature_dim=1024,out_feature_dim=256, feature_sz=settings.feature_sz,
    #                         frozen_backbone_layers=settings.frozen_backbone_layers,
    #                         num_encoder_layers=settings.num_encoder_layers,
    #                         num_decoder_layers=settings.num_decoder_layers,
    #                         use_test_frame_encoding=settings.use_test_frame_encoding)
    net = tompnet.tompnet_convnext(
                                filter_size=settings.target_filter_sz, backbone_pretrained=True, head_feat_blocks=0,
                                head_feat_norm=True, final_conv=True, input_feature_dim=512,out_feature_dim=256, feature_sz=settings.feature_sz,
                                frozen_backbone_layers=settings.frozen_backbone_layers,
                                num_encoder_layers=settings.num_encoder_layers,
                                num_decoder_layers=settings.num_decoder_layers,
                                use_test_frame_encoding=settings.use_test_frame_encoding)
    # init_weight = torch.load(r'/media/dy/ext4/a_genggu/checkpoint/checkpoints/ltr/tomp/tomp50_online_template/OptimizedModule_ep0150.pth.tar')['net'] # timetoken pretrain
    init_weight = torch.load(r'/media/dy/ext4/Project/pytracking/pytracking/networks/tomp50.pth.tar')['net']
    for k in list(init_weight.keys()):
        if 'feature_extractor' in k:
            init_weight.pop(k)
    # for k in list(init_weight.keys()):
    #     init_weight[k.replace('_orig_mod.','')]=init_weight.pop(k)
    miss, unexpected = net.load_state_dict(init_weight, strict=False)
    print("miss: ", miss)
    print("unexpected: ", unexpected)
    print(sum(p.numel() for p in net.parameters()))
    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    net = torch.compile(net)
    objective = {'giou': GIoULoss_new(),'giou2':GIoULoss() ,'test_clf': ltr_losses.LBHinge(error_metric=nn.MSELoss(reduction='none'),threshold=settings.hinge_threshold)}
    # objective = {'giou':GIoULoss() ,'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold)}

    loss_weight = {'giou': settings.weight_giou, 'test_clf': settings.weight_clf}

    actor = actors.ToMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    optimizer = optim.AdamW([
        {'params': actor.net.head.parameters(), 'lr': 1e-4}, # 1e-4
        {'params': actor.net.feature_extractor.parameters(), 'lr': 1e-4},  # 2e-5
        # {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 4e-6},  # 2e-5
        # {'params': actor.net.head.classifier.parameters(), 'lr': 2e-5},  # 1e-4
        # {'params': actor.net.head.bb_regressor.parameters(), 'lr': 2e-5},  # 2e-5
    ], lr=2e-4, weight_decay=0.0001)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,250], gamma=0.2)#[25,50]

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler,
                         freeze_backbone_bn_layers=settings.freeze_backbone_bn_layers)

    trainer.train(settings.num_epochs, load_latest=True, fail_safe=False)
