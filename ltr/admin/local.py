class EnvironmentSettings:
    def __init__(self):
        # self.workspace_dir = '/home/dy/Desktop/genggu/pytracking-master/ltr/checkpoints'    # Base directory for saving network checkpoints.
        self.workspace_dir = '/media/dy/ext4/a_genggu/checkpoint'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.pregenerated_masks = ''
        self.lasot_dir = '/media/dy/ext4/yuandi/lasot'
        self.lasotlmdb_dir = '/media/dy/sda2/dataset/lasot_lmdb'
        self.got10k_dir = '/media/dy/ext4/yuandi/Got10k/train'
        self.got10klmdb_dir = '/media/dy/sda2/dataset/Got10k_lmdb'
        self.trackingnet_dir = '/media/dy/Elements2/yuandi/TrackingNet-devkit-master/TrackingNet'
        self.trackingnetlmdb_dir = '/media/dy/sda2/dataset/trackingnet_lmdb'
        self.coco_dir = '/media/dy/ext4/yuandi/COCO2017'
        self.cocolmdb_dir = '/media/dy/sda2/dataset/COCO2017_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenetlmdb_dir = '/media/dy/sda2/dataset/ILSVRC2015_VID_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = '/media/dy/ext4/yuandi/VOS'
        self.lasot_candidate_matching_dataset_path = ''
