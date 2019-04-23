"""
Configuration defaults specification file
"""

from easydict import EasyDict


def get_default():
    # Project Setup
    config = EasyDict()
    config.project = 'BicycleDetection'
    config.seed = 233
    config.gpus = '0'
    config.num_workers = 1

    # Shared Defaults
    config.run_type = 'detection'
    config.run_id = None
    config.classes = ['cyclist']
    config.resume = -1

    # Model Defaults
    config.model = EasyDict()
    config.model.root_dir = 'models'
    config.model.type = 'ssd_512'
    config.model.id = None
    config.model.pretrained_on = 'coco'

    config.model.backbone = EasyDict()
    config.model.backbone.type = 'resnet50_v1'  # Base network name which serves as feature extraction base.
    config.model.backbone.n_layers = 50

    # Dataset Defaults
    config.data = EasyDict()
    config.data.root_dir = 'data'
    config.data.name = 'cycle'
    config.data.split_id = None

    config.data.shape = 512

    config.data.mixup = False
    config.data.no_mixup_epochs = 20

    # Train Defaults
    config.train = EasyDict()

    config.train.batch_size = 1
    config.train.checkpoint_every = 1  # epochs, 0 is never

    config.train.epochs = None  # voc 20, coco 26

    config.train.learning_rate = 0.001  # voc 0.001, 0.00125*#gpus coco
    config.train.lr_decay = 0.1
    config.train.lr_decay_epochs = ''  # epochs voc [14,20], coco [17,23]
    config.train.lr_warmup = None  # voc -1, coco -1 for 1 gpu, 8000/#gpus
    config.train.momentum = 0.9

    config.train.weight_decay = 0.0005  # voc 0.0005, coco 0.0001 # Weight decay, for regularization

    config.train.resume = -1  # resume? if -1 then auto search dir for latest checkpoint, otherwise specify of None

    config.train.val_every = 1  # epochs, 0 is never

    # Visualisation Defaults
    config.vis = EasyDict()
    config.vis.every = 1000  # minibatches .. 0 is never

    config.log_interval = 100

    config.checkpoint_every = 1

    return config
