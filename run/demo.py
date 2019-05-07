

from gluoncv.data.transforms import bbox as tbbox
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt


from config.functions import load_config
from data_processing.loading import load_datasets, transform_test

cfg = load_config('/media/hayden/UStorage/CODE/BicycleDetection/configs/001.yaml')

net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

train_dataset, val_dataset, test_dataset = load_datasets(cfg.data.root_dir, cfg.data.split_id, cfg.classes)

for i, sample in enumerate(test_dataset):
    x, img = transform_test(sample[0], short=512)

    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxes = net(x)

    oh, ow, _ = img.shape
    bounding_boxes[0] = tbbox.resize(bounding_boxes[0], in_size=(512, 512), out_size=(ow, oh))

    ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                             class_IDs[0], class_names=net.classes)
    plt.show()