import argparse
import fnmatch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

import torch
import glob
from torch.utils.model_zoo import load_url
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from solar_global.networks.imageretrievalnet import init_network, extract_vectors
from solar_global.datasets.testdataset import configdataset
from solar_global.utils.download import download_test
from solar_global.utils.evaluate import compute_map_and_print
from solar_global.utils.general import get_data_root, htime
from solar_global.utils.networks import load_network
from solar_global.utils.plots import plot_ranks, plot_embeddings

# some conflicts between tensorflow and tensoboard 
# causing embeddings to not be saved properly in tb
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except:
    pass


datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', 'revisitop1m', 'gl18']

# test options
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Example')
parser.add_argument('--network', '-n', metavar='NETWORK', default='resnet101-solar-best.pth', 
                    help="network to be evaluated. " )
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='roxford5k,rparis6k',
                    help="comma separated list of test datasets: " +
                        " | ".join(datasets_names) +
                        " (default: 'roxford5k,rparis6k')")
parser.add_argument('--image-size', '-imsize', dest='image_size', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1, 2**(1/2), 1/2**(1/2)]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--soa', action='store_true',
                    help='use soa blocks')
parser.add_argument('--soa-layers', type=str, default='45',
                    help='config soa blocks for second-order attention')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")



def tb_setup(save_dir):
    # Setup for tensorboard
    tb_save_dir = os.path.join(
                    save_dir,
                    'summary',
                    )
    if not os.path.exists(tb_save_dir):
        os.makedirs(tb_save_dir)
    
    trash_list = os.listdir(tb_save_dir)
    for entry in trash_list:
        filename = os.path.join(tb_save_dir, entry)
        if fnmatch.fnmatch(entry, '*tfevents*'):
            os.remove(filename)

    summary = SummaryWriter(log_dir=tb_save_dir)

    return summary


def main():
    args = parser.parse_args()

    # check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network
    net = load_network(network_name=args.network)
    net.mode = 'test'

    print(">>>> loaded network: ")
    print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))

    print(">>>> Evaluating scales: {}".format(ms))

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets:
        summary_ranks  = tb_setup(os.path.join('specs/ranks/', dataset, args.network))
        summary_embeddings = tb_setup(os.path.join('specs/embeddings/', dataset, args.network))
        start = time.time()

        print('')
        print('>> {}: Extracting...'.format(dataset))

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
        images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
        try:
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        except:
            bbxs = None  # for holidaysmanrot and copydays

        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, mode='test')
        vecs = vecs.numpy()

        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, mode='test')
        qvecs = qvecs.numpy()

        print('>> {}: Evaluating...'.format(dataset))

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset, ranks, cfg['gnd'])

        print('')

        # plot retrieval rankings and save to tensorboard summary
        for protocol in ['easy', 'medium', 'hard']:
            plot_ranks(qimages, images, ranks, cfg['gnd'], bbxs, summary_ranks, dataset, 'solar-best: ', 20, protocol)

        print('')

        # plot embeddings for cluster visualisation in tensorboard/projector
        plot_embeddings(images, vecs, summary_embeddings, imsize=64, sample_freq=1)

        print('')
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))

    # args = parser.parse_args()

    # # check if there are unknown datasets
    # for dataset in args.datasets.split(','):
    #     if dataset not in datasets_names:
    #         raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # # check if test dataset are downloaded
    # # and download if they are not
    # # download_test(get_data_root())

    # # setting up the visible GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # # loading network
    # net = load_network(network_name=args.network)
    # net.mode = 'test'
    # # x = torch.randn(1, 3, 256, 256, requires_grad=False)
    # # torch.onnx.export(net, x, "solar.onnx", opset_version=12, verbose=True)

    # print(">>>> loaded network: ")
    # print(net.meta_repr())

    # # setting up the multi-scale parameters
    # ms = list(eval(args.multiscale))

    # print(">>>> Evaluating scales: {}".format(ms))

    # # moving network to gpu and eval mode
    # net.cuda()
    # net.eval()

    # # set up the transform
    # normalize = transforms.Normalize(
    #     mean=net.meta['mean'],
    #     std=net.meta['std']
    # )
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize
    # ])

    # # evaluate on test datasets
    # datasets = args.datasets.split(',')
    # for dataset in datasets:
    #     start = time.time()

    #     print('')
    #     print('>> {}: Extracting...'.format(dataset))

    #     # prepare config structure for the test dataset
    #     dataset_root_path = os.path.join(get_data_root(),'test',dataset)
    #     images = []
    #     qimages = []
    #     images_path = os.listdir(os.path.join(dataset_root_path,'query'))
    #     for dir_name in images_path:
    #         image_paths = glob.glob(os.path.join(dataset_root_path,'query', dir_name, '*.jpg'))
    #         for image_path in image_paths:
    #             qimages.append(image_path)
    #     images_path = os.listdir(os.path.join(dataset_root_path,'gallery'))
    #     for dir_name in images_path:
    #         image_paths = glob.glob(os.path.join(dataset_root_path,'gallery', dir_name, '*.jpg'))
    #         for image_path in image_paths:
    #             images.append(image_path)
    #     try:
    #         # bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
    #         bbxs = None  # for holidaysmanrot and copydays
    #     except:
    #         bbxs = None  # for holidaysmanrot and copydays

    #     # extract database and query vectors
    #     print('>> {}: database images...'.format(dataset))
    #     vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, mode='test')
    #     vecs = vecs.numpy()

    #     print('>> {}: query images...'.format(dataset))
    #     qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, mode='test')
    #     qvecs = qvecs.numpy()

    #     print('>> {}: Evaluating...'.format(dataset))

    #     # search, rank, and print
    #     scores = np.dot(vecs.T, qvecs)
    #     ranks = np.argsort(-scores, axis=0)
    #     scoresT = scores.T
    #     ranksT = ranks.T
    #     top1 = 0
    #     top_one = 0
    #     mAP = 0.0
    #     false_alarm_num = 0
    #     for i in range(ranksT.shape[0]):
    #         t = 0
    #         rank = 0.0
    #         query_id0 = qimages[i][qimages[i].rfind('/')-4:qimages[i].rfind('/')]
    #         gallery_id0 = images[ranksT[i][0]][images[ranksT[i][0]].rfind('/')-4:images[ranksT[i][0]].rfind('/')]
    #         # print(query_id0)
    #         # print(gallery_id0)
    #         if query_id0 == gallery_id0:
    #             top1 += 1
    #         # print(scoresT[i][ranksT[i][0]])
    #         if query_id0 == gallery_id0 and scoresT[i][ranksT[i][0]] > 0.6:
    #             top_one += 1
    #         query_id = qimages[i][qimages[i].rfind('/')-4:qimages[i].rfind('/')]
    #         for j in range(ranksT.shape[1]):
    #             gallery_id = images[ranksT[i][j]][images[ranksT[i][j]].rfind('/')-4:images[ranksT[i][j]].rfind('/')]
    #             if query_id == gallery_id:
    #                 t += 1
    #                 rank += t/(j+1)
    #             if query_id != gallery_id and scoresT[i][ranksT[i][j]] > 0.6:
    #                 false_alarm_num += 1
    #         if t == 0:
    #             continue
    #         mAP += rank / t
    #         print('{}.{} AP = {}%'.format(i, query_id, rank / t * 100))
    #     query_num = len(qimages)
    #     print('TOP1 num: {}'.format(top1))
    #     print('TOP1 recall: {}%'.format(top1 / query_num * 100))
    #     print('mAP = {}%'.format(mAP / query_num * 100))
    #     print('accuray: {}%'.format(top_one / query_num * 100))
    #     print('false num: {}'.format(false_alarm_num))
    #     print('false rate: {}%'.format(false_alarm_num / query_num * 100))


if __name__ == '__main__':
    main()
