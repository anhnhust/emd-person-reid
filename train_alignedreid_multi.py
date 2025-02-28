from __future__ import absolute_import
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import models
from util.losses import CrossEntropyLoss, DeepSupervision, CrossEntropyLabelSmooth, TripletLossAlignedReID
from util import data_manager
from util import transforms as T
from util.dataset_loader import ImageDataset
from util.utils import Logger
from util.utils import AverageMeter, Logger, save_checkpoint
from util.eval_metrics import evaluate
from util.optimizers import init_optim
from util.samplers import RandomIdentitySampler
from IPython import embed
import os
import pandas as pd

parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=384,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=256,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--labelsmooth', action='store_true', help="label smooth")
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=20, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=150, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# triplet hard loss
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--reranking',action= 'store_true', help= 'result re_ranking')

parser.add_argument('--test_distance',type = str, default='global', help= 'test distance type')
parser.add_argument('--unaligned',action= 'store_true', help= 'test local feature with unalignment')

args = parser.parse_args()

def main():
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    pin_memory = True if use_gpu else False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    # data augmentation
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'softmax','metric'}, aligned =True, use_gpu=use_gpu)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    if args.labelsmooth:
        criterion_class = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    else:
        criterion_class = CrossEntropyLoss(use_gpu=use_gpu)
    criterion_metric = TripletLossAlignedReID(margin=args.margin)
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return 0

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        if args.stepsize > 0: scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def test(model, queryloader, galleryloader, use_gpu, mode='average', ranks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """
    Test the model on query and gallery datasets.
    
    Args:
        model: The model to evaluate.
        queryloader: DataLoader for the query dataset.
        galleryloader: DataLoader for the gallery dataset.
        use_gpu: Whether to use GPU for computation.
        mode: Aggregation mode for features ('min', 'max', or 'average').
        ranks: List of ranks to compute CMC metrics.
    
    Returns:
        Rank-1 accuracy.
    """
    batch_time = AverageMeter()
    model.eval()

    with torch.no_grad():
        # Process query set
        qf, q_pids, q_camids, lqf, q_paths = [], [], [], [], []
        for batch_idx, (imgs, pids, camids, paths) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)
            q_paths.extend(paths)
            q_pids.extend(pids)
            q_camids.extend(camids)
        
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        # Aggregate features based on mode
        qf_2, lqf_2, q_pids_2, q_camids_2 = aggregate_features_camid(qf, lqf, q_pids, q_camids, mode)
        print(f"Query set aggregated using mode: {mode}")
        print(f"Query camids: {q_pids_2}")

        # Process gallery set
        gf, g_pids, g_camids, lgf, g_paths = [], [], [], [], []
        for batch_idx, (imgs, pids, camids, paths) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)
            g_paths.extend(paths)
            g_pids.extend(pids)
            g_camids.extend(camids)
        
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        # Aggregate features based on mode
        gf_2, lgf_2, g_pids_2, g_camids_2 = aggregate_features_camid(gf, lgf, g_pids, g_camids, mode)
        print(f"Gallery set aggregated using mode: {mode}")
        print(f"Gallery camids: {g_pids_2}")

    # Feature normalization
    qf_2 = 1. * qf_2 / (torch.norm(qf_2, 2, dim=-1, keepdim=True).expand_as(qf_2) + 1e-12)
    gf_2 = 1. * gf_2 / (torch.norm(gf_2, 2, dim=-1, keepdim=True).expand_as(gf_2) + 1e-12)

    # Compute distance matrix
    m, n = qf_2.size(0), gf_2.size(0)
    distmat = torch.pow(qf_2, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf_2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf_2, gf_2.t())
    distmat = distmat.numpy()

    # Handle local distance if needed
    if args.test_distance != 'global':
        from util.distance import low_memory_local_dist
        lqf_2 = lqf_2.permute(0, 2, 1)
        lgf_2 = lgf_2.permute(0, 2, 1)
        local_distmat = low_memory_local_dist(lqf_2.numpy(), lgf_2.numpy(), aligned=not args.unaligned)
        
        if args.test_distance == 'local':
            distmat = local_distmat
        elif args.test_distance == 'global_local':
            distmat = local_distmat + distmat

    # Compute CMC and mAP
    cmc, mAP = evaluate(distmat, q_pids_2, g_pids_2, q_camids_2, g_camids_2, q_paths, g_paths, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print(f"mAP: {mAP:.1%}")
    print("CMC curve")
    for r in ranks:
        print(f"Rank-{r:<3}: {cmc[r - 1]:.1%}")
    print("------------------")

    # Re-ranking if enabled
    if args.reranking:
        from util.re_ranking import re_ranking
        if args.test_distance == 'global':
            distmat = re_ranking(qf_2, gf_2, k1=20, k2=6, lambda_value=0.3)
        else:
            local_qq_distmat = low_memory_local_dist(lqf_2.numpy(), lqf_2.numpy(), aligned=not args.unaligned)
            local_gg_distmat = low_memory_local_dist(lgf_2.numpy(), lgf_2.numpy(), aligned=not args.unaligned)
            local_dist = np.concatenate(
                [np.concatenate([local_qq_distmat, local_distmat], axis=1),
                 np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
                axis=0)
            if args.test_distance == 'local':
                distmat = re_ranking(qf_2, gf_2, k1=20, k2=6, lambda_value=0.3, local_distmat=local_dist, only_local=True)
            elif args.test_distance == 'global_local':
                distmat = re_ranking(qf_2, gf_2, k1=20, k2=6, lambda_value=0.3, local_distmat=local_dist, only_local=False)
        
        # Compute CMC and mAP after re-ranking
        cmc, mAP = evaluate(distmat, q_pids_2, g_pids_2, q_camids_2, g_camids_2, use_metric_cuhk03=args.use_metric_cuhk03)
        print("Results after re-ranking ----------")
        print(f"mAP(RK): {mAP:.1%}")
        print("CMC curve(RK)")
        for r in ranks:
            print(f"Rank-{r:<3}: {cmc[r - 1]:.1%}")
        print("------------------")

    return cmc[0]


def aggregate_features_camid(features, local_features, pids, camids, mode='min'):
    """
    Aggregate features based on the specified mode.
    
    Args:
        features: Global features.
        local_features: Local features.
        pids: Person IDs.
        camids: Camera IDs.
        mode: Aggregation mode ('min', 'max', or 'average').
    
    Returns:
        Aggregated features, pids, and camids.
    """
    unique_camids = np.unique(camids)
    aggregated_features = {}
    aggregated_local_features = {}
    aggregated_pids = {}

    for camid in unique_camids:
        indices = np.where(camids == camid)[0]
        if mode == 'average':
            aggregated_features[camid] = np.mean(features[indices].numpy(), axis=0)
            aggregated_local_features[camid] = np.mean(local_features[indices].numpy(), axis=0)
        elif mode == 'max':
            aggregated_features[camid] = np.max(features[indices].numpy(), axis=0)
            aggregated_local_features[camid] = np.max(local_features[indices].numpy(), axis=0)
        elif mode == 'min':
            aggregated_features[camid] = np.min(features[indices].numpy(), axis=0)
            aggregated_local_features[camid] = np.min(local_features[indices].numpy(), axis=0)
        aggregated_pids[camid] = pids[indices[0]]  # Use the first PID for this camid

    # Convert to numpy arrays
    camids_2 = np.array(list(aggregated_features.keys()))
    features_2 = np.array(list(aggregated_features.values()))
    local_features_2 = np.array(list(aggregated_local_features.values()))
    pids_2 = np.array(list(aggregated_pids.values()))

    # Convert to torch tensors
    features_2 = torch.from_numpy(features_2)
    local_features_2 = torch.from_numpy(local_features_2)
    pids_2 = pids_2.astype(int)

    return features_2, local_features_2, pids_2, camids_2

def aggregate_features_pid(features, local_features, pids, camids, mode='min'):
    """
    Aggregate features based on PID instead of CAMID.
    
    Args:
        features: Global features (Tensor).
        local_features: Local features (Tensor).
        pids: Person IDs (numpy array).
        camids: Camera IDs (numpy array) (Not used in aggregation).
        mode: Aggregation mode ('min', 'max', or 'average').
    
    Returns:
        Aggregated features, local features, and pids.
    """
    unique_pids = np.unique(pids)
    aggregated_features = {}
    aggregated_local_features = {}
    
    for pid in unique_pids:
        indices = np.where(pids == pid)[0]
        if mode == 'average':
            aggregated_features[pid] = np.mean(features[indices].numpy(), axis=0)
            aggregated_local_features[pid] = np.mean(local_features[indices].numpy(), axis=0)
        elif mode == 'max':
            aggregated_features[pid] = np.max(features[indices].numpy(), axis=0)
            aggregated_local_features[pid] = np.max(local_features[indices].numpy(), axis=0)
        elif mode == 'min':
            aggregated_features[pid] = np.min(features[indices].numpy(), axis=0)
            aggregated_local_features[pid] = np.min(local_features[indices].numpy(), axis=0)
    
    # Convert to numpy arrays
    pids_2 = np.array(list(aggregated_features.keys()))
    features_2 = np.array(list(aggregated_features.values()))
    local_features_2 = np.array(list(aggregated_local_features.values()))
    
    # Convert to torch tensors
    features_2 = torch.from_numpy(features_2)
    local_features_2 = torch.from_numpy(local_features_2)
    pids_2 = pids_2.astype(int)
    
    return features_2, local_features_2, pids_2, camids

if __name__ == '__main__':
    main()