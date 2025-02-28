from __future__ import print_function, absolute_import
import numpy as np
import copy
from collections import defaultdict
import sys
import matplotlib.pyplot as plt

def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids,q_paths,g_paths, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    save_dir="/home/anhhoang/New_Term/AlignedReID/imgs/"
    for i in range(1,2,1):
        fig, axes = plt.subplots(nrows=1, ncols=11, figsize=(16, 4))

        # Plot query image in the first column
        query_img = plt.imread(q_img_paths[i])
        print(q_img_paths[i])
        axes[0].imshow(query_img)
        axes[0].axis('off')
        axes[0].set_title("Query")

        # Plot gallery images in the remaining columns
        for j in range(10):
            gallery_img = plt.imread(g_img_paths[indices[i][j]])
            print(g_img_paths[indices[i][j]])
            axes[j+1].imshow(gallery_img)
            axes[j+1].axis('off')
            if g_pids[indices[i][j]] == q_pids[i]:
                axes[j+1].set_title("%d. ID: %d"%(j+1, g_pids[indices[i][j]]), color='green')
            else:
                axes[j+1].set_title("%d. ID: %d"%(j+1, g_pids[indices[i][j]]), color='red')
        fig.savefig( save_dir+"show%d.png"%i)


    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids,q_paths,g_paths, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    print(distmat)
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    print(g_pids[indices])
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # print(matches)
    # save_dir="/home/micalab/AlignedReID/imgs/"
    # for i in range(1,10,1):
    #     fig, axes = plt.subplots(nrows=1, ncols=11, figsize=(16, 4))

    #     # Plot query image in the first column
    #     query_img = plt.imread(q_paths[i])
    #     #print(q_paths[i])
    #     axes[0].imshow(query_img)
    #     axes[0].axis('off')
    #     axes[0].set_title("Query")

    #     # Plot gallery images in the remaining columns
    #     for j in range(10):
    #         gallery_img = plt.imread(g_paths[indices[i][j]])
    #         #print(g_paths[indices[i][j]])
    #         axes[j+1].imshow(gallery_img)
    #         axes[j+1].axis('off')
    #         if g_pids[indices[i][j]] == q_pids[i]:
    #             axes[j+1].set_title("%d. ID: %d"%(j+1, g_pids[indices[i][j]]), color='green')
    #         else:
    #             axes[j+1].set_title("%d. ID: %d"%(j+1, g_pids[indices[i][j]]), color='red')
    #     fig.savefig( save_dir+"show%d.png"%i)    

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        #keep = np.invert(remove)
        keep = np.ones_like(remove, dtype=bool)


        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, q_paths,g_paths, max_rank=10, use_metric_cuhk03=False):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids,q_paths,g_paths, max_rank)
        #return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids,q_paths,g_paths, max_rank)
        #return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)