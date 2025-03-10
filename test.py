import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets.dataset_loader import build_testloader
from models import build_models
from utils.eval_metrics import evaluate, evaluate_with_clothes
from utils.evaluate import extract_vid_feature
from config import CONFIG


def test(model, queryloader, galleryloader, query, gallery):
    model.cuda()
    model.eval()
    print('======== Extracting query features ========')
    query_features, query_pids, query_camids, query_clothes_ids = extract_vid_feature(
        model=model,
        dataloader=queryloader,
        vid2clip_index=query.vid2clip_index,
        data_length=len(query.dataset),
    )
    print('======== Extracting gallery features ========')
    gallery_features, gallery_pids, gallery_camids, gallery_clothes_ids = extract_vid_feature(
        model,
        galleryloader,
        vid2clip_index=gallery.vid2clip_index,
        data_length=len(gallery.dataset),
    )

    torch.cuda.empty_cache()

    m, n = query_features.size(0), gallery_features.size(0)
    distance_matrix = torch.zeros((m, n))
    query_features, gallery_features = query_features.cuda(), gallery_features.cuda()
    # Cosine similarity
    for i in range(m):
        # distance_matrix[i] = (-torch.mm(query_features[i:i + 1], gallery_features.t())).cpu()
        distance_matrix[i] = (-torch.mm(query_features[i:i + 1], gallery_features.t()) / (torch.norm(query_features[i])*torch.norm(gallery_features, dim=1))).cpu()
    distance_matrix = distance_matrix.numpy()
    query_pids, query_camids, query_clothes_ids = query_pids.numpy(),\
          query_camids.numpy(), query_clothes_ids.numpy()
    gallery_pids, gallery_camids, gallery_clothes_ids = gallery_pids.numpy(),\
          gallery_camids.numpy(), gallery_clothes_ids.numpy()

    if CONFIG.DATA.DATASET in ['vccr', 'ccvid']:
        print("Computing CMC and mAP for Standard setting")
        standard_cmc, standard_mAP = evaluate(distance_matrix, query_pids, gallery_pids,
                            query_camids, gallery_camids)

        print("Computing CMC and mAP for same clothes setting")
        sc_cmc, sc_mAP = evaluate_with_clothes(distance_matrix,
                                        query_pids,
                                        gallery_pids,
                                        query_camids,
                                        gallery_camids,
                                        query_clothes_ids,
                                        gallery_clothes_ids,
                                        mode='SC')

        print("Computing CMC and mAP for cloth-changing setting")
        cc_cmc, cc_mAP = evaluate_with_clothes(distance_matrix,
                                        query_pids,
                                        gallery_pids,
                                        query_camids,
                                        gallery_camids,
                                        query_clothes_ids,
                                        gallery_clothes_ids,
                                        mode='CC')

        return (standard_cmc*100, standard_mAP*100, sc_cmc*100, sc_mAP*100, cc_cmc*100, cc_mAP*100)
    else:
        print("Computing CMC and mAP for cloth-changing setting")
        cc_cmc, cc_mAP = evaluate_with_clothes(distance_matrix,
                                        query_pids,
                                        gallery_pids,
                                        query_camids,
                                        gallery_camids,
                                        query_clothes_ids,
                                        gallery_clothes_ids,
                                        mode='CC')

        return cc_cmc*100, cc_mAP*100

"""
    Testing
"""
state_dict_path = f"work_space/ckpts/{CONFIG.TEST.TEST_SET}/resnet50_attn_stride_1_bn_cal/best_cc_r1.pth"
model_name = "resnet50_attn_stride_1_bn_cal_best_cc_r1"

print(f"Testing Model: {model_name} on {CONFIG.TEST.TEST_SET} with mode: {CONFIG.TEST.TEST_MODE}")

model = build_models(CONFIG, train=False)
model.load_state_dict(torch.load(state_dict_path)["model"], strict=False)
queryloader, galleryloader, query, gallery = build_testloader()

if CONFIG.DATA.DATASET in ['vccr', 'ccvid']:
    (standard_cmc, standard_mAP, sc_cmc, sc_mAP, cc_cmc, cc_mAP) = \
        test(model, queryloader, galleryloader, query, gallery)
    print("==============================")

    sc_results = f"SC | R-1: {sc_cmc[0]:.1f} | R-5: {sc_cmc[4]:.1f} | R-10: {sc_cmc[9]:.1f} | R-20: {sc_cmc[19]:.1f} | mAP: {sc_mAP:.1f}"
    print(sc_results)
    standard_results = f"Both | R-1: {standard_cmc[0]:.1f} | R-5: {standard_cmc[4]:.1f} | R-10: {standard_cmc[9]:.1f} | R-20: {standard_cmc[19]:.1f} | mAP: {standard_mAP:.1f}"
    print(standard_results)
    cc_results = f"CC | R-1: {cc_cmc[0]:.1f} | R-5: {cc_cmc[4]:.1f} | R-10: {cc_cmc[9]:.1f} | R-20: {cc_cmc[19]:.1f} | mAP: {cc_mAP:.1f}"
    print(cc_results)

    # Calculate the rank values for the x-axis
    ranks = np.arange(1, len(standard_cmc)+1)
    ranks = np.arange(1, 41)

    # # Plot the CMC curve
    plt.plot(ranks, sc_cmc[:40], '-o', label=sc_results)
    plt.plot(ranks, standard_cmc[:40], '-o', label=standard_results)
    plt.plot(ranks, cc_cmc[:40], '-x', label=cc_results)

    plt.xlabel('Rank')
    plt.ylabel('Identification Rate')
    plt.title(f'{model_name}_{CONFIG.TEST.TEST_SET}_{CONFIG.TEST.TEST_MODE}')
    plt.grid(False)
    # Save the plot to an output folder
    path = f"work_space/output/{model_name}_{CONFIG.TEST.TEST_SET}_{CONFIG.TEST.TEST_MODE}.png"
    plt.legend()
    plt.savefig(path)
else:
    cc_cmc, cc_mAP = test(model, queryloader, galleryloader, query, gallery)
    cc_results = f"Cloth-changing | R-1: {cc_cmc[0]:.1f} | R-5: {cc_cmc[4]:.1f} | R-10: {cc_cmc[9]:.1f} | mAP: {cc_mAP:.1f}"
    print(cc_results)

    # Calculate the rank values for the x-axis
    ranks = np.arange(1, len(cc_cmc)+1)
    ranks = np.arange(1, 41)

    # Plot the CMC curve
    plt.plot(ranks, cc_cmc[:40], '-x', label=cc_results)

    plt.xlabel('Rank')
    plt.ylabel('Identification Rate')
    plt.title(f'{model_name}_{CONFIG.TEST.TEST_SET}_{CONFIG.TEST.TEST_MODE}')
    plt.grid(False)
    # Save the plot to an output folder
    path = f"work_space/output/{model_name}_{CONFIG.TEST.TEST_SET}_{CONFIG.TEST.TEST_MODE}.png"
    plt.legend()
    plt.savefig(path)


