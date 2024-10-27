import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm

import cfg
from conf import settings
from func_3d.utils import eval_seg

args = cfg.parse_args()


GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
# criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# paper_loss = CombinedLoss(dice_weight=1 / 21, focal_weight=20 / 21)
seed = torch.randint(1,11,(1,7))

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []


def test_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    print('Number of batch:', n_val)
    mix_res = (0,) * 1 * 2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    # lossfunc = criterion_G
    # lossfunc = paper_loss

    prompt = args.prompt

    # segment_results = {}

    with tqdm(total=n_val, desc='Inference round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            # mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))

            train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(bbox_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue
            object_ids = torch.tensor(obj_list, dtype=torch.uint8).to(device=GPUdevice)

            name = pack['image_meta_dict']['filename_or_obj']

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results

                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: torch.nn.functional.interpolate(
                            torch.sigmoid(out_mask_logits[i].unsqueeze(0)),
                            size=args.out_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # segment_results[name[0]] = video_segments

                # loss = 0
                # pred_iou = 0
                # pred_dice = 0
                # for id in frame_id:
                #     for ann_obj_id in obj_list:
                #         pred = video_segments[id][ann_obj_id]
                #         pred = pred.unsqueeze(0)
                #         # pred = torch.sigmoid(pred)
                #         try:
                #             mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                #         except KeyError:
                #             mask = torch.zeros_like(pred).to(device=GPUdevice)
                #         if args.vis:
                #             os.makedirs(f'./temp/val/{name[0]}/{id}', exist_ok=True)
                #             fig, ax = plt.subplots(1, 3)
                #             ax[0].imshow(imgs_tensor[id, :, :, :].cpu().permute(1, 2, 0).numpy().astype(int))
                #             ax[0].axis('off')
                #             ax[1].imshow(pred[0, 0, :, :].cpu().numpy() > 0.5, cmap='gray')
                #             ax[1].axis('off')
                #             ax[2].imshow(mask[0, 0, :, :].cpu().numpy(), cmap='gray')
                #             ax[2].axis('off')
                #             plt.savefig(f'./temp/val/{name[0]}/{id}/{ann_obj_id}.png', bbox_inches='tight',
                #                         pad_inches=0)
                #             plt.close()

                # total_num = len(frame_id) * len(obj_list)
                # loss = loss / total_num
                # temp = (pred_iou / total_num, pred_dice / total_num)
                # tot += loss
                #
                # mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

                os.makedirs(f'./output/test_labels/{name[0]}', exist_ok=True)
                for id in frame_id:

                    # Stack tensors along a new dimension
                    stacked_tensors = torch.stack([video_segments[id][ann_obj_id].squeeze(0).squeeze(0)
                                                   for ann_obj_id in obj_list])

                    # Find best predictions
                    max_values, max_indices = torch.max(stacked_tensors, dim=0)

                    # Apply mask and set result on GPU
                    result = torch.zeros(max_values.shape, dtype=torch.uint8).to(device=GPUdevice)
                    mask = max_values >= 0.5
                    result[mask] = object_ids[max_indices][mask]

                    # save image
                    real_id = id if args.dataset != 'leaderboard' else id + 1
                    torchvision.io.write_png(result.unsqueeze(0).cpu(), f'./output/test_labels/{name[0]}/{real_id}.png', compression_level=0)


            net.reset_state(train_state)
            pbar.update()

    # return tot / n_val, tuple([a / n_val for a in mix_res])
    # return segment_results