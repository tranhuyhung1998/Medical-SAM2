import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox


class Leaderboard(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):

        # Set the data list for training
        self.mode_map = {
            'Training': 'train',
            'Test': 'val',
            'Deploy': 'test1'
        }
        self.name_list = os.listdir(os.path.join(data_path, f'{self.mode_map[mode]}_images'))

        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

        if mode == 'Deploy':
            self.bbox_dict = self.build_bboxes()

    def build_bboxes(self):
        bbox_path = os.path.join(self.data_path, f'{self.mode_map[self.mode]}_bbox.txt')
        
        bbox_dict = {}
        with open(bbox_path) as f:
            for line in f:
                idx, box = line.split(':')
                ct, slice, organ = [part.strip() for part in idx.strip()[1:-1].split(',')]
                ct = f"{ct:02}"
                slice = int(slice)
                organ = int(organ)
                if (ct, slice) not in bbox_dict:
                    bbox_dict[(ct, slice)] = {}
        
                x0, y0, x1, y1 = [int(part.strip()) for part in box.strip()[1:-1].split(',')]
                bbox_dict[(ct, slice)][int(organ)] = np.array([x0, y0, x1, y1]) * 2
        return bbox_dict

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, f'{self.mode_map[self.mode]}_images', name)
        num_frame = len(os.listdir(img_path))
        
        assert self.mode != 'Deploy' or self.prompt != 'click'
            
        # data_seg_3d_shape = Image.open(mask_path + '/1.png').convert('L').size
        # data_seg_3d_shape = np.load(mask_path + '/0.npy').shape
        
        # data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame + 1,))
        # for i in range(1, num_frame + 1):
        #     data_seg_3d[..., i] = Image.open(os.path.join(mask_path, f'{i}.png')).convert('L')
        # for i in range(data_seg_3d.shape[-1]):
        #     if np.sum(data_seg_3d[..., i]) > 0:
        #         data_seg_3d = data_seg_3d[..., i:]
        #         break
        # starting_frame_nonzero = i
        # for j in reversed(range(data_seg_3d.shape[-1])):
        #     if np.sum(data_seg_3d[..., j]) > 0:
        #         data_seg_3d = data_seg_3d[..., :j + 1]
        #         break
        # num_frame = data_seg_3d.shape[-1]
        if self.mode != 'Deploy':
            mask_path = os.path.join(self.data_path, f'{self.mode_map[self.mode]}_labels', name)
            if self.video_length is None:
                video_length = int(num_frame / 4)
            else:
                video_length = self.video_length
            if num_frame > video_length and self.mode == 'Training':
                starting_frame = 1 + np.random.randint(0, num_frame - video_length + 1)
            else:
                starting_frame = 1

            mask_dict = {}
            point_label_dict = {}
            pt_dict = {}

        else:
            video_length = num_frame
            starting_frame = 1

        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            img = Image.open(os.path.join(img_path, f'{frame_index}.png')).convert('RGB')
            # mask = data_seg_3d[..., frame_index]
            if self.mode != 'Deploy':
                mask = np.asarray(Image.open(os.path.join(mask_path, f'{frame_index}.png')))
                # mask = np.rot90(mask)
                obj_list = np.unique(mask[mask > 0])

                diff_obj_mask_dict = {}

                if self.prompt == 'bbox':
                    diff_obj_bbox_dict = {}
                elif self.prompt == 'click':
                    diff_obj_pt_dict = {}
                    diff_obj_point_label_dict = {}
                else:
                    raise ValueError('Prompt not recognized')

            else:
                assert self.prompt == 'bbox'
                obj_list = list(self.bbox_dict[(name, frame_index)].keys())
                diff_obj_bbox_dict = {}   
            
            for obj in obj_list:
                if self.mode != 'Deploy':
                    obj_mask = mask == obj
                    # if self.transform_msk:
                    obj_mask = Image.fromarray(obj_mask)
                    obj_mask = obj_mask.resize(newsize)
                    obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                    # obj_mask = self.transform_msk(obj_mask).int()
                    diff_obj_mask_dict[obj] = obj_mask

                    if self.prompt == 'click':
                        diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                            point_label, seed=None)
                    if self.prompt == 'bbox':
                        diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                                seed=self.seed)
                else:
                    assert self.prompt == 'bbox'
                    diff_obj_bbox_dict[obj] = self.bbox_dict[(name, frame_index)][obj]

            # if self.transform:
            # state = torch.get_rng_state()
            # img = self.transform(img)
            # torch.set_rng_state(state)
            img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            if self.mode != 'Deploy':
                mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
                if self.prompt == 'bbox':
                    bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
                elif self.prompt == 'click':
                    pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                    point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict
            else:
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict


        image_meta_dict = {'filename_or_obj': name}
        if self.mode == 'Deploy':
            return {
                'image': img_tensor,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
            }

        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
            }