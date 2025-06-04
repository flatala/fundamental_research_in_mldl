import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import glob as gb
import numpy as np

from libs.utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class UCF101(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 vis=False):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.vis = vis
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.vis:
            return clip, target, path, frame_indices
        else:
            return clip, target

    def __len__(self):
        return len(self.data)
        
def _is_video_valid_det(cur_vid_dets, n_frames, det_th=0.3, ratio_th=0.7):
    """Return True if ≥ratio_th of frames contain at least one detection
       whose score ≥ det_th."""
    good = sum(np.sum(v['human_boxes'][:, -1] > det_th) > 0
               for v in cur_vid_dets.values())
    return good / n_frames >= ratio_th


def _make_dataset_human_det(root_path, annotation_path, subset,
                            n_samples_for_each_video, sample_duration, dets):
    """Same as make_dataset but keeps only videos with decent detections."""
    data             = load_annotation_data(annotation_path)
    video_names, ann = get_video_names_and_annotations(data, subset)
    cls2idx          = get_class_labels(data)
    idx2cls          = {v: k for k, v in cls2idx.items()}

    orig_samples, final_samples = 0, []
    for i, vid_rel_path in enumerate(video_names):
        if i % 1000 == 0:
            print(f'dataset loading [{i}/{len(video_names)}]')

        vid_path   = os.path.join(root_path, vid_rel_path)
        if not os.path.exists(vid_path):
            continue

        n_frames   = int(load_value_file(os.path.join(vid_path, 'n_frames')))
        if n_frames <= 0:
            continue

        # how many samples would we have *without* filtering?
        if n_samples_for_each_video == 1:
            orig_samples += 1
        else:
            step = (sample_duration if n_samples_for_each_video <= 1 else
                    max(1, math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1))))
            orig_samples += len(range(1, n_frames, step))

        cur_cls = vid_path.split('/')[-2]           # class folder
        vid_key = vid_path.split('/')[-1]           # full video folder name
        if not _is_video_valid_det(dets[cur_cls][vid_key], n_frames):
            continue

        base = {'video'   : vid_path,
                'segment' : [1, n_frames],
                'n_frames': n_frames,
                'video_id': vid_key,
                'label'   : cls2idx[ann[i]['label']]}

        if n_samples_for_each_video == 1:
            sample = dict(base, frame_indices=list(range(1, n_frames + 1)))
            final_samples.append(sample)
        else:
            step = (sample_duration if n_samples_for_each_video <= 1 else
                    max(1, math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1))))
            for j in range(1, n_frames, step):
                fi = list(range(j, min(n_frames + 1, j + sample_duration)))
                final_samples.append(dict(base, frame_indices=fi))

    print(f'len(dataset) after filtering: {len(final_samples)}/{orig_samples}')

    # Optional: repeat so #samples matches the unfiltered count
    if len(final_samples) < orig_samples:
        rep = int(np.ceil(orig_samples / len(final_samples)))
        dup = []
        for _ in range(rep - 1):
            tmp = copy.deepcopy(final_samples)
            np.random.shuffle(tmp)
            dup.extend(tmp)
        final_samples.extend(dup)
        final_samples = final_samples[:orig_samples]

    return final_samples, idx2cls


def _gen_mask(img, dets, th=0.3):
    """Return a single-channel mask (0 = keep FG, 255 = erase) & flag."""
    mask_img = Image.new('L', (img.width, img.height), 255)
    cnt = 0
    for det in dets:
        if det[-1] >= th:
            cnt += 1
            poly = [(det[0], det[1]), (det[0], det[3]),
                    (det[2], det[3]), (det[2], det[1])]
            ImageDraw.Draw(mask_img).polygon(poly, fill=0)
    return mask_img, cnt > 0

def _maskout_human(img, dets, th=0.3):
    """Blur (actually: mean-fill) the human region, return img′ & cnt>0 flag."""
    pix_mean = tuple(np.round(np.mean(img, axis=(0, 1))).astype(np.int32))
    cnt = 0
    for det in dets:
        if det[-1] >= th:
            cnt += 1
            poly = [(det[0], det[1]), (det[0], det[3]),
                    (det[2], det[3]), (det[2], det[1])]
            ImageDraw.Draw(img).polygon(poly, fill=pix_mean)
    return img, cnt


class UCF101_adv(data.Dataset):
    """
    Adds (hard or soft) *scene / place* supervision.
    Returns clip, target, <place_index OR place_soft_target>.
    """
    def __init__(self, root_path, annotation_path, subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None, temporal_transform=None,
                 target_transform=None, sample_duration=16,
                 get_loader=get_default_video_loader,
                 place_pred_path=None, is_place_soft_label=False):

        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        if place_pred_path is None:
            raise ValueError('place_pred_path required for *_adv* loader')
            
        tag = 'train' if 'train' in subset else 'val'

        self.place_pred = {}
        
        pp_files = gb.glob(os.path.join(place_pred_path, f'*{tag}*.npy'))
        print(f'len_pp_files: {len(pp_files)}')
        for f in pp_files:
            self.place_pred.update(np.load(f, allow_pickle=True).item())
            
        print(len(self.place_pred))
        vid0 = list(self.place_pred[list(self.place_pred.keys())[0]].keys())[0]
        self.multiple_place_inds = isinstance(
            self.place_pred[list(self.place_pred.keys())[0]][vid0]['pred_cls'],
            np.ndarray)

        self.is_place_soft     = is_place_soft_label
        self.spatial_transform = spatial_transform
        self.temporal_transform= temporal_transform
        self.target_transform  = target_transform
        self.loader            = get_loader()

    # ---------------------------------------
    def __getitem__(self, index):
        sample = self.data[index]
        path, frame_indices = sample['video'], sample['frame_indices']
        if self.temporal_transform:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)

        if self.spatial_transform:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(i) for i in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = sample if self.target_transform is None \
                          else self.target_transform(sample)

        # ----- place label -----
        place_dict = self.place_pred[path.split('/')[-2]][sample['video_id']]
        if self.is_place_soft:
            out = np.mean(place_dict['probs'], 0) if self.multiple_place_inds \
                 else place_dict['probs']
            return clip, target, out
        else:
            idx = place_dict['pred_cls']
            if self.multiple_place_inds:
                idx = np.bincount(idx).argmax()
            return clip, target, idx

    def __len__(self):
        return len(self.data)
        

class UCF101_bkgmsk(data.Dataset):
    """
    Outputs clips whose BACKGROUND is optionally *blanked* (mask_ratio)
    – identical to Kinetics_bkgmsk but adapted to UCF-101 keys.
    """
    def __init__(self, root_path, annotation_path, subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None, temporal_transform=None,
                 target_transform=None, sample_duration=16,
                 get_loader=get_default_video_loader,
                 detection_path=None, mask_ratio=0.5):

        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        # ---------- human detections ----------
        if detection_path is None:
            raise ValueError('detection_path is required for *_bkgmsk* loader')
        tag = 'train' if 'train' in subset else 'val'
        det_file = os.path.join(
            detection_path, f'detection_{tag}_merged_rearranged.npy')
        print(f'loading human dets from {det_file} …')
        self.human_dets = np.load(det_file, allow_pickle=True).item()
        print('… done.')

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform  = target_transform
        self.loader            = get_loader()
        self.mask_ratio        = mask_ratio
        self.subset            = subset

    # ---------------------------------------
    def __getitem__(self, index):
        sample = self.data[index]
        path, frame_indices = sample['video'], sample['frame_indices']
        if self.temporal_transform:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)

        # grab detections for every frame
        cls   = path.split('/')[-2]
        vidid = path.split('/')[-1]
        dets  = self.human_dets[cls][vidid]

        fg_imgs = []
        r = np.random.rand(1)
        for fi, img in zip(frame_indices, clip):
            cur_det = dets[fi]['human_boxes']
            if cur_det.shape[0] and r < self.mask_ratio:
                mask, _ = _gen_mask(img, cur_det)
                blank   = Image.new('L', (img.width, img.height), 0)
                img     = Image.composite(blank, img, mask)
            fg_imgs.append(img)

        if self.spatial_transform:
            self.spatial_transform.randomize_parameters()
            fg_imgs = [self.spatial_transform(i) for i in fg_imgs]
        clip = torch.stack(fg_imgs, 0).permute(1, 0, 2, 3)

        target = sample if self.target_transform is None \
                          else self.target_transform(sample)
        return clip, target

    def __len__(self):
        return len(self.data)


class UCF101_human_msk(data.Dataset):
    """
    Returns (clip, target, is_masking) where *is_masking* is True when the
    current clip had ≥mask_th of its frames human-masked.
    """

    def __init__(self, root_path, annotation_path, subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None, temporal_transform=None,
                 target_transform=None, sample_duration=16,
                 get_loader=get_default_video_loader,
                 detection_path=None, mask_ratio=0.5, mask_th=0.5):

        # ---------------------------- load detections -----------------
        if detection_path is None:
            raise ValueError('detection_path is required for human-msk loader')

        det_file = os.path.join(
            detection_path,
            f'detection_{"train" if "train" in subset else "val"}'
            '_merged_rearranged.npy'
        )
        print(f'loading human dets from {det_file} …')
        self.human_dets = np.load(det_file, allow_pickle=True).item()
        print('… done.')

        # ---------------------------- build list ----------------------
        self.data, self.class_names = _make_dataset_human_det(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration, self.human_dets)

        self.subset            = subset
        self.spatial_transform = spatial_transform
        self.temporal_transform= temporal_transform
        self.target_transform  = target_transform
        self.loader            = get_loader()
        self.mask_ratio        = mask_ratio
        self.mask_th           = mask_th

    # ----------------------------------------------------------------
    def __getitem__(self, index):
        sample = self.data[index]
        path, frame_indices = sample['video'], sample['frame_indices']

        if self.temporal_transform:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)

        # ------------- retrieve detections for these frames -----------
        cur_cls, vid_key = path.split('/')[-2], path.split('/')[-1]
        detections = self.human_dets[cur_cls][vid_key]

        r = np.random.rand(1)
        mask_cnt, processed = 0, []
        for fi, img in zip(frame_indices, clip):
            dets = detections[fi]['human_boxes']
            if dets.shape[0] and (r < self.mask_ratio):
                img, cnt = _maskout_human(img, dets)
                mask_cnt += cnt
            processed.append(img)

        is_masking = mask_cnt / len(processed) >= self.mask_th

        if self.spatial_transform:
            self.spatial_transform.randomize_parameters()
            processed = [self.spatial_transform(im) for im in processed]
        clip_tensor = torch.stack(processed, 0).permute(1, 0, 2, 3)

        target = sample
        if self.target_transform:
            target = self.target_transform(target)

        return clip_tensor, target, is_masking

    # ----------------------------------------------------------------
    def __len__(self):
        return len(self.data)


class UCF101_adv_msk(data.Dataset):
    """
    Joint loader: scene/place label **and** human-mask-confusion signal.
    Returns clip, target, place_label/soft, is_masking (bool).
    """
    def __init__(self, root_path, annotation_path, subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None, temporal_transform=None,
                 target_transform=None, sample_duration=16,
                 get_loader=get_default_video_loader,
                 place_pred_path=None, is_place_soft_label=False,
                 detection_path=None, mask_ratio=0.5, mask_th=0.5):

        # ---------- base list ----------
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)
        self.subset = subset

        # ---------- place predictions ----------
        if place_pred_path is None:
            raise ValueError('place_pred_path required for *_adv_msk* loader')
        tag = 'train' if 'train' in subset else 'val'
        pp_files = gb.glob(os.path.join(place_pred_path, f'*{tag}*.npy'))
        self.place_pred = {}
        for f in pp_files:
            self.place_pred.update(np.load(f, allow_pickle=True).item())
        vid0 = list(self.place_pred[list(self.place_pred.keys())[0]].keys())[0]
        self.multiple_place_inds = isinstance(
            self.place_pred[list(self.place_pred.keys())[0]][vid0]['pred_cls'],
            np.ndarray)
        self.is_place_soft = is_place_soft_label

        # ---------- human detections ----------
        if detection_path is None:
            raise ValueError('detection_path required for *_adv_msk* loader')
        det_file = os.path.join(
            detection_path, f'detection_{tag}_merged_rearranged.npy')
        print(f'loading human dets from {det_file} …')
        self.human_dets = np.load(det_file, allow_pickle=True).item()
        print('… done.')

        self.spatial_transform = spatial_transform
        self.temporal_transform= temporal_transform
        self.target_transform  = target_transform
        self.loader            = get_loader()
        self.mask_ratio        = mask_ratio
        self.mask_th           = mask_th

    # ---------------------------------------
    def __getitem__(self, index):
        sample = self.data[index]
        path, frame_indices = sample['video'], sample['frame_indices']
        if self.temporal_transform:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)

        cls   = path.split('/')[-2]
        vidid = path.split('/')[-1]
        dets  = self.human_dets[cls][vidid]

        fg_imgs, mask_cnt = [], 0
        r = np.random.rand(1)
        for fi, img in zip(frame_indices, clip):
            cur_det = dets[fi]['human_boxes']
            if cur_det.shape[0] and r < self.mask_ratio:
                img, cnt = _maskout_human(img, cur_det)
                mask_cnt += cnt
            fg_imgs.append(img)
        is_masking = mask_cnt / len(fg_imgs) >= self.mask_th

        if self.spatial_transform:
            self.spatial_transform.randomize_parameters()
            fg_imgs = [self.spatial_transform(i) for i in fg_imgs]
        clip = torch.stack(fg_imgs, 0).permute(1, 0, 2, 3)

        target = sample if self.target_transform is None \
                          else self.target_transform(sample)

        place_dict = self.place_pred[cls][sample['video_id']]
        if self.is_place_soft:
            out = np.mean(place_dict['probs'], 0) if self.multiple_place_inds \
                 else place_dict['probs']
            return clip, target, out, is_masking
        else:
            idx = place_dict['pred_cls']
            if self.multiple_place_inds:
                idx = np.bincount(idx).argmax()
            return clip, target, idx, is_masking

    def __len__(self):
        return len(self.data)
