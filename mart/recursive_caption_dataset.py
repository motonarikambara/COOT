"""
Captioning dataset.

References:
    Copyright (c) 2017 Jie Lei
    Licensed under The MIT License, see https://choosealicense.com/licenses/mit/
    @inproceedings{lei2020mart,
        title={MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning},
        author={Lei, Jie and Wang, Liwei and Shen, Yelong and Yu, Dong and Berg, Tamara L and Bansal, Mohit},
        booktitle={ACL},
        year={2020}
    }

    History:
    https://github.com/jayleicn/recurrent-transformer
    Current version 2021 https://github.com/gingsi/coot-videotext
"""
import copy
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import nltk
import numpy as np
import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import sys

from mart.configs_mart import MartConfig, MartPathConst
from nntrainer.typext import ConstantHolder
from nntrainer.utils_torch import create_shared_array


class DataTypesConstCaption(ConstantHolder):
    """
    Possible video data types for the dataset: Video features or COOT embeddings.
    """
    VIDEO_FEAT = "video_feat"
    COOT_EMB = "coot_emb"


class RecursiveCaptionDataset(data.Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"
    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    IGNORE = -1  # used to calculate loss

    """
    recurrent: if True, return recurrent data
    """

    def __init__(self, dset_name: str, max_t_len, max_v_len, max_n_sen, mode="train",
                 recurrent=True, untied=False,
                 video_feature_dir: Optional[str] = None,
                 coot_model_name=None, coot_mode="all", coot_dim_vid=768, coot_dim_clip=384,
                 annotations_dir: str = "annotations", coot_feat_dir="provided_embeddings",
                 dataset_max: Optional[int] = None, preload: bool = False):
        # metadata settings
        self.dset_name = dset_name
        self.annotations_dir = Path(annotations_dir)

        # COOT feature settings
        self.coot_model_name = coot_model_name
        self.coot_mode = coot_mode  # "all" for vid and clip repr, "clip" for only clip repr
        self.coot_dim_vid = coot_dim_vid
        self.coot_dim_clip = coot_dim_clip
        self.coot_feat_dir = Path(coot_feat_dir)

        # Video feature settings
        self.video_feature_dir = Path(video_feature_dir) / self.dset_name
        self.duration_file = self.annotations_dir / self.dset_name / "captioning_video_feat_duration.csv"
        self.word2idx_file = self.annotations_dir / self.dset_name / "mart_word2idx.json"
        self.word2idx = json.load(self.word2idx_file.open("rt", encoding="utf8"))
        self.idx2word = {int(v): k for k, v in list(self.word2idx.items())}
        print(f"WORD2IDX: {self.word2idx_file} len {len(self.word2idx)}")

        # Parameters for sequence lengths
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sen
        self.max_n_sen = max_n_sen

        # Train or val mode
        self.mode = mode
        self.preload = preload

        # Recurrent or untied, different data styles for different models
        self.recurrent = recurrent
        self.untied = untied
        assert not (self.recurrent and self.untied), "untied and recurrent cannot be True for both"

        # ---------- Load metadata ----------

        # determine metadata file
        if self.dset_name == "activitynet":
            if mode == "train":  # 10000 videos
                data_path = self.annotations_dir / self.dset_name / "train.json"
            elif mode == "val":  # 2500 videos
                data_path = self.annotations_dir / self.dset_name / "captioning_val_1.json"
            elif mode == "test":  # 2500 videos
                data_path = self.annotations_dir / self.dset_name / "captioning_test_1.json"
            else:
                raise ValueError(f"Mode must be [train, val, test] for {self.dset_name}, got {mode}")
        elif self.dset_name == "youcook2":
            tmp_path = "youcook2_next"
            if mode == "train":  # 1333 videos
                # data_path = self.annotations_dir / self.dset_name / "captioning_train.json"
                data_path = self.annotations_dir / tmp_path / "captioning_train.json"
            elif mode == "val":  # 457 videos
                # data_path = self.annotations_dir / self.dset_name / "captioning_val.json"
                data_path = self.annotations_dir / tmp_path / "captioning_val.json"
            else:
                raise ValueError(f"Mode must be [train, val] for {self.dset_name}, got {mode}")
        else:
            raise ValueError(f"Unknown dataset {self.dset_name}")

        # load and process captions and video data
        raw_data = json.load(data_path.open("rt", encoding="utf8"))
        coll_data = []
        for i, (k, line) in enumerate(tqdm(list(raw_data.items()))):
            if dataset_max is not None and i >= dataset_max > 0:
                break
            line["name"] = k
            # line["timestamps"] = line["timestamps"][:self.max_n_sen]
            # line["sentences"] = line["sentences"][:self.max_n_sen]
            coll_data.append(line)

        if self.recurrent:  # recurrent
            self.data = coll_data
        else:  # non-recurrent single sentence
            single_sentence_data = []
            for d in coll_data:
                num_sen = min(self.max_n_sen, len(d["sentences"]))
                single_sentence_data.extend([
                    {
                        "duration": d["duration"],
                        "name": d["name"],
                        "timestamp": d["timestamps"][idx],
                        "sentence": d["sentences"][idx],
                        "idx": idx
                    } for idx in range(num_sen)])
            self.data = single_sentence_data

        # ---------- Load video data ----------

        # Decide whether to load COOT embeddings or video features
        if self.coot_model_name is not None:
            # COOT embeddings
            self.data_type = DataTypesConstCaption.COOT_EMB

            # for activitynet, coot val split contains both ae-val and ae-test splits
            coot_dataset_mode = "val" if self.mode == "test" else self.mode

            self.coot_emb_h5_file = self.coot_feat_dir / f"{self.coot_model_name}_{coot_dataset_mode}.h5"
            assert self.coot_emb_h5_file.is_file(), f"Coot embeddings file not found: {self.coot_emb_h5_file}"

            # load coot embeddings data
            data_file = h5py.File(self.coot_emb_h5_file, "r")

            if "key" not in data_file:
                # backwards compatible to old h5+json embeddings
                vtdata = json.load((self.coot_feat_dir / f"{self.coot_model_name}_{mode}.json").open(
                    "rt", encoding="utf8"))
                clip_nums = vtdata["clip_nums"]
                vid_ids = vtdata["vid_ids"]
                clip_ids = vtdata["clip_ids"]
            else:
                # new version, everything in the h5
                # decode video ids from byte to utf8
                # vid_ids = [key.decode("utf8") for key in data_file["key"]]
                vid_ids = [key for key in data_file["key"]]

                # load clip information
                clip_nums = data_file["clip_num"]
                clip_ids = []
                assert len(vid_ids) == len(clip_nums)
                for vid_id, clip_num in zip(vid_ids, clip_nums):
                    for c in range(clip_num):
                        clip_ids.append((vid_id, c))
            self.coot_clip_nums = np.array(clip_nums)
            # map video id to video number
            self.coot_vid_id_to_vid_number = {}
            for i, vid_id in enumerate(vid_ids):
                self.coot_vid_id_to_vid_number[vid_id] = i

            # map video id and clip id to clip number
            self.coot_vid_clip_id_to_clip_number = {}
            for i, (vid_id, clip_id) in enumerate(clip_ids):
                self.coot_vid_clip_id_to_clip_number[f"{vid_id}/{clip_id}"] = i

            self.frame_to_second = None  # Don't need this for COOT embeddings

        print(f"Dataset {self.dset_name} #{len(self)} {self.mode} input {self.data_type}")

        self.preloading_done = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta

    def _load_mart_video_feature(self, raw_name: str) -> np.array:
        """
        Load given mart video feature

        Args:
            raw_name: Video ID

        Returns:
            Mart video feature with shape (len_sequence, 3072)
        """
        if self.preload and self.preloading_done:
            return self.preloaded_videos[raw_name]
        video_name = raw_name[2:] if self.dset_name == "activitynet" else raw_name
        feat_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
        feat_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
        video_feature = np.concatenate([np.load(feat_path_resnet), np.load(feat_path_bn)], axis=1)
        return video_feature

    def _load_coot_video_feature(self, raw_name: str) -> Tuple[np.array, np.array, List[np.array]]:
        """
        Load given COOT video features.

        Args:
            raw_name: Video ID

        Returns:
            Tuple of:
                video with shape (dim_video)
                context with shape (dim_clip)
                clips with shape (dim_clip)
        """
        if self.preload and self.preloading_done:
            return self.preloaded_videos[raw_name]
        try:
            # load video with default name
            vid_num = self.coot_vid_id_to_vid_number[raw_name]
            fixed_name = raw_name
        except KeyError:
            # not found, have to modify the name for activitynet
            mode = "val_1" if self.mode == "val" else self.mode
            fixed_name = f"{raw_name[2:]}_{mode}"
            vid_num = self.coot_vid_id_to_vid_number[fixed_name]
        h5 = h5py.File(self.coot_emb_h5_file, "r")

        if "vid_emb" not in h5:
            # backwards compatibility
            embs = ['vid_norm', 'vid', 'clip_norm', 'clip', 'vid_ctx_norm', 'vid_ctx',
                    'par_norm', 'par', 'sent_norm', 'sent', 'par_ctx_norm', 'par_ctx']
            (f_vid_emb, f_vid_emb_before_norm, f_clip_emb, f_clip_emb_before_norm,
             f_vid_context, f_vid_context_before_norm,
             f_par_emb, f_par_emb_before_norm, f_sent_emb, f_sent_emb_before_norm,
             f_par_context, f_par_context_before_norm) = embs
        else:
            # new version
            f_vid_emb, f_clip_emb, f_vid_context, f_par_emb, f_sent_emb, f_par_context = [
                'vid_emb', 'clip_emb', 'vid_context', 'par_emb', 'sent_emb', 'par_context']

        # 動画に関する特徴量を取得
        vid_feat = np.array(h5[f_vid_emb][vid_num])
        vidctx_feat = np.array(h5[f_vid_context][vid_num])
        num_clips = self.coot_clip_nums[vid_num]
        clip_feats = []
        future_clip = []
        clip_vid = np.zeros((15, 1, 384))
        future_vid = np.zeros((15, 1, 384))
        for clip in range(num_clips-1):
            future_num = clip + 1
            clip_num = self.coot_vid_clip_id_to_clip_number[f"{fixed_name}/{clip}"]
            future = self.coot_vid_clip_id_to_clip_number[f"{fixed_name}/{future_num}"]
            clip_feat = np.array(h5[f_clip_emb][clip_num])
            future_feat = np.array(h5[f_clip_emb][future])
            clip_vid[clip, 0, :] = clip_feat
            # future_vid[clip, 0, :] = future_feat
            clip_feats.append(clip_feat)
            future_clip.append(future_feat)
        clip_feat = np.array(h5[f_clip_emb][0])
        # if 全てのクリップ
        # clip_feats.append(clip_feat)
        clip_feats = np.stack(clip_feats, axis=0)
        future_clip = np.stack(future_clip, axis=0)
        # ver. future
        return clip_feats, future_clip

    def convert_example_to_features(self, example):
        """
        example single snetence
        {"name": str,
         "duration": float,
         "timestamp": [st(float), ed(float)],
         "sentence": str
        } or
        {"name": str,
         "duration": float,
         "timestamps": list([st(float), ed(float)]),
         "sentences": list(str)
        }
        """
        raw_name = example["name"]
        if self.data_type == DataTypesConstCaption.VIDEO_FEAT:
            # default video features from MART paper
            video_feature = self._load_mart_video_feature(raw_name)
        else:
            # ver. future
            clip_feats, future_feats = self._load_coot_video_feature(raw_name)
            video_feature = clip_feats          

        # print("loaded features", name, video_name, video_feature.shape)
        # print(video_feature)
        # sys.exit()

        if self.recurrent:
            # recurrent
            num_sen = len(example["sentences"])
            single_video_features = []
            single_video_meta = []
            for clip_idx in range(num_sen):
                cur_data, cur_meta = self.clip_sentence_to_feature(
                    example["name"], example["timestamps"][clip_idx], example["sentences"][clip_idx], video_feature, future_feats,
                    clip_idx)
                single_video_features.append(cur_data)
                single_video_meta.append(cur_meta)
            # cur_data:video特徴量を含むdict
            # cur_data, cur_meta = self.clip_sentence_to_feature(
            #     example["name"], example["timestamps"], example["sentences"], video_feature, clip_vid)
            # single_video_features.append(cur_data)
            # single_video_meta.append(cur_meta)
            # single_video_features: video特徴量を含むdict
            return single_video_features, single_video_meta
        else:
            print("recursive_caption_dataset.py")
            sys.exit()

    # ver. original
    # def clip_sentence_to_feature(self, name, timestamp, sentence, video_feature, clip_idx: int):
    # ver. future
    def clip_sentence_to_feature(self, name, timestamp, sentence, video_feature, future_clip, clip_idx: int):
    # ver. lstm
    # def clip_sentence_to_feature(self, name, timestamp, sentence, video_feature, clip_vid):
        """
        make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            video_feature: Either np.array of rgb+flow features or Dict[str, np.array] of COOT embeddings
            clip_idx: clip number in the video (needed to loat COOT features)
        """
        frm2sec = None
        if self.data_type == DataTypesConstCaption.VIDEO_FEAT:
            frm2sec =\
                self.frame_to_second[name[2:]] if \
                self.dset_name == "activitynet" else self.frame_to_second[name]

        # future
        feat, video_tokens, video_mask, future_clip =\
            self._load_indexed_video_feature(
                video_feature, timestamp, frm2sec, clip_idx, future_clip)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        input_tokens = video_tokens + text_tokens

        input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens]
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        input_labels = [self.IGNORE] * len(video_tokens) +\
            [self.IGNORE if m == 0 else tid for tid, m in zip(
                input_ids[-len(text_mask):], text_mask)][1:] + [self.IGNORE]
        input_mask = video_mask + text_mask
        token_type_ids = [0] * self.max_v_len + [1] * self.max_t_len

        # ver. future
        coll_data = dict(
            name=name, input_tokens=input_tokens,
            input_ids=np.array(input_ids).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64),
            input_mask=np.array(input_mask).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
            video_feature=feat.astype(np.float32),
            future_clips=future_clip.astype(np.float32))
        meta = dict(
            name=name, timestamp=timestamp, sentence=sentence, )
        return coll_data, meta

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, frm2sec):
        """
        convert wall time st_ed to feature index st_ed
        """
        st = int(math.floor(timestamp[0] / frm2sec))
        ed = int(math.ceil(timestamp[1] / frm2sec))
        ed = min(ed, feat_len - 1)
        st = min(st, ed - 1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(
            st, ed, feat_len)
        return st, ed

    # future
    def _get_vt_features(self, video_feat_tuple, clip_idx, max_v_l, future_feats):
    # ひとまとめにしたvideo関連の特徴量から必要なものを抽出
        clip_feats = video_feat_tuple
        clip_feat = clip_feats[clip_idx]
        future_feat = future_feats[clip_idx]
        # only clip (1, 384)
        valid_l = 0
        feat = np.zeros((max_v_l, self.coot_dim_clip))
        feat[valid_l] = clip_feat

        future = np.zeros((max_v_l, self.coot_dim_clip))
        future[valid_l] = future_feat            
        valid_l += 1
        # print("CLIP")

        assert valid_l == max_v_l, f"valid {valid_l} max {max_v_l}"
        # return feat, valid_l
        # future
        return feat, valid_l, future

    # def _load_indexed_video_feature(self, raw_feat, timestamp, frm2sec, clip_idx):
    # future
    def _load_indexed_video_feature(self, raw_feat, timestamp, frm2sec, clip_idx, future_feats):
        """
        [CLS], [VID], ..., [VID], [SEP], [PAD], ..., [PAD],
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat is padded to length of (self.max_v_len + self.max_t_len,)
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """
        # COOT video text data as input
        max_v_l = self.max_v_len - 2
        # future
        raw_feat, valid_l, future =\
            self._get_vt_features(raw_feat, clip_idx, max_v_l, future_feats)
        video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l +\
                        [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)
        mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)
        # 上記のように特徴量を配置
        # feat∈25×1152
        feat = np.zeros((self.max_v_len + self.max_t_len, raw_feat.shape[1]))  # includes [CLS], [SEP]
        # feat[1:len(raw_feat) + 1] = raw_feat
        feat[1:4] = raw_feat
        # future
        future_feat = np.zeros((self.max_v_len + self.max_t_len, future.shape[1]))  # includes [CLS], [SEP]
        future_feat[1:len(future) + 1] = future            
        # feat = raw_feat
        # return feat, video_tokens, mask
        # future
        return feat, video_tokens, mask, future_feat

    def _tokenize_pad_sentence(self, sentence):
        """
        [BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD],
            len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_t_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True,
                                return_sentence_only=True) -> str:
        """
        A list of token ids
        """
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)

    def collate_fn(self, batch):
        """
        Args:
            batch:

        Returns:
        """
        if self.recurrent:
            # recurrent collate function. original docstring:
            # HOW to batch clip-sentence pair? 1) directly copy the last 
            # sentence, but do not count them in when
            # back-prop OR put all -1 to their text token label, treat

            # collect meta
            raw_batch_meta = [e[1] for e in batch]
            batch_meta = []
            for e in raw_batch_meta:
                cur_meta = dict(
                    name=None,
                    timestamp=[],
                    gt_sentence=[]
                )
                for d in e:
                    cur_meta["name"] = d["name"]
                    cur_meta["timestamp"].append(d["timestamp"])
                    cur_meta["gt_sentence"].append(d["sentence"])
                batch_meta.append(cur_meta)

            batch = [e[0] for e in batch]
            # Step1: pad each example to max_n_sen
            max_n_sen = max([len(e) for e in batch])
            raw_step_sizes = []

            padded_batch = []
            padding_clip_sen_data = copy.deepcopy(
                batch[0][0])  # doesn"t matter which one is used
            padding_clip_sen_data["input_labels"][:] = RecursiveCaptionDataset.IGNORE
            for ele in batch:
                cur_n_sen = len(ele)
                if cur_n_sen < max_n_sen:
                    # noinspection PyAugmentAssignment
                    ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
                raw_step_sizes.append(cur_n_sen)
                padded_batch.append(ele)

            # Step2: batching each steps individually in the batches
            collated_step_batch = []
            for step_idx in range(max_n_sen):
                collated_step = step_collate([e[step_idx] for e in padded_batch])
                collated_step_batch.append(collated_step)
            return collated_step_batch, raw_step_sizes, batch_meta

        # single sentences / untied

        # collect meta
        batch_meta = [{
            "name": e[1]["name"],
            "timestamp": e[1]["timestamp"],
            "gt_sentence": e[1]["sentence"]
        } for e in batch]  # change key
        padded_batch = step_collate([e[0] for e in batch])
        return padded_batch, None, batch_meta


def prepare_batch_inputs(batch, use_cuda: bool, non_blocking=False):
    batch_inputs = dict()
    bsz = len(batch["name"])
    for k, v in list(batch.items()):
        assert bsz == len(v), (bsz, k, v)
        if use_cuda:
            if isinstance(v, torch.Tensor):
                v = v.cuda(non_blocking=non_blocking)
        batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """
    The same step (clip-sentence pair) from each example
    """
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def create_mart_datasets_and_loaders(
        cfg: MartConfig, coot_feat_dir: str = MartPathConst.COOT_FEAT_DIR,
        annotations_dir: str = MartPathConst.ANNOTATIONS_DIR,
        video_feature_dir: str = MartPathConst.VIDEO_FEATURE_DIR
) -> Tuple[RecursiveCaptionDataset,
           RecursiveCaptionDataset,
           data.DataLoader,
           data.DataLoader]:
    # create the dataset
    dset_name_train = cfg.dataset_train.name
    train_dataset = RecursiveCaptionDataset(
        dset_name_train, cfg.max_t_len, cfg.max_v_len, cfg.max_n_sen,
        mode="train", recurrent=cfg.recurrent, untied=cfg.untied or cfg.mtrans,
        video_feature_dir=video_feature_dir,
        coot_model_name=cfg.coot_model_name, coot_mode=cfg.coot_mode, coot_dim_vid=cfg.coot_dim_vid,
        coot_dim_clip=cfg.coot_dim_clip, annotations_dir=annotations_dir, coot_feat_dir=coot_feat_dir,
        dataset_max=cfg.dataset_train.max_datapoints, preload=cfg.dataset_train.preload)
    # add 10 at max_n_sen to make the inference stage use all the segments
    # max_n_sen_val = cfg.max_n_sen + 10
    max_n_sen_val = cfg.max_n_sen
    val_dataset = RecursiveCaptionDataset(
        cfg.dataset_val.name, cfg.max_t_len, cfg.max_v_len, max_n_sen_val,
        mode="val", recurrent=cfg.recurrent, untied=cfg.untied or cfg.mtrans,
        video_feature_dir=video_feature_dir,
        coot_model_name=cfg.coot_model_name, coot_mode=cfg.coot_mode, coot_dim_vid=cfg.coot_dim_vid,
        coot_dim_clip=cfg.coot_dim_clip, annotations_dir=annotations_dir, coot_feat_dir=coot_feat_dir,
        dataset_max=cfg.dataset_val.max_datapoints, preload=cfg.dataset_val.preload)

    train_loader = data.DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn, batch_size=cfg.train.batch_size,
        shuffle=cfg.dataset_train.shuffle, num_workers=cfg.dataset_train.num_workers,
        pin_memory=cfg.dataset_train.pin_memory)
    val_loader = data.DataLoader(
        val_dataset, collate_fn=val_dataset.collate_fn, batch_size=cfg.val.batch_size, shuffle=cfg.dataset_val.shuffle,
        num_workers=cfg.dataset_val.num_workers, pin_memory=cfg.dataset_val.pin_memory)

    return train_dataset, val_dataset, train_loader, val_loader
