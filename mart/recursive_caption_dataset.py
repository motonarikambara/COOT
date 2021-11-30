"""
Captioning dataset.
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


class DataTypesConstCaption(ConstantHolder):
    """
    Possible video data types for the dataset:
    Video features or COOT embeddings.
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

    def __init__(
        self,
        dset_name: str,
        max_t_len,
        max_v_len,
        max_n_sen,
        mode="train",
        recurrent=True,
        untied=False,
        video_feature_dir: Optional[str] = None,
        coot_model_name=None,
        coot_mode="all",
        coot_dim_vid=768,
        coot_dim_clip=384,
        annotations_dir: str = "annotations",
        coot_feat_dir="provided_embeddings",
        dataset_max: Optional[int] = None,
        preload: bool = False,
    ):
        # metadata settings
        self.dset_name = dset_name
        self.annotations_dir = Path(annotations_dir)

        # COOT feature settings
        self.coot_model_name = coot_model_name
        self.coot_mode = coot_mode  # "clip" for only clip repr
        self.coot_dim_vid = coot_dim_vid
        self.coot_dim_clip = coot_dim_clip
        self.coot_feat_dir = Path(coot_feat_dir)

        # Video feature settings
        duration_file = "captioning_video_feat_duration.csv"
        self.video_feature_dir = Path(video_feature_dir) / self.dset_name
        self.duration_file = self.annotations_dir / self.dset_name / duration_file
        self.word2idx_file = (
            self.annotations_dir / self.dset_name / "mart_word2idx.json"
        )
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
        assert not (
            self.recurrent and self.untied
        ), "untied and recurrent cannot be True for both"

        # ---------- Load metadata ----------

        # determine metadata file
        tmp_path = "ponnet_data"
        if mode == "train":  # 1333 videos
            data_path = self.annotations_dir / tmp_path / "captioning_train.json"
        elif mode == "val":  # 457 videos
            data_path = self.annotations_dir / tmp_path / "captioning_val.json"
        else:
            raise ValueError(
                f"Mode must be [train, val] for {self.dset_name}, got {mode}"
            )

        # load and process captions and video data
        # clip_num, annnotation
        raw_data = json.load(data_path)
        coll_data = []
        for i, (k, line) in enumerate(tqdm(list(raw_data.items()))):
            if dataset_max is not None and i >= dataset_max > 0:
                break
            line["name"] = k
            # line: annotation
            coll_data.append(line)
        self.data = coll_data

        # ---------- Load video data ----------

        # Decide whether to load COOT embeddings or video features
        # COOT embeddings
        self.data_type = DataTypesConstCaption.COOT_EMB
        # map video id and clip id to clip number
        self.clip_nums = []
        for clip in tqdm(range(len(self.data))):
            clip.append(str(clip))

        self.frame_to_second = None  # Don't need this for COOT embeddings

        print(
            f"Dataset {self.dset_name} #{len(self)} {self.mode} input {self.data_type}"
        )

        self.preloading_done = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta

    def _load_ponnnet_video_feature(
        self, raw_name: str
    ) -> Tuple[np.array, np.array, List[np.array]]:
        """
        Load given S3D video features.
        Args:
            raw_name: Video ID
        Returns:
            Tuple of:
                video with shape (dim_video)
                context with shape (dim_clip)
                clips with shape (dim_clip)
        """
        # 動画に関する特徴量を取得
        feat_file = raw_name + ".npy"
        emb_feat = np.load(feat_file)
        return emb_feat

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
        # ver. future
        emb_feat = self._load_ponnet_video_feature(
            raw_name
        )
        video_feature = emb_feat
        single_video_features = []
        single_video_meta = []
        # cur_data:video特徴量を含むdict
        cur_data, cur_meta = self.clip_sentence_to_feature(
            example["name"],
            example["sentences"],
            video_feature,
        )
        # single_video_features: video特徴量を含むdict
        single_video_features.append(cur_data)
        single_video_meta.append(cur_meta)
        return single_video_features, single_video_meta

    def clip_sentence_to_feature(
        self,
        name,
        sentence,
        video_feature,
    ):
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

        # future
        feat, video_tokens, video_mask = self._load_indexed_video_feature(
            video_feature, frm2sec
        )
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        input_tokens = video_tokens + text_tokens

        input_ids = [
            self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens
        ]
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        input_labels = (
            [self.IGNORE] * len(video_tokens)
            + [
                self.IGNORE if m == 0 else tid
                for tid, m in zip(input_ids[-len(text_mask) :], text_mask)
            ][1:]
            + [self.IGNORE]
        )
        input_mask = video_mask + text_mask
        token_type_ids = [0] * self.max_v_len + [1] * self.max_t_len

        # ver. future
        coll_data = dict(
            name=name,
            input_tokens=input_tokens,
            input_ids=np.array(input_ids).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64),
            input_mask=np.array(input_mask).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
            video_feature=feat.astype(np.float32),
        )
        meta = dict(name=name, sentence=sentence)
        return coll_data, meta

    def _get_vt_features(
        self, video_feat_tuple, max_v_l
    ):
        # ひとまとめにしたvideo関連の特徴量から必要なものを抽出
        clip_feat = video_feat_tuple
        # only clip (1, 384)
        valid_l = 0
        feat = np.zeros((max_v_l, self.coot_dim_clip))
        feat[valid_l] = clip_feat
        valid_l += 1
        return feat, valid_l

    # future
    def _load_indexed_video_feature(
        self, raw_feat, frm2sec
    ):
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
        raw_feat, valid_l = self._get_vt_features(
            raw_feat, max_v_l
        )
        video_tokens = (
            [self.CLS_TOKEN]
            + [self.VID_TOKEN] * valid_l
            + [self.SEP_TOKEN]
            + [self.PAD_TOKEN] * (max_v_l - valid_l)
        )
        mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)
        # 上記のように特徴量を配置
        # feat∈25×1152
        # includes [CLS], [SEP]
        feat = np.zeros((self.max_v_len + self.max_t_len, raw_feat.shape[1]))
        # feat[1:len(raw_feat) + 1] = raw_feat
        feat[1:4] = raw_feat
        # includes [CLS], [SEP]
        return feat, video_tokens, mask

    def _tokenize_pad_sentence(self, sentence):
        """
        [BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD],
            len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_t_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[: max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(
        self, ids, rm_padding=True, return_sentence_only=True
    ) -> str:
        """
        A list of token ids
        """
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [
                self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]
            ]
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
        # recurrent collate function. original docstring:
        # HOW to batch clip-sentence pair? 1) directly copy the last
        # sentence, but do not count them in when
        # back-prop OR put all -1 to their text token label, treat

        # collect meta
        raw_batch_meta = [e[1] for e in batch]
        batch_meta = []
        for e in raw_batch_meta:
            cur_meta = dict(name=None, timestamp=[], gt_sentence=[])
            for d in e:
                cur_meta["name"] = d["name"]
                cur_meta["gt_sentence"].append(d["sentence"])
            batch_meta.append(cur_meta)

        batch = [e[0] for e in batch]
        # Step1: pad each example to max_n_sen
        max_n_sen = max([len(e) for e in batch])
        raw_step_sizes = []

        padded_batch = []
        padding_clip_sen_data = copy.deepcopy(
            batch[0][0]
        )  # doesn"t matter which one is used
        padding_clip_sen_data["input_labels"][:] =\
            RecursiveCaptionDataset.IGNORE
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
    cfg: MartConfig,
    coot_feat_dir: str = MartPathConst.COOT_FEAT_DIR,
    annotations_dir: str = MartPathConst.ANNOTATIONS_DIR,
    video_feature_dir: str = MartPathConst.VIDEO_FEATURE_DIR,
) -> Tuple[
    RecursiveCaptionDataset, RecursiveCaptionDataset, data.DataLoader, data.DataLoader
]:
    # create the dataset
    dset_name_train = cfg.dataset_train.name
    train_dataset = RecursiveCaptionDataset(
        dset_name_train,
        cfg.max_t_len,
        cfg.max_v_len,
        cfg.max_n_sen,
        mode="train",
        recurrent=cfg.recurrent,
        untied=cfg.untied or cfg.mtrans,
        video_feature_dir=video_feature_dir,
        coot_model_name=cfg.coot_model_name,
        coot_mode=cfg.coot_mode,
        coot_dim_vid=cfg.coot_dim_vid,
        coot_dim_clip=cfg.coot_dim_clip,
        annotations_dir=annotations_dir,
        coot_feat_dir=coot_feat_dir,
        dataset_max=cfg.dataset_train.max_datapoints,
        preload=cfg.dataset_train.preload,
    )
    # add 10 at max_n_sen to make the inference stage use all the segments
    # max_n_sen_val = cfg.max_n_sen + 10
    max_n_sen_val = cfg.max_n_sen
    val_dataset = RecursiveCaptionDataset(
        cfg.dataset_val.name,
        cfg.max_t_len,
        cfg.max_v_len,
        max_n_sen_val,
        mode="val",
        recurrent=cfg.recurrent,
        untied=cfg.untied or cfg.mtrans,
        video_feature_dir=video_feature_dir,
        coot_model_name=cfg.coot_model_name,
        coot_mode=cfg.coot_mode,
        coot_dim_vid=cfg.coot_dim_vid,
        coot_dim_clip=cfg.coot_dim_clip,
        annotations_dir=annotations_dir,
        coot_feat_dir=coot_feat_dir,
        dataset_max=cfg.dataset_val.max_datapoints,
        preload=cfg.dataset_val.preload,
    )

    train_loader = data.DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.dataset_train.shuffle,
        num_workers=cfg.dataset_train.num_workers,
        pin_memory=cfg.dataset_train.pin_memory,
    )
    val_loader = data.DataLoader(
        val_dataset,
        collate_fn=val_dataset.collate_fn,
        batch_size=cfg.val.batch_size,
        shuffle=cfg.dataset_val.shuffle,
        num_workers=cfg.dataset_val.num_workers,
        pin_memory=cfg.dataset_val.pin_memory,
    )

    return train_dataset, val_dataset, train_loader, val_loader
