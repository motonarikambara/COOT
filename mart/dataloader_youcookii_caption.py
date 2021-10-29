from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import pickle
import re
import random
import io

class Youcook_Caption_DataLoader(Dataset):
    """Youcook dataset loader."""
    def __init__(
            self,
            csv,
            data_path,
            features_path,
            tokenizer,
            feature_framerate=1.0,
            max_words=30,
            max_frames=100,
    ):
        """
        Args:
        """
        # csv: video_id, feature_file
        # data_dict: video_id, end, start, caption
        # feature_dict: feature_file, feature(sec*1024)
        self.csv = pd.read_csv(csv)
        self.data_dict = pickle.load(open(data_path, 'rb'))
        self.feature_dict = pickle.load(open(features_path, 'rb'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        self.feature_size = self.feature_dict[self.csv["feature_file"].values[0]].shape[-1]

        # Get iterator video ids
        video_id_list = [itm for itm in self.csv['video_id'].values]
        self.video_id2idx_dict = {video_id: id for id, video_id in enumerate(video_id_list)}
        # Get all captions
        self.iter2video_pairs_dict = {}
        iter_idx_ = 0
        for video_id in video_id_list:
            data_dict = self.data_dict[video_id]
            n_caption = len(data_dict['start'])
            # for sub_id in range(n_caption):
            for sub_id in range(1, n_caption):
                self.iter2video_pairs_dict[iter_idx_] = (video_id, sub_id)
                iter_idx_ += 1

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def _get_text(self, video_id, sub_id):
        data_dict = self.data_dict[video_id]
        k = 1
        r_ind = [sub_id]

        starts = np.zeros(k)
        ends = np.zeros(k)
        start_f = np.zeros(k)
        end_f = np.zeros(k)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i in range(k):
            ind = r_ind[i]
            start_f[i], end_f[i] = data_dict['start'][ind], data_dict['end'][ind]
            start_, end_ = data_dict['start'][ind - 1], data_dict['end'][ind - 1]
            starts[i], ends[i] = start_, end_
            total_length_with_CLS = self.max_words - 1
            words = self.tokenizer.tokenize(data_dict['transcript'][ind])

            words = ["[CLS]"] + words
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

            # For generate captions
            caption_words = self.tokenizer.tokenize(data_dict['text'][ind])
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels,\
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, starts, ends, start_f, end_f

    def _get_video(self, idx, s, e, s_f, e_f):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(s)

        video_features = self.feature_dict[self.csv["feature_file"].values[idx]]
        video = np.zeros((len(s), self.max_frames, self.feature_size), dtype=np.float)
        for i in range(len(s)):
            start = int(s[i] * self.feature_framerate)
            end = int(e[i] * self.feature_framerate) + 1
            video_slice = video_features[start:end]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}, start: {}, end: {}".format(self.csv["video_id"].values[idx], start, end))
                # pass
            else:
                video[i][:slice_shape[0]] = video_slice


        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        ### Future video
        video_f = np.zeros((len(s_f), self.max_frames, self.feature_size), dtype=np.float)
        video_mask_f = np.zeros((len(s_f), self.max_frames), dtype=np.long)
        max_video_length_f = [0] * len(s_f)
        for i in range(len(s_f)):
            start = int(s_f[i] * self.feature_framerate)
            end = int(e_f[i] * self.feature_framerate) + 1
            video_slice = video_features[start:end]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length_f[i] = max_video_length_f[i] if max_video_length_f[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}, start_f: {}, end_f: {}".format(self.csv["video_id"].values[idx], start, end))
                # pass
            else:
                video_f[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length_f):
            video_mask_f[i][:v_length] = [1] * v_length
        
        masked_video_f = video_f.copy()
        for i, video_pair_ in enumerate(masked_video_f):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length_f[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_video_f[i][j] = [0.] * video_f.shape[-1]
        ###

        return video, video_mask, masked_video, video_labels_index, video_f, video_mask_f, masked_video_f

    def __getitem__(self, feature_idx):

        video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
        idx = self.video_id2idx_dict[video_id]

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, pairs_input_caption_ids, \
        pairs_decoder_mask, pairs_output_caption_ids, starts, ends, start_f, end_f = self._get_text(video_id, sub_id)

        video, video_mask, masked_video, video_labels_index, video_f, video_mask_f, masked_video_f= self._get_video(idx, starts, ends, start_f, end_f)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, video_f, \
               video_mask_f, masked_video_f