"""
MART model.

"""
import copy
import logging
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard.summary import video

from mart.configs_mart import MartConfig, MartPathConst
from mart.masked_transformer import MTransformer
from mart.loss_caption import LabelSmoothingLoss
from nntrainer.utils_torch import count_parameters


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# # default infinity (config.inf = 0), works with fp32. this can lead to NaN values in some circumstances
INF = float('inf')


# # this should be "infinite enough" for -INF to give 0 for masked softmax attention values.
# INF = 1e19
# for fp16 need something like 255


def create_mart_model(cfg: MartConfig, vocab_size: int, cache_dir: str = MartPathConst.CACHE_DIR,
                      verbose: bool = True) -> nn.Module:
    """
    Args:
        cfg: Experiment config.
        vocab_size: Vocabulary, calculated in mart as len(train_set.word2idx).
        cache_dir: Cache directory.
        verbose: Print model name and number of parameters.

    Returns:
        MART model.
    """
    cfg.max_position_embeddings = cfg.max_v_len + cfg.max_t_len
    cfg.vocab_size = vocab_size
    if cfg.recurrent:
        if cfg.xl:
            # logger.info(f"Use recurrent model - TransformerXL with gradient {cfg.xl_grad}")
            # model = TransformerXL(cfg)
            print("XL should be false")
            sys.exit()
        else:
            logger.info("Use recurrent model - Mine")
            model = RecursiveTransformer(cfg)
    if cfg.use_glove:
        if hasattr(model, "embeddings"):
            logger.info("Load GloVe as word embedding")
            model.embeddings.set_pretrained_embedding(torch.from_numpy(torch.load(
                Path(cache_dir) / f"{cfg.dataset_train.name}_vocab_glove.pt")).float(), freeze=cfg.freeze_glove)
        else:
            logger.warning("This model has no embeddings, cannot load glove vectors into the model")

    # output model properties
    if verbose:
        print(f"Model: {model.__class__.__name__}")
        count_parameters(model)
        if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
            count_parameters(model.embeddings.word_embeddings)

    return model


def gelu(x):
    """
    Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    Pytorch公式実装のgeluで良さそう
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    """
    MultiHead Attention層
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:
        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    """
    TransformerにおけるFF層
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    """
    TransformerにおけるEncoder Layer
    """
    def __init__(self, config):
        super().__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Intermediate(nn.Module):
    """
    geluを用いた1層線形変換
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    """
    GeneratorにおけるFF層
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0, decoder=False):
    """
    Args:
        input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
        max_v_len: int, the first `max_v_len` is for video and its padding, the length
            of the rest of the bits is `max_t_len`. We have L = `max_v_len` + `max_t_len`.
            Note max_v_len may also include the memory len (M), thus max_v_len += M
        max_t_len: int
        memory_len: int, M
    Returns:

    >>> max_v_len_ = 2
    >>> max_t_len_ = 3
    >>> input_mask_ = torch.randn(2, 5)
    >>> make_pad_shifted_mask(input_mask_, max_v_len_, max_t_len_)[0]
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])
    """
    bsz, seq_len = input_mask.shape
    assert max_v_len + max_t_len + memory_len == seq_len
    shifted_mask = input_mask.new_zeros(bsz, max_v_len + max_t_len, seq_len)  # (N, L, M+L)
    shifted_mask[:, :, :memory_len + max_v_len] = 1
    shifted_mask[:, max_v_len:, memory_len + max_v_len:] =\
        torch.tril(input_mask.new_ones(max_t_len, max_t_len), diagonal=0)
    if decoder:
        shifted_mask = torch.ones(shifted_mask.size())
    return shifted_mask


def make_pad_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0, decoder=False):
    """
    input_mask: (N, L),
    """
    shifted_mask =\
        make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=memory_len, decoder=False)
    # It's correct to use `input_mask.unsqueeze(1)' instead of
    # `torch.bmm(input_mask.unsqueeze(2), input_mask.unsqueeze(1))'
    # since the rest of the bits are still masked in the subsequent processing steps.
    pad_shifted_mask = shifted_mask * input_mask.unsqueeze(1)
    return pad_shifted_mask


def make_video_only_mask(input_mask, max_v_len):
    video_only_mask = copy.deepcopy(input_mask)
    video_only_mask[:, max_v_len:] = 0
    return video_only_mask



class LayerWithMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = Attention(config)
        self.memory_initilizer = MemoryInitializer(config)
        self.memory_updater = MemoryUpdater(config)
        self.memory_augmented_attention = SelfAttention(config)
        self.hidden_intermediate = Intermediate(config)
        self.memory_projection = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output = Output(config)

    def forward(self, prev_m, hidden_states, attention_mask):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        max_v_len, max_t_len = self.config.max_v_len, self.config.max_t_len
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)
        intermediate_output = self.hidden_intermediate(attention_output)

        if prev_m is None:
            # only allow the initializer to see video part, not text part,
            # as it will be used for generation at current step
            init_memory_mask = make_video_only_mask(attention_mask, max_v_len)
            prev_m = self.memory_initilizer(intermediate_output, init_memory_mask)  # (N, L, Di)

        # update memory, use raw attention_mask, no need to hide the text
        updated_m = self.memory_updater(prev_m, intermediate_output, attention_mask)  # (N, M, Di)

        concat_mh = torch.cat([prev_m, intermediate_output], dim=1)  # [(N, M, Di); (N, L, Di)] => [N, M+L, Di]
        bsz, n_memory_cells = prev_m.shape[:2]
        raw_memory_attention_mask = torch.cat(
            [attention_mask.new_ones(bsz, n_memory_cells), attention_mask], -1)  # (N, M+L)
        memory_attention_mask = make_pad_shifted_mask(
            raw_memory_attention_mask, max_v_len, max_t_len, memory_len=n_memory_cells)
        memory_attention_output = self.memory_augmented_attention(
            intermediate_output, concat_mh, concat_mh, memory_attention_mask)  # (N, L, Di)
        memory_attention_output = self.memory_projection(memory_attention_output)  # (N, L, Di) -> (N, L, D)

        layer_output = self.output(memory_attention_output, attention_output)  # (N, L, D)

        return updated_m, layer_output


class DecoderOutput(nn.Module):
    """
    DecoderにおけるFF層
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EncoderWithMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([LayerWithMemory(config) for _ in range(config.num_hidden_layers)])

    def forward(self, prev_ms, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            prev_ms: [(N, M, D), ] * num_hidden_layers or None at first step. Memory states for each layer
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:

        Returns:
        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            prev_ms[layer_idx], hidden_states = layer_module(prev_ms[layer_idx], hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return prev_ms, all_encoder_layers


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = Attention(config)
        self.output = DecoderOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        max_v_len, max_t_len = self.config.max_v_len, self.config.max_t_len
        # self-attention, need to shift right
        shifted_self_mask =\
            make_pad_shifted_mask(attention_mask, max_v_len, max_t_len, decoder=True)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)

        layer_output = self.output(attention_output, attention_output)  # (N, L, D)

        return layer_output


class Decoder(nn.Module):
    def __init__(self, config, num_hidden_layers=3):
        super().__init__()
        self.layer = nn.ModuleList([DecoderLayer(config) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:

        Returns:
        """
        all_decoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            all_decoder_layers.append(hidden_states)
        return all_decoder_layers



class EmbeddingsWithVideo(nn.Module):
    """
    Construct the embeddings from word (+ video), position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    """

    def __init__(self, config, add_postion_embeddings=True):
        super().__init__()
        """
        add_postion_embeddings: whether to add absolute positional embeddings
        """
        self.add_postion_embeddings = add_postion_embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
            LayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.video_embeddings = nn.Sequential(
            LayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
                                                        max_len=config.max_position_embeddings * 2)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """
        Note the from_pretrained does not work in-place, so you need to assign value to the embedding
        """
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)

    def forward(self, input_ids, video_features, token_type_ids):
        """
        Args:
            input_ids: (N, L)
            video_features: (N, L, D)
            token_type_ids: (N, L, D)

        Returns:
        """
        words_embeddings = self.word_fc(self.word_embeddings(input_ids))
        video_embeddings = self.video_embeddings(video_features)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # print("words", words_embeddings.shape, "vid", video_embeddings.shape, "token",token_type_embeddings.shape)
        words_embeddings += token_type_embeddings
        embeddings = words_embeddings + video_embeddings + token_type_embeddings
        # embeddings = torch.cat([words_embeddings, video_embeddings], dim=1)

        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (N, L, D)


class MemoryInitializer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # init memory
        self.n_memory_cells = config.n_memory_cells
        self.init_memory_bias = nn.Parameter(
            torch.randn(1, config.n_memory_cells, 1))  # (1, M, D)
        self.init_memory_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            LayerNorm(config.hidden_size),
            nn.Dropout(config.memory_dropout_prob)
        )

    def forward(self, input_states, attention_mask):
        """
        initialize the model with the first input states
            input_states: (N, L, D)
            attention_mask: (N, L)
        """
        pooled_input_states = torch.sum(input_states * attention_mask.unsqueeze(-1), dim=1)  # (N, D)
        pooled_input_states = pooled_input_states / attention_mask.sum(1, keepdim=True)  # (N, D) no zero here
        pooled_input_states = pooled_input_states.unsqueeze(1).repeat(1, self.n_memory_cells, 1)  # (N, M, D)
        pooled_input_states = pooled_input_states + self.init_memory_bias  # (N, M, D)
        init_memory = self.init_memory_fc(pooled_input_states)  # (N, M, D)
        return init_memory


class MemoryUpdater(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory_update_attention = SelfAttention(config)

        self.mc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sc = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.mz = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sz = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, prev_m, input_states, attention_mask):
        """
        This module should have access to all the text at this step,
        since its state will not be used for generation at current step
        Args:
            prev_m: (N, M, D), M is memory size
            input_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        # memory attended inputs
        n_memory_cells = prev_m.shape[1]
        update_mask = attention_mask.unsqueeze(1).repeat(1, n_memory_cells, 1)  # (N, M, L)
        s_t = self.memory_update_attention(prev_m, input_states, input_states, update_mask)  # (N, M, D),

        c_t = torch.tanh(self.mc(prev_m) + self.sc(s_t))  # (N, M, D)

        z_t = torch.sigmoid(self.mz(prev_m) + self.sz(s_t))  # (N, M, D)

        updated_memory = (1 - z_t) * c_t + z_t * prev_m  # (N, M, D)
        return updated_memory


class PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """
        (N, L, D)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights=None):
        super().__init__()
        self.transform = PredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None,\
                "bert_model_embedding_weights should not be None "\
                "when setting --share_wd_cls_weight flag to be true"
            assert config.hidden_size == bert_model_embedding_weights.size(1),\
                "hidden size has be the same as word embedding size when "\
                "sharing word embedding weight and classifier weight"
            self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                     bert_model_embedding_weights.size(0),
                                     bias=False)
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """
        (N, L, D)
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output



# MART model
class RecursiveTransformer(nn.Module):
    def __init__(self, cfg: MartConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = EmbeddingsWithVideo(cfg, add_postion_embeddings=True)
        self.encoder = EncoderWithMemory(cfg)
        decoder_classifier_weight = self.embeddings.word_embeddings.weight\
            if self.cfg.share_wd_cls_weight else None
        self.decoder = LMPredictionHead(cfg, decoder_classifier_weight)
        self.transformerdecoder = Decoder(cfg)
        if self.cfg.label_smoothing != 0:
            self.loss_func = LabelSmoothingLoss(cfg.label_smoothing, cfg.vocab_size, ignore_index=-1)
        else:
            self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

        # self.future_linear = nn.Sequential(nn.Linear(384, 786),
        #                                     nn.GELU(),
        #                                     nn.Linear(786, 384),
        #                                     nn.GELU())
        # self.future_linear = nn.Linear(384, 384)
        # self.future_loss = nn.MSELoss()
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_step(self, prev_ms, input_ids, video_features, input_masks,
                     token_type_ids):
        """
        single step forward in the recursive structure
        """
        embeddings = self.embeddings(input_ids, video_features, token_type_ids)  # (N, L, D)

        prev_ms, encoded_layer_outputs = self.encoder(
            prev_ms, embeddings, input_masks, output_all_encoded_layers=False)  # both outputs are list
        decoded_layer_outputs = self.transformerdecoder(encoded_layer_outputs[-1], input_masks)
        prediction_scores = self.decoder(decoded_layer_outputs[-1])  # (N, L, vocab_size)
        return prev_ms, encoded_layer_outputs, prediction_scores

    #ver. future
    def forward(self, input_ids_list, video_features_list, input_masks_list,
                token_type_ids_list, input_labels_list, gt_clip, return_memory=False):
        """
        Args:
            input_ids_list: [(N, L)] * step_size
            video_features_list: [(N, L, D_v)] * step_size
            input_masks_list: [(N, L)] * step_size with 1 indicates valid bits
            token_type_ids_list: [(N, L)] * step_size, with `0` on the first `max_v_len` bits,
                `1` on the last `max_t_len`
            input_labels_list: [(N, L)] * step_size, with `-1` on ignored positions,
                will not be used when return_memory is True, thus can be None in this case
            return_memory: bool,

        Returns:
        """
        # _, video_feature = self.lstm(video_features_list)
        # [(N, M, D)] * num_hidden_layers, initialized internally
        prev_ms = [None] * self.cfg.num_hidden_layers
        step_size = len(input_ids_list)
        memory_list = []  # [(N, M, D)] * num_hidden_layers * step_size
        encoded_outputs_list = []  # [(N, L, D)] * step_size
        prediction_scores_list = []  # [(N, L, vocab_size)] * step_size
        future_loss = 0
        for idx in range(step_size):
            prev_ms, encoded_layer_outputs, prediction_scores =\
            self.forward_step(prev_ms, input_ids_list[idx], video_features_list[idx],
                                input_masks_list[idx], token_type_ids_list[idx])           
            memory_list.append(prev_ms)
            encoded_outputs_list.append(encoded_layer_outputs)
            prediction_scores_list.append(prediction_scores)

        if return_memory:  # used to analyze memory
            return memory_list
        else:  # normal training/evaluation mode
            # compute loss, get predicted words
            caption_loss = 0.0
            for idx in range(step_size):
                tmp_loss =  self.loss_func(prediction_scores_list[idx].view(-1, self.cfg.vocab_size),
                                               input_labels_list[idx].view(-1))
                caption_loss += 0.1 * tmp_loss
                caption_loss += future_loss
            return caption_loss, prediction_scores_list
