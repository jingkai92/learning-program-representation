import operator
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pymodels.submodels.positionalencoding import PositionalEncoding
from tokenizer.vocab_dict import TokenizationConstant
from utils.pymodels_util import convert_logit_to_indices, to_cuda, compute_loss


class TransformerEncoderModel(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderModel, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.vocab_len = self.config.token_vocab_dict.vocab_size()
        self.word_emb_dim = self.config.word_emb_dims
        self.embedding = nn.Embedding(self.vocab_len, self.word_emb_dim)

        # Model Specified Parameters
        self.d_model = self.config.transformer['d_model']
        self.pos_dropout = self.config.transformer['pos_dropout']
        self.nhead = self.config.transformer['nhead']
        self.enc_n_layers = self.config.transformer['enc_nlayers']
        self.dim_feedforward = self.config.transformer['dim_feedforward']
        self.trans_dropout = self.config.transformer['tf_dropout']

        self.embedding = nn.Embedding(self.vocab_len, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model, self.pos_dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                       dropout=self.trans_dropout,
                                       dim_feedforward=self.dim_feedforward), num_layers=self.enc_n_layers)
        self.fforward = nn.Linear(self.d_model, self.config.class_num)
        if self.use_cuda:
            self.pos_enc = self.pos_enc.cuda()
            self.embedding = self.embedding.cuda()
            self.transformer_encoder = self.transformer_encoder.cuda()
            self.fforward = self.fforward.cuda()

    def forward(self, batch_dict, running_mode, loss_fn):
        # src = [bsz, seq_len]
        fn_tensors = to_cuda(batch_dict['fn_tensors'], use_cuda=self.use_cuda)
        fn_len = batch_dict['funcs_lens']
        class_target = batch_dict['tgt']
        pos_embed_src = self._embed_seq(fn_tensors)
        src_pad_mask = ~self._create_pad_mask(fn_tensors)  # Mask of non-transposed source, padding mask
        trans_encoded_src = self.transformer_encoder(pos_embed_src, src_key_padding_mask=src_pad_mask)
        encoded_src = trans_encoded_src.permute(1, 0, 2)
        last_seq_idxs = torch.LongTensor([x - 1 for x in fn_len])
        last_seq_items = encoded_src[range(encoded_src.shape[0]), last_seq_idxs, :]
        dense_output = F.leaky_relu(self.fforward(last_seq_items))

        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(torch.LongTensor(class_target), use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss

    def _embed_seq(self, seq):
        """
        Embed seq or tgt through self.embedding and positional encoding
        :param seq: A [bsz, seq_len] tensor
        :return: Return a Embedded Source
        """
        src_transposed = seq.transpose(1, 0)  # [seq_len, bsz]
        src_transposed_emb = self.embedding(src_transposed)  # [seq_len, bsz, self.d_model]
        if self.config.strong_pos_enc:
            src_transposed_emb = src_transposed_emb
        else:
            src_transposed_emb = src_transposed_emb * math.sqrt(self.d_model)
        src_pos_emb = self.pos_enc(src_transposed_emb)  # [seq_len, bsz, self.d_model]
        return src_pos_emb

    def _create_pad_mask(self, seq):
        # 3 is token of PAD
        return to_cuda((seq != 3).bool(), self.use_cuda)