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


class TransformerSummarizeSingleVocabModel(nn.Module):
    def __init__(self, config):
        super(TransformerSummarizeSingleVocabModel, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.vocab_len = self.config.token_vocab_dict.vocab_size()

        self.d_model = self.config.transformer['d_model']
        self.pos_dropout = self.config.transformer['pos_dropout']
        self.nhead = self.config.transformer['nhead']
        self.enc_n_layers = self.config.transformer['enc_nlayers']  # Use the same layer for dec and enc
        self.dec_n_layers = self.config.transformer['enc_nlayers']
        self.dim_feedforward = self.config.transformer['dim_feedforward']
        self.trans_dropout = self.config.transformer['tf_dropout']

        self.embedding = nn.Embedding(self.vocab_len, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model, self.pos_dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                       dropout=self.trans_dropout,
                                       dim_feedforward=self.dim_feedforward), num_layers=self.enc_n_layers)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead,
                                       dim_feedforward=self.dim_feedforward,
                                       dropout=self.trans_dropout), num_layers=self.dec_n_layers)

        self.W1 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W2 = nn.Linear(self.d_model, self.d_model, bias=False)

        # self.linear_dropout = nn.Dropout(self.trans_dropout)
        self.fc = nn.Linear(self.d_model, self.vocab_len)
        if self.use_cuda:
            self.pos_enc = self.pos_enc.cuda()
            self.embedding = self.embedding.cuda()
            self.transformer_encoder = self.transformer_encoder.cuda()
            self.transformer_decoder = self.transformer_decoder.cuda()
            self.fc = self.fc.cuda()

    def forward(self, batch_dict, running_mode, loss_fn):
        # src = [bsz, seq_len]
        src = to_cuda(batch_dict["fn_tensors"], self.use_cuda)

        src = src[:, 1:]  # Removing SOS
        batch_size, src_seq_len = src.shape
        pos_embed_src = self._embed_seq(src)
        src_pad_mask = ~self._create_pad_mask(src)  # Mask of non-transposed source, padding mask

        # Encode the Source with Transformer Encoder
        trans_encoded_src = self.transformer_encoder(pos_embed_src, src_key_padding_mask=src_pad_mask)

        if running_mode in ["train", "val"]:
            tgt = to_cuda(batch_dict["tgt_tensors"], self.use_cuda)
            pos_embed_tgt = self._embed_seq(tgt[:, :-1])  # Remove EOS Symbol
            tgt_pad_mask = ~self._create_pad_mask(tgt[:, :-1])
            tgt_mask = self._generate_square_subsequent_mask(tgt.shape[1] - 1)
            decoded = self.transformer_decoder(pos_embed_tgt, trans_encoded_src,
                                               tgt_mask=tgt_mask,
                                               tgt_key_padding_mask=tgt_pad_mask,
                                               memory_key_padding_mask=src_pad_mask)  # seq_len, bsz, hdim
            logits = self.fc(decoded)
            model_output = convert_logit_to_indices(logits)
            model_output = model_output.transpose(1, 0)
        else:
            tgt_len = self.config.max_sequence_length
            model_output = self.infer(trans_encoded_src, src_pad_mask, batch_size, tgt_len)
            if type(model_output) == list:
                return src.cpu().tolist(), model_output, 0
            else:
                return src.cpu().tolist(), model_output.cpu().tolist(), 0

        if running_mode in ["train", "val"]:
            # tgt = [bsz, seq_len]
            tgt = tgt[:, 1:].transpose(1, 0)
            # if self.config.loss_big:
            #     loss = 0
            #     for i in range(logits.shape[0]):
            #         tmp_loss = compute_loss(loss_fn, logits[i].unsqueeze(0), tgt[i].unsqueeze(0))
            #         loss += tmp_loss
            # else:
            loss = loss_fn(rearrange(logits, 'b t v -> (b t) v'), rearrange(tgt, 'b o -> (b o)'))
        else:
            loss = 0
        return src.cpu().tolist(), model_output.cpu().tolist(), loss

    def infer(self, src, src_pad_mask, bsz, target_seq_size):
        # if self.config.beam_size == 1:  # Greedy Decode
        #     model_output = self.greedy_decode(src, src_pad_mask, bsz, target_seq_size)
        # else:
        #     model_output = self.beam_decode(src, src_pad_mask, bsz, target_seq_size)
        model_output = self.greedy_decode(src, src_pad_mask, bsz, target_seq_size)
        return model_output

    def greedy_decode(self, src, src_pad_mask, bsz, target_seq_size):
        model_output = to_cuda(torch.zeros(bsz, target_seq_size).long(), self.use_cuda)
        logits = to_cuda(torch.zeros(bsz, target_seq_size, self.vocab_len), self.use_cuda)

        for t in range(1, target_seq_size):
            cur_mo = model_output[:, :t]
            tgt_emb = self._embed_seq(cur_mo)
            tgt_mask = self._generate_square_subsequent_mask(t)
            decoder_output = self.transformer_decoder(tgt_emb, src, tgt_mask=tgt_mask,
                                                      memory_key_padding_mask=src_pad_mask)
            pred_proba_t = self.fc(decoder_output)[-1, :, :]
            logits[:, t] = pred_proba_t
            output_t = pred_proba_t.data.topk(1)[1].squeeze()
            model_output[:, t] = output_t
        # return logits.permute(1, 0, 2), model_output
        return model_output

    def _create_pad_mask(self, seq):
        # 3 is token of PAD
        return to_cuda((seq != 3).bool(), self.use_cuda)

    def _generate_square_subsequent_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return to_cuda(mask, self.use_cuda)

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
