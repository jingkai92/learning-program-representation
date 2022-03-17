import torch
import torch.nn as nn
import torch.nn.functional as F


# Should be Bahdaunau Attention (Additive(*))
# A bit different from original Bahdaunau Attention
class Attention(nn.Module):
    def __init__(self, attn_emb_dims, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(attn_emb_dims, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # encoder_output = [bsz, seq_len, 2 * unit]
        src_len = encoder_outputs.shape[1]
        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        cat = torch.cat((hidden, encoder_outputs), dim=2)
        a = self.attn(cat)
        energy = torch.tanh(a)
        attention = self.v(energy).squeeze(2)
        # Return a [bsz, seq_len] of probabilities
        attention = F.softmax(attention, dim=1)
        return attention
