import transformer
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F


class TFModel(nn.Module):
    def __init__(self,
                 word2vec,
                 vocab_size,
                 hidden_size,
                 max_len,
                 pad_index,
                 cls_index,
                 num_class=2,
                 num_heads=4,
                 num_layers=1,
                 dropout=0.0,
                 attn_dropout=0.0,
                 ):
        super().__init__()
        self.word_embs = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_index)
        self.position_embs = nn.Embedding(max_len + 1, hidden_size)

        self.net = transformer.TransformerEncoder(
            hidden_size,
            num_layers,
            num_heads,
            4 * hidden_size,
            attn_dropout,
            dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.prj = nn.Linear(hidden_size, num_class)

        self.cls_index = cls_index
        self.pad_index = pad_index
        self.word2vec = word2vec
        nn.init.normal_(self.word_embs.weight, std=.02)
        nn.init.normal_(self.position_embs.weight, std=.02)

        self.apply(self._init_weights)
        self.word_embs.weight.data.copy_(torch.from_numpy(word2vec))
        self.word_embs.weight.requires_grad = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_ids):

        batch_size, device = input_ids.size(0), input_ids.device

        cls_tokens = torch.zeros((batch_size, 1)).to(input_ids.device) + self.cls_index
        cls_tokens = cls_tokens.long()
        input_ids = torch.cat((cls_tokens, input_ids), dim=1)

        embs = self.word_embs(input_ids)

        seq_len = embs.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids[None, :]
        position_embs = self.position_embs(position_ids)

        embs = embs + position_embs
        embs = self.dropout(embs)

        attention_mask = (input_ids == self.pad_index)  # (batch_size, seq_len)
        hidden_states = self.net(embs, attention_mask)

        cls_hidden_state = hidden_states[:, 0, :]
        cls_hidden_state = self.dropout(cls_hidden_state)

        output = self.prj(cls_hidden_state)
        output = F.log_softmax(output, dim=-1)

        return output
