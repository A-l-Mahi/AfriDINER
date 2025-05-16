from torch import nn
from transformers import pipeline 
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data_utills import *
from component import *
import pandas as pd 

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim=None, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        ff_dim = ff_dim or hidden_dim * 4  # Default to 4x hidden size
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class ABSAmodel(nn.Module):
    def __init__(self, pretrained):
        super(ABSAmodel, self).__init__()
        self.model = pretrained
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.model.config.hidden_size, 3)

    def forward(self, text_input_ids, text_attention_mask):
        outputs = self.model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )

        if "pooler_output" in outputs:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]

        dropped_output = self.drop(pooled_output)
        return self.out(dropped_output)
class ABSAmodell(nn.Module):
    def __init__(self, pretrained, dropout=0.1):
        super(ABSAmodell, self).__init__()
        self.model = pretrained
        hidden_dim  = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.attn_k = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.attn_q = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)

        self.ffn_c = PositionwiseFeedForward(hidden_dim, dropout=dropout)
        self.ffn_t = PositionwiseFeedForward(hidden_dim, dropout=dropout)

        self.attn_s1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)

        self.out = nn.Linear(hidden_dim * 3, 3)

    def forward(self, text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask):
        text_outputs = self.model(input_ids=text_input_ids, attention_mask=text_attention_mask, return_dict=True)
        aspect_outputs = self.model(input_ids=aspect_input_ids, attention_mask=aspect_attention_mask, return_dict=True)

        # Use full hidden states instead of just [CLS]
        text_emb = self.dropout(text_outputs.last_hidden_state)  # [batch_size, seq_len, hidden_dim]
        aspect_emb = self.dropout(aspect_outputs.last_hidden_state)  # [batch_size, seq_len, hidden_dim]

        hc, _ = self.attn_k(text_emb, text_emb, text_emb)
        hc = self.ffn_c(hc)

        ht, _ = self.attn_q(aspect_emb, aspect_emb, aspect_emb)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht, ht)

        # Mean pooling across sequence length
        hc_mean = hc.mean(dim=1)
        ht_mean = ht.mean(dim=1)
        s1_mean = s1.mean(dim=1)

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)

        return self.out(x)

class CFABSAmodel(nn.Module):
    def __init__(self, pretrained):
        super(CFABSAmodel, self).__init__()
        self.model_all = pretrained
        self.out_all = nn.Linear(self.model_all.config.hidden_size, 3)
        self.model_aspect = pretrained
        self.out_aspect = nn.Linear(self.model_aspect.config.hidden_size, 3)
        self.model_text = pretrained
        self.clf = tde_classifier(num_classes=3, feat_dim=self.model_all.config.hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=0.1)
    def forward(self, all_input_ids, all_attention_mask,text_input_ids, text_attention_mask,aspect_input_ids, aspect_attention_mask,labels = None):
        if labels == None:
            #测试阶段
            all_out = self.model_all(
                input_ids=all_input_ids,
                attention_mask=all_attention_mask
                )["pooler_output"]
            all_out = self.out_all(all_out) 

            text_out = self.model_text(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
                )["pooler_output"]
            text_out = self.clf(text_out)

            aspect_out = self.model_aspect(
                input_ids=aspect_input_ids,
                attention_mask=aspect_attention_mask
                )["pooler_output"]
            aspect_out = self.out_aspect(aspect_out)
            logits = self.softmax(all_out + torch.tanh(text_out) + torch.tanh(aspect_out))
            return logits
        else:
            all_returned = self.model_all(
                input_ids=all_input_ids,
                attention_mask=all_attention_mask
                )
            all_pooled_output = all_returned["pooler_output"]
            all_output = self.drop(all_pooled_output)
            all_out = self.out_all(all_output)

            text_returned = self.model_text(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
                )
            text_pooled_output = text_returned["pooler_output"]
            text_out = self.clf(text_pooled_output)

            aspect_returned = self.model_aspect(
                input_ids=aspect_input_ids,
                attention_mask=aspect_attention_mask
                )
            aspect_pooled_output = aspect_returned["pooler_output"]
            aspect_output = self.drop(aspect_pooled_output)
            aspect_out = self.out_aspect(aspect_output)
            logits = all_out + torch.tanh(text_out) + torch.tanh(aspect_out)
            return logits, all_out, text_out, aspect_out

class CFABSA_XLMR(nn.Module):
    def __init__(self, pretrained):
        super(CFABSA_XLMR, self).__init__()
        self.model_all =  pretrained
        self.out_all = nn.Linear(self.model_all.config.hidden_size, 3)
        self.model_aspect =  pretrained
        self.out_aspect = nn.Linear(self.model_aspect.config.hidden_size, 3)
        self.model_text =  pretrained
        self.clf = tde_classifier(num_classes=3, feat_dim=self.model_all.config.hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=0.1)

    def get_cls_output(self, model, input_ids, attention_mask):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]  # XLM-R does not have 'pooler_output'

    def forward(self, all_input_ids, all_attention_mask,text_input_ids, text_attention_mask,aspect_input_ids, aspect_attention_mask,labels = None):
        if labels == None:
            #测试阶段
            all_out = self.get_cls_output(
                self.model_all, all_input_ids, all_attention_mask
            )
            all_out = self.out_all(all_out)

            text_out = self.get_cls_output(
                self.model_text, input_ids = text_input_ids, attention_mask = text_attention_mask
            )
            text_out = self.clf(text_out)

            aspect_out = self.get_cls_output(
                self.model_aspect, aspect_input_ids, aspect_attention_mask
            )
            aspect_out = self.out_aspect(aspect_out)

            logits = self.softmax(all_out + torch.tanh(text_out) + torch.tanh(aspect_out))
            return logits
        else:
            all_out = self.out_all(
                self.drop(
                    self.get_cls_output(
                        self.model_all, all_input_ids, all_attention_mask)
                        )
                    )

            all_returned = self.model_all(
                input_ids=all_input_ids,
                attention_mask=all_attention_mask
                )

            text_out = self.clf(
                self.get_cls_output(
                    self.model_text, text_input_ids, text_attention_mask))

            aspect_out = self.out_aspect(
                self.drop(
                    self.get_cls_output(
                        self.model_aspect, aspect_input_ids, aspect_attention_mask)))

            logits = all_out + torch.tanh(text_out) + torch.tanh(aspect_out)
            return logits, all_out, text_out, aspect_out
