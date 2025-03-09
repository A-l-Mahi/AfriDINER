from torch import nn
from transformers import pipeline 
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data_utills import *
from component import *
import pandas as pd 

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
