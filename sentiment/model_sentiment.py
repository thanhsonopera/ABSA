import torch
import torch.nn as nn
from transformers import AutoModel
from torch import Tensor


class SentimentClassifier(nn.Module):
    def __init__(self, config):
        super(SentimentClassifier, self).__init__()

        self.config = config

        self.bert = AutoModel.from_pretrained(
            config['name_model'], output_hidden_states=True)

        self.dropout = nn.Dropout(config['drop_rate'][0])

        if config['model'] == 2:
            self.layer_norm = nn.LayerNorm(3072).to(config['device'])
            self.fc = nn.Linear(3072, 256).to(config['device'])
            self.layer_norm2 = nn.LayerNorm(256).to(config['device'])
            self.dropout2 = nn.Dropout(config['drop_rate'][1])
            self.fcs = [nn.Linear(256, 4).to(config['device'])
                        for _ in range(config['num_classes'])]

        elif config['model'] == 1:
            self.fcs = [nn.Linear(3072, 4).to(config['device'])
                        for _ in range(config['num_classes'])]

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if self.config['model'] == 3:
            _, pooled_output = self.bert(input_ids=input_ids,
                                         attention_mask=attention_mask, return_dict=False)

        else:
            berto = self.bert(input_ids=input_ids,
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask)

            hidden_states = berto.hidden_states

            pooled_output = torch.cat(
                tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)

            pooled_output = pooled_output[:, 0, :]

        if self.config['model'] == 2:
            if self.config['layer_norm']:
                pooled_output = self.layer_norm(pooled_output)
            pooled_output = self.dropout(pooled_output)
            pooled_output = self.fc(pooled_output)
            pooled_output = self.layer_norm2(pooled_output)
            pooled_output = torch.nn.Tanh()(pooled_output)
            pooled_output = self.dropout2(pooled_output)

        elif (self.config['model'] == 1) or (self.config['model'] == 3):
            pooled_output = self.dropout(pooled_output)

        if self.config['losses'] == 1:
            outputs = [nn.Softmax(dim=1)(fc(pooled_output)) for fc in self.fcs]
        else:
            outputs = [fc(pooled_output) for fc in self.fcs]

        return outputs
