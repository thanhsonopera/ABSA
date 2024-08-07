import torch
import torch.nn as nn
from transformers import AutoModel
from torch import Tensor


class BertClassifier(nn.Module):
    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(
            config['name_model'], output_hidden_states=True)

        self.dropout = nn.Dropout(config['drop_rate'])

        self.norm1 = nn.LayerNorm(3072).to(config['device'])
        self.fc1 = nn.Linear(3072, 256).to(config['device'])
        self.norm2 = nn.LayerNorm(256).to(config['device'])
        self.dropout2 = nn.Dropout(config['drop_rate'] * 1.5)

        self.fcs = [nn.Linear(256, 1).to(config['device'])
                    for _ in range(config['num_classes'])]

        # self.fcs = [nn.Linear(3072, 1).to(config['device'])
        #             for _ in range(config['num_classes'])]

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor) -> Tensor:

        berto = self.bert(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask)

        hidden_states = berto.hidden_states

        pooled_output = torch.cat(
            tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)

        pooled_output = pooled_output[:, 0, :]

        pooled_output = self.norm1(pooled_output)

        pooled_output = self.dropout(pooled_output)

        pooled_output = self.fc1(pooled_output)
        pooled_output = self.norm2(pooled_output)

        pooled_output = self.dropout(pooled_output)

        # print(pooled_output.shape)
        outputs = [fc(pooled_output) for fc in self.fcs]

        return outputs


class BertClassifierVer2(nn.Module):
    def __init__(self, config):
        super(BertClassifierVer2, self).__init__()
        self.bert = AutoModel.from_pretrained(
            config['name_model'], output_hidden_states=True)

        self.dropout = nn.Dropout(config['drop_rate'])

        self.norm1 = nn.LayerNorm(3072).to(config['device'])
        self.fc1 = nn.Linear(3072, 256).to(config['device'])
        self.norm2 = nn.LayerNorm(256).to(config['device'])
        self.dropout2 = nn.Dropout(config['drop_rate'] * 1.5)

        self.fcs = nn.Linear(256, 5).to(config['device'])

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor) -> Tensor:

        berto = self.bert(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask)

        hidden_states = berto.hidden_states

        pooled_output = torch.cat(
            tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)

        pooled_output = pooled_output[:, 0, :]

        pooled_output = self.norm1(pooled_output)

        pooled_output = self.dropout(pooled_output)

        pooled_output = self.fc1(pooled_output)
        pooled_output = self.norm2(pooled_output)

        pooled_output = self.dropout(pooled_output)

        # print(pooled_output.shape)
        outputs = self.fcs(pooled_output)

        return outputs
