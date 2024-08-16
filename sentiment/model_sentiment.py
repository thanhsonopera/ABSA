import torch
import torch.nn as nn
from transformers import AutoModel
from torch import Tensor


class SentimentClassifier(nn.Module):
    def __init__(self, config):
        super(SentimentClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(
            config['name_model'], output_hidden_states=True)

        self.dropout = nn.Dropout(config['drop_rate'])

        self.fcs = [nn.Linear(3072, 4).to(config['device'])
                    for _ in range(config['num_classes'])]

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor) -> Tensor:

        berto = self.bert(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask)

        hidden_states = berto.hidden_states

        pooled_output = torch.cat(
            tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)

        pooled_output = pooled_output[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        outputs = [nn.Softmax(dim=1)(fc(pooled_output)) for fc in self.fcs]
        return outputs
