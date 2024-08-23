import torch.nn as nn
import numpy as np
import torch
from torch import Tensor


class WKPooling(nn.Module):
    def __init__(self, layer_start: int = 4, context_window_size: int = 2):
        super(WKPooling, self).__init__()
        self.layer_start = layer_start
        self.context_window_size = context_window_size

    def forward(self, all_hidden_states, features):
        ft_all_layers = all_hidden_states
        org_device = ft_all_layers.device
        all_layer_embedding = ft_all_layers.transpose(1, 0)
        # Start from 4th layers output
        all_layer_embedding = all_layer_embedding[:, self.layer_start:, :, :]

        # torch.qr is slow on GPU (see https://github.com/pytorch/pytorch/issues/22573). So compute it on CPU until issue is fixed
        # all_layer_embedding = all_layer_embedding.cpu()

        attention_mask = features.cpu().numpy()
        # Not considering the last item
        unmask_num = np.array([sum(mask) for mask in attention_mask]) - 1
        embedding = []

        # One sentence at a time
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index,
                                                   :, :unmask_num[sent_index], :]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:, token_index, :]
                # 'Unified Word Representation'
                token_embedding = self.unify_token(token_feature)
                one_sentence_embedding.append(token_embedding)

            # features.update({'sentence_embedding': features['cls_token_embeddings']})

            one_sentence_embedding = torch.stack(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(
                sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        output_vector = torch.stack(embedding).to(org_device)
        return output_vector

    def unify_token(self, token_feature):
        # Unify Token Representation
        window_size = self.context_window_size

        alpha_alignment = torch.zeros(
            token_feature.size()[0], device=token_feature.device)
        alpha_novelty = torch.zeros(
            token_feature.size()[0], device=token_feature.device)

        for k in range(token_feature.size()[0]):
            left_window = token_feature[k - window_size:k, :]
            right_window = token_feature[k + 1:k + window_size + 1, :]
            window_matrix = torch.cat(
                [left_window, right_window, token_feature[k, :][None, :]])
            Q, R = torch.linalg.qr(window_matrix.T, mode='reduced')
            r = R[:, -1]
            alpha_alignment[k] = torch.mean(self.norm_vector(
                R[:-1, :-1], dim=0), dim=1).matmul(R[:-1, -1]) / torch.norm(r[:-1])
            alpha_alignment[k] = 1 / \
                (alpha_alignment[k] * window_matrix.size()[0] * 2)
            alpha_novelty[k] = torch.abs(r[-1]) / torch.norm(r)

        # Sum Norm
        alpha_alignment = alpha_alignment / \
            torch.sum(alpha_alignment)  # Normalization Choice
        alpha_novelty = alpha_novelty / torch.sum(alpha_novelty)

        alpha = alpha_novelty + alpha_alignment
        alpha = alpha / torch.sum(alpha)  # Normalize

        out_embedding = torch.mv(token_feature.t(), alpha)
        return out_embedding

    def norm_vector(self, vec, p=2, dim=0):
        # Implements the normalize() function from sklearn
        vec_norm = torch.norm(vec, p=p, dim=dim)
        return vec.div(vec_norm.expand_as(vec))

    def unify_sentence(self, sentence_feature, one_sentence_embedding):
        # Unify Sentence By Token Importance
        sent_len = one_sentence_embedding.size()[0]

        var_token = torch.zeros(sent_len, device=one_sentence_embedding.device)
        for token_index in range(sent_len):
            token_feature = sentence_feature[:, token_index, :]
            sim_map = self.cosine_similarity_torch(token_feature)
            var_token[token_index] = torch.var(sim_map.diagonal(-1))

        var_token = var_token / torch.sum(var_token)
        sentence_embedding = torch.mv(one_sentence_embedding.t(), var_token)

        return sentence_embedding

    def cosine_similarity_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
