import torch


def self_attention(Q, K, V):  # batch, len, dim
    S = (
        Q @ K.transpose(-1, -2) / torch.sqrt(torch.tensor(Q.shape[-1]))
    )  # batch, len, len
    score = torch._safe_softmax(S, dim=-1) @ V
    return score
