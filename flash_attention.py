import torch


def self_attention(Q, K, V):  # batch, len, dim
    sqrt_d = torch.sqrt(torch.tensor(Q.shape[-1]))
    Q_batch = 64
    K_batch = 64
    batch, len, dim = Q.shape

    score = torch.zeros_like(Q)

    for i in range(0, len, Q_batch):
        tmp_Q = Q[:, i : i + Q_batch, :]
        pre_max = torch.zeros([batch, Q_batch])
        pres = torch.zeros([batch, Q_batch, dim])
        prel = torch.zeros([batch, Q_batch])

        for j in range(0, len, K_batch):
            tmp_K = K[:, j : j + K_batch, :]
            tmp_V = V[:, j : j + K_batch, :]
            S = tmp_Q @ tmp_K.transpose(-1, -2) / sqrt_d  # batch, Q_batch , K_batch
            if j == 0:
                pre_max = torch.max(S, dim=-1).values
                S -= pre_max[:, :, None]
                pres = torch.exp(S) @ tmp_V
                prel = torch.sum(torch.exp(S), dim=-1)
            else:
                now_max = torch.max(S, dim=-1).values
                delta = torch.max(now_max - pre_max, torch.zeros_like(pre_max))
                pres = pres * torch.exp(-delta.unsqueeze(-1))
                prel = prel * torch.exp(-delta)
                pre_max += delta

                S -= pre_max[:, :, None]
                pres += torch.exp(S) @ tmp_V
                prel += torch.sum(torch.exp(S), dim=-1)

        score[:, i : i + Q_batch, :] = pres / prel.unsqueeze(-1)
    return score


# seq_len | naive_time | flash_time |  naive_mem |  flash_mem |   max_diff
# --------------------------------------------------------------------------------
#    512 |     0.53ms |    11.73ms |     54.0MB |     38.5MB |   0.000000
#   1024 |     0.17ms |    45.34ms |    114.0MB |     45.5MB |   0.000001
#   2048 |     0.93ms |   245.17ms |    340.0MB |     58.5MB |   0.000000
#   4096 |     3.66ms |   738.33ms |   1224.0MB |     84.5MB |   0.000001
