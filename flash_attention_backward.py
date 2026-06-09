import torch


def flash_forward(Q, K, V, Q_batch, K_batch):  # batch, len, dim
    batch, len, dim = Q.shape
    sqrt_d = torch.sqrt(torch.tensor(dim))
    score = torch.zeros_like(Q)

    L = torch.zeros((batch, len)).to("cuda")
    for i in range(0, len, Q_batch):
        tmp_Q = Q[:, i : i + Q_batch, :]
        pre_max = torch.zeros([batch, Q_batch]).to("cuda")
        pres = torch.zeros([batch, Q_batch, dim]).to("cuda")
        prel = torch.zeros([batch, Q_batch]).to("cuda")

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
        L[:, i : i + Q_batch] = pre_max + torch.log(prel)

    return score, L


def flash_backward(L, Q, K, V, dO, Q_batch, K_batch):
    batch, len, dim = Q.shape
    sqrt_d = torch.sqrt(torch.tensor(dim)).to("cuda")

    dV = torch.zeros_like(V)
    sumS = torch.zeros([batch, len]).to("cuda")
    for i in range(0, len, Q_batch):
        tmp_Q = Q[:, i : i + Q_batch, :]
        for j in range(0, len, K_batch):
            tmp_K = K[:, j : j + K_batch, :]
            P = tmp_Q @ tmp_K.transpose(-1, -2) / sqrt_d  # batch, len_Q, len_K
            S = torch.exp(P - L[:, i : i + Q_batch, None])  # batch, len_Q, len_K
            dV[:, j : j + K_batch, :] += S.transpose(-1, -2) @ dO[:, i : i + Q_batch, :]

            dS = dO[:, i : i + Q_batch, :] @ V.transpose(-1, -2)[:, :, j : j + K_batch]
            sumS[:, i : i + Q_batch] += (dS * S).sum(dim=-1)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    for i in range(0, len, Q_batch):
        tmp_Q = Q[:, i : i + Q_batch, :]
        for j in range(0, len, K_batch):
            tmp_K = K[:, j : j + K_batch, :]
            P = tmp_Q @ tmp_K.transpose(-1, -2) / sqrt_d  # batch, len_Q, len_K
            S = torch.exp(P - L[:, i : i + Q_batch, None])  # batch, len_Q, len_K

            dS = dO[:, i : i + Q_batch, :] @ V.transpose(-1, -2)[:, :, j : j + K_batch]
            dP = S * (dS - sumS[:, i : i + Q_batch, None])

            dQ[:, i : i + Q_batch, :] += dP @ K[:, j : j + K_batch, :]
            dK[:, j : j + K_batch, :] += dP.transpose(-1, -2) @ Q[:, i : i + Q_batch, :]

    dQ /= sqrt_d
    dK /= sqrt_d
    return dQ, dK, dV


class flash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Q_batch, K_batch):
        res, L = flash_forward(Q, K, V, Q_batch, K_batch)
        ctx.save_for_backward(L, Q, K, V)
        ctx.Q_batch = Q_batch
        ctx.K_batch = K_batch
        return res

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V = ctx.saved_tensors
        grad_Q, grad_K, grad_V = flash_backward(
            L, Q, K, V, dO, ctx.Q_batch, ctx.K_batch
        )
        return grad_Q, grad_K, grad_V, None, None


def self_attention(Q, K, V):
    return flash.apply(Q, K, V, 64, 64)


# seq_len |  naive_fwd |  flash_fwd |  naive_bwd |  flash_bwd |  naive_mem |  flash_mem |   out_diff
# --------------------------------------------------------------------------------------------------------------
#    512 |     0.43ms |    20.43ms |     0.71ms |    31.40ms |    107.0MB |     85.8MB |   0.000000
#   1024 |     0.51ms |    76.48ms |     0.26ms |   125.10ms |    227.0MB |    113.5MB |   0.000000
#   2048 |     0.98ms |   299.01ms |     1.21ms |   474.03ms |    646.0MB |    162.6MB |   0.000000
#   4096 |     3.68ms |   950.96ms |     4.84ms |  1410.88ms |   2252.0MB |    260.6MB |   0.000000
