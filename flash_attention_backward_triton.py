import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def kernel_forward(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    batch,
    len,
    dim: tl.constexpr,
    Q_strides,
    K_strides,
    V_strides,
    O_strides,
    L_strides,
    BLOCK_QL: tl.constexpr,
    BLOCK_KL: tl.constexpr,
):
    b_id = tl.program_id(0)
    l_st = tl.program_id(1) * BLOCK_QL
    ql_off = l_st + tl.arange(0, BLOCK_QL)
    kl_off = tl.arange(0, BLOCK_KL)
    d_off = tl.arange(0, dim)

    Q_ptrs = Q_ptr + (ql_off[:, None] * Q_strides[1] + d_off[None, :] * Q_strides[2])
    Q_ptrs += b_id * Q_strides[0]  # len_q, dim
    K_ptrs = K_ptr + (kl_off[None, :] * K_strides[1] + d_off[:, None] * K_strides[2])
    K_ptrs += b_id * K_strides[0]  # dim, len_k
    V_ptrs = V_ptr + (kl_off[:, None] * V_strides[1] + d_off[None, :] * V_strides[2])
    V_ptrs += b_id * V_strides[0]  # len_k, dim

    Q = tl.load(Q_ptrs, mask=ql_off[:, None] < len)
    pre_max = tl.zeros((BLOCK_QL,), dtype=tl.float32) - float("inf")
    pres = tl.zeros((BLOCK_QL, dim), dtype=tl.float32)
    prel = tl.zeros((BLOCK_QL,), dtype=tl.float32)

    for i in range(0, len, BLOCK_KL):  # [i,i+len_k]
        k_mask = (i + kl_off)[None, :] < len
        v_mask = (i + kl_off)[:, None] < len
        K = tl.load(K_ptrs, mask=k_mask)
        V = tl.load(V_ptrs, mask=v_mask)

        qk = tl.dot(Q, K)  # len_q, len_k

        now_max = tl.max(qk, axis=1)  # len_q
        new_max = tl.maximum(pre_max, now_max)
        pres = pres * tl.exp(pre_max - new_max)[:, None]
        prel = prel * tl.exp(pre_max - new_max)
        pre_max = new_max

        qk = tl.exp(qk - pre_max[:, None])
        pres += tl.dot(qk, V)
        prel += tl.sum(qk, axis=1)

        K_ptrs += BLOCK_KL * K_strides[1]
        V_ptrs += BLOCK_KL * V_strides[1]

    O_ptrs = O_ptr + b_id * O_strides[0]
    O_ptrs += ql_off[:, None] * O_strides[1] + d_off[None, :] * O_strides[2]  # len, dim
    tl.store(O_ptrs, pres / prel[:, None], mask=ql_off[:, None] < len)

    L_ptrs = L_ptr + b_id * L_strides[0] + ql_off * L_strides[1]
    tl.store(L_ptrs, pre_max + tl.log(prel), mask=ql_off < len)


def flash_forward(Q, K, V, Q_batch, K_batch):
    batch, len, dim = Q.shape
    Q = Q / torch.sqrt(torch.tensor(dim))
    grid = lambda meta: (
        batch,
        triton.cdiv(len, meta["BLOCK_QL"]),
    )

    score = torch.zeros_like(Q)
    L = torch.zeros([batch, len]).to("cuda")
    kernel_forward[grid](
        Q,
        K,
        V,
        score,
        L,
        batch,
        len,
        dim,
        Q.stride(),
        K.stride(),
        V.stride(),
        score.stride(),
        L.stride(),
        Q_batch,
        K_batch,
    )
    return score, L


@triton.jit
def kernel_backward(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,
    dO_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    batch,
    len,
    dim: tl.constexpr,
    Q_strides,
    K_strides,
    V_strides,
    L_strides,
    dO_strides,
    dQ_strides,
    dK_strides,
    dV_strides,
    BLOCK_QL: tl.constexpr,
    BLOCK_KL: tl.constexpr,
):
    b_id = tl.program_id(0)
    l_st = tl.program_id(1) * BLOCK_QL
    ql_off = l_st + tl.arange(0, BLOCK_QL)
    kl_off = tl.arange(0, BLOCK_KL)
    d_off = tl.arange(0, dim)

    Q_ptrs = Q_ptr + (ql_off[:, None] * Q_strides[1] + d_off[None, :] * Q_strides[2])
    Q_ptrs += b_id * Q_strides[0]  # len_q, dim
    K_ptrs = K_ptr + (kl_off[None, :] * K_strides[1] + d_off[:, None] * K_strides[2])
    K_ptrs += b_id * K_strides[0]  # dim, len_k K^T
    V_ptrs = V_ptr + (kl_off[None, :] * V_strides[1] + d_off[:, None] * V_strides[2])
    V_ptrs += b_id * V_strides[0]  # dim, len_k V^T
    L_ptrs = L_ptr + b_id * L_strides[0] + ql_off * L_strides[1]

    dO_ptrs = dO_ptr + ql_off[:, None] * dO_strides[1] + d_off[None, :] * dO_strides[2]
    dO_ptrs += b_id * dO_strides[0]  # len_q, dim
    dV_ptrs = dV_ptr + kl_off[:, None] * dV_strides[1] + d_off[None, :] * dV_strides[2]
    dV_ptrs += b_id * dV_strides[0]  # len_k, dim

    L = tl.load(L_ptrs, mask=ql_off < len)
    Q = tl.load(Q_ptrs, mask=ql_off[:, None] < len)
    dO = tl.load(dO_ptrs, mask=ql_off[:, None] < len)
    sumS = tl.zeros((BLOCK_QL,), dtype=tl.float32)
    for i in range(0, len, BLOCK_KL):
        K = tl.load(K_ptrs, mask=i + kl_off[None, :] < len)
        V = tl.load(V_ptrs, mask=i + kl_off[None, :] < len)

        P = tl.dot(Q, K)  # len_q, len_k
        S = tl.exp(P - L[:, None])
        upd_dV = tl.dot(S.trans(1, 0), dO)  # len_k,dim
        tl.atomic_add(dV_ptrs, upd_dV, mask=i + kl_off[:, None] < len)

        dS = tl.dot(dO, V)  # len_q,len_k
        sumS += (dS * S).sum(axis=-1)

        K_ptrs += BLOCK_KL * K_strides[1]
        V_ptrs += BLOCK_KL * V_strides[1]
        dV_ptrs += BLOCK_KL * dV_strides[1]

    K_ptrs = K_ptr + (kl_off[None, :] * K_strides[1] + d_off[:, None] * K_strides[2])
    K_ptrs += b_id * K_strides[0]
    V_ptrs = V_ptr + (kl_off[None, :] * V_strides[1] + d_off[:, None] * V_strides[2])
    V_ptrs += b_id * V_strides[0]  # reload K^T and V^T

    dQ_ptrs = dQ_ptr + ql_off[:, None] * dQ_strides[1] + d_off[None, :] * dQ_strides[2]
    dQ_ptrs += b_id * dQ_strides[0]  # len_q, dim
    dK_ptrs = dK_ptr + kl_off[:, None] * dK_strides[1] + d_off[None, :] * dK_strides[2]
    dK_ptrs += b_id * dK_strides[0]  # len_k, dim
    for i in range(0, len, BLOCK_KL):
        K = tl.load(K_ptrs, mask=i + kl_off[None, :] < len)
        V = tl.load(V_ptrs, mask=i + kl_off[None, :] < len)
        P = tl.dot(Q, K)  # len_q, len_k
        S = tl.exp(P - L[:, None])

        dS = tl.dot(dO, V)  # len_q,len_k
        dP = S * (dS - sumS[:, None])

        upd_dQ = tl.dot(dP, K.trans(1, 0))  # len_q, dim
        upd_dK = tl.dot(dP.trans(1, 0), Q)  # len_k, dim
        tl.atomic_add(dQ_ptrs, upd_dQ, mask=ql_off[:, None] < len)
        tl.atomic_add(dK_ptrs, upd_dK, mask=i + kl_off[:, None] < len)

        K_ptrs += BLOCK_KL * K_strides[1]
        V_ptrs += BLOCK_KL * V_strides[1]
        dK_ptrs += BLOCK_KL * dK_strides[1]


def flash_backward(Q, K, V, L, dO, Q_batch, K_batch):
    batch, len, dim = Q.shape
    sqrt_d = torch.sqrt(torch.tensor(dim))
    Q = Q / sqrt_d
    grid = lambda meta: (
        batch,
        triton.cdiv(len, meta["BLOCK_QL"]),
    )

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    kernel_backward[grid](
        Q,
        K,
        V,
        L,
        dO,
        dQ,
        dK,
        dV,
        batch,
        len,
        dim,
        Q.stride(),
        K.stride(),
        V.stride(),
        L.stride(),
        dO.stride(),
        dQ.stride(),
        dK.stride(),
        dV.stride(),
        Q_batch,
        K_batch,
    )
    dQ = dQ / sqrt_d
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
            Q, K, V, L, dO, ctx.Q_batch, ctx.K_batch
        )
        return grad_Q, grad_K, grad_V, None, None


# seq_len |  naive_fwd |  flash_fwd |  naive_bwd |  flash_bwd |  naive_mem |  flash_mem |   out_diff
# --------------------------------------------------------------------------------------------------------------
#    512 |     0.16ms |     0.16ms |     1.12ms |     0.40ms |    107.0MB |     87.0MB |   0.002397
#   2048 |     0.96ms |     0.28ms |     1.22ms |     1.83ms |    633.0MB |    162.1MB |   0.000734
#   8192 |    14.33ms |     2.08ms |    18.22ms |    21.90ms |   8484.0MB |    456.3MB |   0.000385
#   4096 |     3.67ms |     0.67ms |     4.99ms |     6.04ms |   2408.0MB |    424.1MB |   0.000622
