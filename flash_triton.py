import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    batch,
    len,
    dim: tl.constexpr,
    stride_qb,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vl,
    stride_vd,
    stride_ob,
    stride_ol,
    stride_od,
    BLOCK_QL: tl.constexpr,
    BLOCK_KL: tl.constexpr,
):
    b_id = tl.program_id(0)
    l_st = tl.program_id(1) * BLOCK_QL
    ql_off = l_st + tl.arange(0, BLOCK_QL)
    kl_off = tl.arange(0, BLOCK_KL)
    d_off = tl.arange(0, dim)

    Q_ptrs = Q_ptr + (ql_off[:, None] * stride_ql + d_off[None, :] * stride_qd)
    Q_ptrs += b_id * stride_qb  # len_q, dim
    K_ptrs = K_ptr + (kl_off[None, :] * stride_kl + d_off[:, None] * stride_kd)
    K_ptrs += b_id * stride_kb  # dim, len_k
    V_ptrs = V_ptr + (kl_off[:, None] * stride_vl + d_off[None, :] * stride_vd)
    V_ptrs += b_id * stride_vb  # len_k, dim

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

        K_ptrs += BLOCK_KL * stride_kl
        V_ptrs += BLOCK_KL * stride_vl

    O_ptrs = O_ptr + b_id * stride_ob
    O_ptrs += ql_off[:, None] * stride_ol + d_off[None, :] * stride_od  # len, dim
    tl.store(O_ptrs, pres / prel[:, None], mask=ql_off[:, None] < len)


def flash_attention(Q, K, V):
    batch, len, dim = Q.shape
    Q = Q / torch.sqrt(torch.tensor(dim))
    grid = lambda meta: (
        batch,
        triton.cdiv(len, meta["BLOCK_QL"]),
    )
    O = torch.zeros_like(Q)
    kernel[grid](
        Q,
        K,
        V,
        O,
        batch,
        len,
        dim,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        64,
        64,
    )
    return O


# seq_len | naive_time | flash_time |  naive_mem |  flash_mem |   max_diff
# --------------------------------------------------------------------------------
#    512 |     0.16ms |     0.09ms |     55.0MB |     40.0MB |   0.001515
#   1024 |     0.17ms |     0.10ms |    116.0MB |     49.0MB |   0.001848
#   2048 |     0.94ms |     0.21ms |    344.0MB |     66.0MB |   0.000969
#   4096 |     3.68ms |     0.62ms |   1232.0MB |    100.0MB |   0.000777
