import transformer
import flash_attention_backward
import torch
import time

torch.set_default_device("cuda")

batch = 8
dim = 64
seq_lens = [512, 1024, 2048, 4096]
Q_batch = 64
K_batch = 64

header = (
    f"{'seq_len':>6} | "
    f"{'naive_fwd':>10} | {'flash_fwd':>10} | "
    f"{'naive_bwd':>10} | {'flash_bwd':>10} | "
    f"{'naive_mem':>10} | {'flash_mem':>10} | "
    f"{'out_diff':>10}"
)
print(header)
print("-" * 110)

for seq_len in seq_lens:
    Q = torch.randn(batch, seq_len, dim)
    K = torch.randn(batch, seq_len, dim)
    V = torch.randn(batch, seq_len, dim)

    # === naive transformer: forward + backward ===
    Q_naive = Q.clone().detach().requires_grad_(True)
    K_naive = K.clone().detach().requires_grad_(True)
    V_naive = V.clone().detach().requires_grad_(True)

    for _ in range(3):
        out = transformer.self_attention(Q_naive, K_naive, V_naive)
        out.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    out_naive = transformer.self_attention(Q_naive, K_naive, V_naive)
    out_naive.sum().backward()
    torch.cuda.synchronize()
    naive_total = (time.time() - t0) * 1000
    naive_mem = torch.cuda.max_memory_allocated() / 1024**2

    # === flash attention: forward + backward ===
    Q_flash = Q.clone().detach().requires_grad_(True)
    K_flash = K.clone().detach().requires_grad_(True)
    V_flash = V.clone().detach().requires_grad_(True)

    for _ in range(3):
        out = flash_attention_backward.flash.apply(
            Q_flash, K_flash, V_flash, Q_batch, K_batch
        )
        out.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    out_flash = flash_attention_backward.flash.apply(
        Q_flash, K_flash, V_flash, Q_batch, K_batch
    )
    out_flash.sum().backward()
    torch.cuda.synchronize()
    flash_total = (time.time() - t0) * 1000
    flash_mem = torch.cuda.max_memory_allocated() / 1024**2

    # === separate forward-only timing ===
    Q1 = Q.clone().detach()
    K1 = K.clone().detach()
    V1 = V.clone().detach()
    for _ in range(3):
        _ = transformer.self_attention(Q1, K1, V1)
    torch.cuda.synchronize()
    t0 = time.time()
    _ = transformer.self_attention(Q1, K1, V1)
    torch.cuda.synchronize()
    naive_fwd = (time.time() - t0) * 1000

    Q2 = Q.clone().detach()
    K2 = K.clone().detach()
    V2 = V.clone().detach()
    for _ in range(3):
        _ = flash_attention_backward.flash.apply(Q2, K2, V2, Q_batch, K_batch)
    torch.cuda.synchronize()
    t0 = time.time()
    _ = flash_attention_backward.flash.apply(Q2, K2, V2, Q_batch, K_batch)
    torch.cuda.synchronize()
    flash_fwd = (time.time() - t0) * 1000

    # backward time = total - forward
    naive_bwd = naive_total - naive_fwd
    flash_bwd = flash_total - flash_fwd

    out_diff = (out_naive - out_flash).abs().max().item()

    print(
        f"{seq_len:6d} | "
        f"{naive_fwd:8.2f}ms | {flash_fwd:8.2f}ms | "
        f"{naive_bwd:8.2f}ms | {flash_bwd:8.2f}ms | "
        f"{naive_mem:8.1f}MB | {flash_mem:8.1f}MB | "
        f"{out_diff:10.6f}"
    )
