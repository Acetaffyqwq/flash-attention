import transformer
import flash_attention
import flash_triton
import torch
import time

batch = 8
dim = 64
seq_lens = [512, 1024, 2048, 4096]

print(
    f"{'seq_len':>6} | {'naive_time':>10} | {'flash_time':>10} | {'naive_mem':>10} | {'flash_mem':>10} | {'max_diff':>10}"
)
print("-" * 80)

for seq_len in seq_lens:
    Q = torch.randn(batch, seq_len, dim, device="cuda")
    K = torch.randn(batch, seq_len, dim, device="cuda")
    V = torch.randn(batch, seq_len, dim, device="cuda")

    Test = flash_triton.flash_attention(Q, K, V)

    # === naive attention ===
    with torch.no_grad():
        for _ in range(3):
            _ = transformer.self_attention(Q, K, V)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        naive_result = transformer.self_attention(Q, K, V)
        torch.cuda.synchronize()
    naive_time = (time.time() - t0) * 1000
    naive_mem = torch.cuda.max_memory_allocated() / 1024**2

    # === flash attention ===
    with torch.no_grad():
        for _ in range(3):
            _ = flash_triton.flash_attention(Q, K, V)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        flash_result = flash_triton.flash_attention(Q, K, V)
        torch.cuda.synchronize()
    flash_time = (time.time() - t0) * 1000
    flash_mem = torch.cuda.max_memory_allocated() / 1024**2

    # === numerical comparison ===
    max_diff = (naive_result - flash_result).abs().max().item()

    print(
        f"{seq_len:6d} | {naive_time:8.2f}ms | {flash_time:8.2f}ms | {naive_mem:8.1f}MB | {flash_mem:8.1f}MB | {max_diff:10.6f}"
    )
