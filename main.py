import transformer
import flash_attention_backward
import torch

torch.set_default_device("cuda")

batch = 2
dim = 64
seq_lens = [128, 256, 512]
Q_batch = 64
K_batch = 64

print(
    f"{'seq_len':>6} | {'out_diff':>12} | {'dQ_diff':>12} | {'dK_diff':>12} | {'dV_diff':>12}"
)
print("-" * 68)

for seq_len in seq_lens:
    Q = torch.randn(batch, seq_len, dim, device="cuda")
    K = torch.randn(batch, seq_len, dim, device="cuda")
    V = torch.randn(batch, seq_len, dim, device="cuda")

    # === reference: transformer with autograd ===
    Q_ref = Q.clone().detach().requires_grad_(True)
    K_ref = K.clone().detach().requires_grad_(True)
    V_ref = V.clone().detach().requires_grad_(True)

    out_ref = transformer.self_attention(Q_ref, K_ref, V_ref)
    out_ref.sum().backward()

    # === flash attention with custom backward ===
    Q_flash = Q.clone().detach().requires_grad_(True)
    K_flash = K.clone().detach().requires_grad_(True)
    V_flash = V.clone().detach().requires_grad_(True)

    out_flash = flash_attention_backward.flash.apply(
        Q_flash, K_flash, V_flash, Q_batch, K_batch
    )
    out_flash.sum().backward()

    # === numerical comparison ===
    out_diff = (out_ref - out_flash).abs().max().item()
    dQ_diff = (Q_ref.grad - Q_flash.grad).abs().max().item()
    dK_diff = (K_ref.grad - K_flash.grad).abs().max().item()
    dV_diff = (V_ref.grad - V_flash.grad).abs().max().item()

    print(
        f"{seq_len:6d} | {out_diff:12.6f} | {dQ_diff:12.6f} | {dK_diff:12.6f} | {dV_diff:12.6f}"
    )
