# step2_run_msgpass.py
import torch
from msgpass_ext import message_passing  # compiled in step 0
# from your step 1 file:
#   build_csr_from_many(folder) -> indptr, indices, V, E

from step1_build_csr import build_csr_from_many  # adjust import/path

F = 64          # feature size
STEPS = 2       # message-passing iterations
ALPHA = 0.5     # residual blend
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def main(bpmn_dir: str):
    indptr, indices, V, E = build_csr_from_many(bpmn_dir)
    if DEVICE != "cuda":
        raise RuntimeError("This minimal op expects CUDA tensors. Run on a GPU box.")

    # --- features: random for now (replace with real features later)
    x = torch.randn(V, F, device=DEVICE)

    # --- in-degree for mean aggregation
    deg_in = (indptr[1:] - indptr[:-1]).clamp(min=1).to(torch.float32)

    # --- K-step propagation with residual + ReLU
    def propagate(x, steps=STEPS, alpha=ALPHA):
        for _ in range(steps):
            y = message_passing(indptr, indices, x)     # sum of neighbors
            y = y / deg_in[:, None]                     # mean aggregation
            x = torch.relu(alpha * x + (1.0 - alpha) * y)
        return x

    torch.cuda.synchronize()
    z = propagate(x)
    torch.cuda.synchronize()

    print({
        "V": int(V), "E": int(E), "F": F,
        "steps": STEPS, "device": DEVICE,
        "embed_shape": tuple(z.shape)
    })

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python step2_run_msgpass.py /path/to/sap_sam/bpmn")
        raise SystemExit(2)
    main(sys.argv[1])
