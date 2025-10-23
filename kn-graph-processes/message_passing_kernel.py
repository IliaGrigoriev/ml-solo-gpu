# message_passing_kernel.py
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
SRC_CPP = ROOT / "message_passing.cpp"
SRC_CU  = ROOT / "message_passing.cu"
BUILD_DIR = ROOT / "_build_msgpass"

def _load_ext():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    # env tweaks (optional)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")  # 4090
    os.environ.setdefault("MAX_JOBS", "4")
    ext = load(
        name="msgpass_ext",
        sources=[str(SRC_CPP), str(SRC_CU)],
        verbose=True,
        build_directory=str(BUILD_DIR),
        use_ninja=False,  # avoids VS Code ninja weirdness
    )
    return ext.message_passing

class MsgPassingKernel:
    def __init__(self):
        self._op = None  # lazy init

    def _ensure_loaded(self):
        if self._op is None:
            self._op = _load_ext()

    @torch.no_grad()
    def run_msg_passing(self, indptr, indices, V, D, steps=2, alpha=0.5, mean=True):
        """Return node embeddings [V, D] after K steps."""
        self._ensure_loaded()
        device = indptr.device
        x = torch.randn(V, D, device=device)
        deg_in = (indptr[1:] - indptr[:-1]).clamp(min=1).to(torch.float32)

        for _ in range(steps):
            y = self._op(indptr, indices, x)          # sum of in-neighbors
            if mean:
                y = y / deg_in[:, None]               # mean aggregation
            x = torch.relu(alpha * x + (1 - alpha) * y)
        return x

    # nice-to-have: callable shorthand
    def __call__(self, indptr, indices, V, D, **kw):
        return self.run_msg_passing(indptr, indices, V, D, **kw)
