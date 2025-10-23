# step2_run_msgpass.py
import torch
from msgpass_ext import message_passing  # compiled in step 0
# from your step 1 file:
#   build_csr_from_many(folder) -> indptr, indices, V, E

#from step1_build_csr import build_csr_from_many  # adjust import/path

class MsgPassingKernel:

    # Constructor    
    def __init__(self):
        pass

    # Run MSG passing
    @torch.no_grad()
    def run_msg_passing(indptr, indices, V, D, steps=2, alpha=0.5):
        x = torch.randn(V, D, device=indptr.device)
        deg_in = (indptr[1:] - indptr[:-1]).clamp(min=1).to(torch.float32)
        for _ in range(steps):
            y = message_passing(indptr, indices, x)   # sum over in-neighbors
            y = y / deg_in[:, None]                   # mean
            x = torch.relu(alpha * x + (1 - alpha) * y)
        return x  # [V, D]