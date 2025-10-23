# step3_process_embeddings.py
# Usage:
#   python step3_process_embeddings.py /path/to/sap_sam/bpmn --limit 200 --feat 64 --steps 2
import os, argparse, xml.etree.ElementTree as ET
import torch

from build_graph            import BuildGraph
from message_passing_kernel import MsgPassingKernel

class ProcessLevelEmb:

    # Constructor
    def __init__(self):
        pass

    def _cosine_sim(self, E):  # E: [N, D]
        En = torch.nn.functional.normalize(E, dim=1)
        return En @ En.T

    def embed(self,
              bpmn_dir      : str, 
              graph_builder : BuildGraph, 
              mess_pass     : MsgPassingKernel,
              limit         : int = 200,
              feat          : int = 64,
              steps         : int = 2
              ):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("GPU required for this step.")

        indptr, indices, proc_slices, names, V = graph_builder.build_merged_csr_with_slices(
                                                                bpmn_dir, limit, device = device
                                                            )
        Z = mess_pass.run_msg_passing(indptr, indices, V, feat, steps=steps)  # [V,D]

        # pool per process (mean)
        proc_embs = []
        for s, e in proc_slices:
            proc_embs.append(Z[s:e].mean(dim=0))
        Eproc = torch.stack(proc_embs, dim=0).cpu()
        S = self._cosine_sim(Eproc)

        # top-10 similar pairs (exclude self)
        pairs = []
        n = len(names)
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((float(S[i, j]), i, j))
        pairs.sort(reverse=True)
        print("Top similar process pairs:")
        for s, i, j in pairs[:10]:
            print(f"{s:.4f}\t{names[i]}\t{names[j]}")
