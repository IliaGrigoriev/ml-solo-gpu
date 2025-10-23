# step3_process_embeddings.py
# Usage:
#   python step3_process_embeddings.py /path/to/sap_sam/bpmn --limit 200 --feat 64 --steps 2
import os, argparse, xml.etree.ElementTree as ET
import torch
from msgpass_ext import message_passing

BPMN_NS = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
KEEP = {
    "task","userTask","serviceTask","scriptTask",
    "startEvent","endEvent","intermediateThrowEvent","intermediateCatchEvent",
    "exclusiveGateway","parallelGateway","inclusiveGateway",
}

def parse_bpmn(path):
    try:
        t = ET.parse(path)
    except ET.ParseError:
        return [], []
    r = t.getroot()
    nodes, id2idx = [], {}
    for e in r.findall(".//bpmn:process//bpmn:*", BPMN_NS):
        tag = e.tag.split("}")[-1]
        if tag in KEEP:
            nid = e.attrib.get("id")
            if nid and nid not in id2idx:
                id2idx[nid] = len(nodes)
                nodes.append(nid)
    if not nodes:
        return [], []
    edges = []
    for sf in r.findall(".//bpmn:sequenceFlow", BPMN_NS):
        s, t_ = sf.attrib.get("sourceRef"), sf.attrib.get("targetRef")
        if s in id2idx and t_ in id2idx:
            edges.append((id2idx[s], id2idx[t_]))
    return nodes, edges

def build_merged_csr_with_slices(bpmn_dir, limit=None, device="cuda"):
    files = [f for f in os.listdir(bpmn_dir) if f.lower().endswith(".bpmn")]
    if limit: files = files[:limit]
    Vtot, rows, cols = 0, [0], []
    proc_slices, names = [], []
    for fname in files:
        nodes, edges = parse_bpmn(os.path.join(bpmn_dir, fname))
        n = len(nodes)
        if n < 2 or not edges:
            continue
        # dest-centric CSR for this process, then offset
        by_dst = {}
        for u, v in edges:
            by_dst.setdefault(v, []).append(u)
        for v in range(n):
            srcs = by_dst.get(v, [])
            cols.extend([u + Vtot for u in srcs])
            rows.append(rows[-1] + len(srcs))
        proc_slices.append((Vtot, Vtot + n))   # global [start, end) for this process
        names.append(fname)
        Vtot += n
    if Vtot == 0:
        raise RuntimeError("No valid BPMN graphs parsed.")
    indptr = torch.tensor(rows, dtype=torch.int64, device=device)
    indices = torch.tensor(cols, dtype=torch.int64, device=device)
    return indptr, indices, proc_slices, names, Vtot

@torch.no_grad()
def run_msg_passing(indptr, indices, V, D, steps=2, alpha=0.5):
    x = torch.randn(V, D, device=indptr.device)
    deg_in = (indptr[1:] - indptr[:-1]).clamp(min=1).to(torch.float32)
    for _ in range(steps):
        y = message_passing(indptr, indices, x)   # sum over in-neighbors
        y = y / deg_in[:, None]                   # mean
        x = torch.relu(alpha * x + (1 - alpha) * y)
    return x  # [V, D]

def cosine_sim(E):  # E: [N, D]
    En = torch.nn.functional.normalize(E, dim=1)
    return En @ En.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bpmn_dir")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--feat", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required for this step.")

    indptr, indices, proc_slices, names, V = build_merged_csr_with_slices(
        args.bpmn_dir, args.limit, device=device
    )
    Z = run_msg_passing(indptr, indices, V, args.feat, steps=args.steps)  # [V,D]

    # pool per process (mean)
    proc_embs = []
    for s, e in proc_slices:
        proc_embs.append(Z[s:e].mean(dim=0))
    Eproc = torch.stack(proc_embs, dim=0).cpu()
    S = cosine_sim(Eproc)

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

if __name__ == "__main__":
    main()
