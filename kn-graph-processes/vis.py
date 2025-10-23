# step4_visualize_and_search.py
# Usage:
#   python step4_visualize_and_search.py /path/to/sap_sam/bpmn \
#       --limit 200 --feat 64 --steps 2 --topk 5 --outdir reports
import os, argparse, xml.etree.ElementTree as ET
from pathlib import Path
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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
        by_dst = {}
        for u, v in edges:
            by_dst.setdefault(v, []).append(u)
        for v in range(n):
            srcs = by_dst.get(v, [])
            cols.extend([u + Vtot for u in srcs])
            rows.append(rows[-1] + len(srcs))
        proc_slices.append((Vtot, Vtot + n))
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
        y = message_passing(indptr, indices, x)   # sum neighbors
        y = y / deg_in[:, None]                   # mean
        x = torch.relu(alpha * x + (1 - alpha) * y)
    return x

def cosine_sim(E):
    En = E / (E.norm(dim=1, keepdim=True) + 1e-12)
    return En @ En.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bpmn_dir")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--feat", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--clusters", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="reports")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required.")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    indptr, indices, proc_slices, names, V = build_merged_csr_with_slices(
        args.bpmn_dir, args.limit, device=device
    )
    Z = run_msg_passing(indptr, indices, V, args.feat, steps=args.steps)  # [V,D]

    # Pool to process embeddings
    Eproc = torch.stack([Z[s:e].mean(dim=0) for s, e in proc_slices], dim=0).cpu()
    torch.save({"names": names, "embeddings": Eproc}, outdir / "process_embeddings.pt")

    # Similarity + top-k neighbors
    S = cosine_sim(Eproc).numpy()
    with open(outdir / "sim_topk.tsv", "w", encoding="utf-8") as f:
        f.write("process\tneighbor\tscore\n")
        for i, name in enumerate(names):
            order = np.argsort(-S[i])
            k = 0
            for j in order:
                if j == i: continue
                f.write(f"{name}\t{names[j]}\t{S[i,j]:.4f}\n")
                k += 1
                if k >= args.topk: break

    # PCA 2D + KMeans overlay
    X = Eproc.numpy()
    pca = PCA(n_components=2).fit(X)
    X2 = pca.transform(X)
    km = KMeans(n_clusters=min(args.clusters, len(names)), n_init=10, random_state=0).fit(X)
    labels = km.labels_

    plt.figure(figsize=(8,6))
    plt.scatter(X2[:,0], X2[:,1], c=labels, s=16)
    plt.title("SAP-SAM Process Embeddings (PCA)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    for i, nm in enumerate(names[:50]):  # avoid clutter
        plt.text(X2[i,0], X2[i,1], str(i), fontsize=6)
    plt.tight_layout()
    plt.savefig(outdir / "pca2d.png", dpi=160)

    print({
        "n_processes": len(names),
        "emb_dim": Eproc.shape[1],
        "pca_var_explained": pca.explained_variance_ratio_[:2].round(3).tolist(),
        "topk_file": str(outdir / "sim_topk.tsv"),
        "plot": str(outdir / "pca2d.png")
    })

if __name__ == "__main__":
    main()
