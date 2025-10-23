# build_graph.py  (CSV with Signavio JSON -> CSR)
import os
from pathlib import Path
import json
import pandas as pd
import torch

# node stencil ids we keep (case-sensitive as seen in Signavio JSON)
NODE_STENCILS = {
    "Task", "UserTask", "ServiceTask", "ScriptTask",
    "StartEvent", "EndEvent", "IntermediateThrowEvent", "IntermediateCatchEvent",
    "ExclusiveGateway", "ParallelGateway", "InclusiveGateway",
}
EDGE_STENCIL = "SequenceFlow"  # Signavio edge element

class BuildGraph:
    def __init__(self):
        pass

    # --- CSV helpers ---------------------------------------------------------
    def _detect_json_col(self, df: pd.DataFrame) -> str:
        """Pick the column that actually contains the Signavio JSON."""
        cand = [c for c in df.columns if "json" in c.lower() or "model" in c.lower()]
        cols = cand or list(df.columns)
        for c in cols:
            v = df[c].dropna().astype(str).head(3).to_list()
            if any(vv.strip().startswith("{") and '"stencilset"' in vv for vv in v):
                return c
        # fallback: first column that looks like JSON
        for c in df.columns:
            v = df[c].dropna().astype(str).head(3).to_list()
            if any(vv.strip().startswith("{") for vv in v):
                return c
        raise RuntimeError("Could not detect JSON column in CSV.")

    # --- Signavio JSON -> (nodes, edges) -------------------------------------
    def _extract_graph_from_signavio(self, data: dict):
        """
        Returns (nodes_list, edges_pairs) for ONE diagram JSON.
        nodes_list: list of resourceIds we keep (stencil in NODE_STENCILS)
        edges_pairs: list of (src_idx, dst_idx)
        """
        nodes = {}          # resourceId -> idx
        edge_shapes = {}    # edge resourceId -> (srcId, dstId)
        pending_out = []    # (srcNodeId, outgoingRefId) where outgoingRefId may be node or edge

        def visit(el: dict):
            if not isinstance(el, dict):
                return
            stencil = (el.get("stencil") or {}).get("id", "")
            rid = el.get("resourceId")

            # collect nodes
            if rid and stencil in NODE_STENCILS and rid not in nodes:
                nodes[rid] = len(nodes)

            # collect explicit sequence flows (source/target)
            if stencil == EDGE_STENCIL:
                src = (el.get("source") or {}).get("resourceId")
                tgt = (el.get("target") or {}).get("resourceId")
                if src and tgt:
                    edge_shapes[rid or f"__edge_{len(edge_shapes)}"] = (src, tgt)

            # collect outgoing refs from nodes
            outs = el.get("outgoing") or []
            if rid and outs:
                for o in outs:
                    oid = o.get("resourceId")
                    if oid:
                        pending_out.append((rid, oid))

            # recurse children
            for ch in el.get("childShapes") or []:
                visit(ch)

        # kick off traversal from top-level childShapes (Signavio root has 'childShapes')
        for el in data.get("childShapes") or []:
            visit(el)

        # resolve edges
        edges = []
        # 1) sequence flows
        for (s, t) in edge_shapes.values():
            if s in nodes and t in nodes:
                edges.append((nodes[s], nodes[t]))
        # 2) outgoing pointers (either direct to node or via edge id)
        for s, ref in pending_out:
            if s not in nodes:
                continue
            if ref in nodes:  # direct node reference
                edges.append((nodes[s], nodes[ref]))
            elif ref in edge_shapes:
                _, t = edge_shapes[ref]
                if t in nodes:
                    edges.append((nodes[s], nodes[t]))

        # finalize ordered node list (by index)
        node_ids = [None] * len(nodes)
        for rid, i in nodes.items():
            node_ids[i] = rid
        return node_ids, edges

    # --- Public: build merged CSR from CSV directory -------------------------
    def build_merged_csr_with_slices(self, csv_dir, limit=None, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        files = list(Path(csv_dir).glob("*.csv"))
        if limit:
            files = files[:limit]

        Vtot, rows, cols = 0, [0], []
        proc_slices, names = [], []
        processed = 0

        for fp in files:
            # stream rows to avoid big memory spikes
            try:
                for chunk in pd.read_csv(fp, chunksize=512):
                    json_col = self._detect_json_col(chunk)
                    for _, row in chunk.iterrows():
                        js = row.get(json_col)
                        if not isinstance(js, str) or not js.strip().startswith("{"):
                            continue
                        try:
                            data = json.loads(js)
                        except json.JSONDecodeError:
                            continue

                        node_ids, edge_pairs = self._extract_graph_from_signavio(data)
                        n = len(node_ids)
                        if n < 2 or not edge_pairs:
                            continue

                        # dest-centric CSR for this process
                        by_dst = {}
                        for u, v in edge_pairs:
                            by_dst.setdefault(v, []).append(u)
                        for v in range(n):
                            srcs = by_dst.get(v, [])
                            cols.extend([u + Vtot for u in srcs])
                            rows.append(rows[-1] + len(srcs))

                        proc_slices.append((Vtot, Vtot + n))
                        # try to name by a stable column if present; else file row index
                        names.append(os.path.basename(fp))
                        Vtot += n
                        processed += 1
                        if limit and processed >= limit:
                            raise StopIteration
            except StopIteration:
                break
            except Exception:
                # skip corrupt CSVs quietly
                continue

        if Vtot == 0:
            raise RuntimeError("No valid graphs parsed from CSV+JSON. Check column detection or JSON format.")

        indptr = torch.tensor(rows, dtype=torch.int64, device=device)
        indices = torch.tensor(cols, dtype=torch.int64, device=device)
        return indptr, indices, proc_slices, names, Vtot
