import os, xml.etree.ElementTree as ET
import torch

BPMN_NS = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
KEEP = {
    "task","userTask","serviceTask","scriptTask",
    "startEvent","endEvent","intermediateThrowEvent","intermediateCatchEvent",
    "exclusiveGateway","parallelGateway","inclusiveGateway",
}

class BuildGraph:

    # Constructor
    def __init__(self):
        pass

    # Parse 
    def parse_bpmn(self, path):
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

    # ...
    def build_merged_csr_with_slices(self, bpmn_dir, limit=None, device="cuda"):
        files = [f for f in os.listdir(bpmn_dir) if f.lower().endswith(".bpmn")]
        if limit: files = files[:limit]
        Vtot, rows, cols = 0, [0], []
        proc_slices, names = [], []
        for fname in files:
            nodes, edges = self.parse_bpmn(os.path.join(bpmn_dir, fname))
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
