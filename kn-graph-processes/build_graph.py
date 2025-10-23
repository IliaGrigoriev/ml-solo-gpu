import os, xml.etree.ElementTree as ET
import torch

BPMN_NS = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"}

def read_bpmn_file(path):
    t = ET.parse(path); r = t.getroot()
    # nodes: tasks, gateways, events
    q = ".//bpmn:process//bpmn:*"
    nodes = []
    for e in r.findall(q, BPMN_NS):
        tag = e.tag.split("}")[1]
        if tag in {"task","userTask","serviceTask","scriptTask",
                   "startEvent","endEvent","intermediateThrowEvent",
                   "exclusiveGateway","parallelGateway","inclusiveGateway"}:
            nid = e.attrib.get("id"); 
            if nid: nodes.append(nid)
    nodes = list(dict.fromkeys(nodes))
    idx = {n:i for i,n in enumerate(nodes)}
    # edges: sequenceFlow (src -> dst)
    edges = []
    for sf in r.findall(".//bpmn:sequenceFlow", BPMN_NS):
        s, t_ = sf.attrib.get("sourceRef"), sf.attrib.get("targetRef")
        if s in idx and t_ in idx: edges.append((idx[s], idx[t_]))
    return idx, edges  # per diagram

def build_csr_from_many(folder):
    # (Option) merge diagrams into one graph by offsetting node ids
    V, E = 0, 0
    rows = [0]; cols = []
    for fp in [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".bpmn")][:200]:
        idx, edge_list = read_bpmn_file(fp)
        if not idx: continue
        # We want dest-centric CSR: group by dst
        by_dst = {}
        for u,v in edge_list:
            by_dst.setdefault(v, []).append(u)
        n = len(idx)
        for v in range(n):
            srcs = [u for u in by_dst.get(v, [])]
            cols.extend([u + V for u in srcs])
            rows.append(rows[-1] + len(srcs))
        V += n; E += len(edge_list)
    indptr = torch.tensor(rows, dtype=torch.int64, device="cuda")
    indices = torch.tensor(cols, dtype=torch.int64, device="cuda")
    return indptr, indices, V, E
