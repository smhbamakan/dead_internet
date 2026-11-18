"""Graph-level CPAG inference with evidence lineage (dependence-aware fusion), SCC handling and conflict routing.

evaluate(nodes, edges, decision) -> dict of node opinions, flags

nodes: {id: dict(layer=..., prop=..., prior=opinion or None, origin=set or None, cap=float)}
edges: list of (src, dst, type) with type in ATT, TRN, CC, DEL, CON
Semantics
  * every opinion refers to the proposition attached to the destination node (prop field);
    an edge u->v is admissible only if it carries evidence about prop(v) (ATT/TRN/CC/CON) or
    authority over v (DEL); mixing propositions raises ValueError.
  * DEL edges discount (R1) the opinion of the source about v's claim by the trust in the source.
  * evidence-bearing edges are fused cumulatively (R3) ONLY across independent lineages; opinions
    sharing an origin are combined by averaging fusion (no double counting).
  * per-source evidence is capped at `cap` (Theorem 1).
  * strongly connected components are condensed; evidence inside a cycle is not re-used (no
    self-reinforcement); nodes in a cycle receive the fused opinion of their external inputs.
  * a pair of incoming opinions with DC > theta_c raises a review flag and the node is not fused.
Complexity: O(|V| + |E|) graph work plus O(|E| * L) set operations for lineage size L.
"""
import numpy as np, networkx as nx
import sl

ADMIT = {"ATT": {"device_state", "data_integrity"}, "TRN": {"data_integrity", "process_integrity", "content_truth"},
         "CC": {"content_truth"}, "CON": {"content_truth", "data_integrity"}, "DEL": {"authorised", "content_truth"}}


def cap_op(w, c):
    r, s = sl.to_evidence(w) if w[2] > 0 else (1e9 * w[0], 1e9 * w[1])
    e = r + s
    if e > c:
        r, s = r * c / e, s * c / e
    return sl.from_evidence(r, s, w[3])


def avg_fuse(ws):
    """Averaging fusion for dependent opinions (Josang 2016): mean of evidence."""
    ev = np.array([sl.to_evidence(w) for w in ws]); r, s = ev.mean(0)
    return sl.from_evidence(r, s, ws[0][3])


def evaluate(nodes, edges, theta_c=0.5, cap=20.0):
    G = nx.DiGraph(); G.add_nodes_from(nodes); G.add_edges_from([(u, v, {"t": t}) for u, v, t in edges])
    for u, v, t in edges:
        if nodes[v]["prop"] not in ADMIT[t]:
            raise ValueError(f"edge {u}->{v} type {t} carries no evidence about {nodes[v]['prop']}")
    C = nx.condensation(G); order = list(nx.topological_sort(C))
    op, lineage, flags = {}, {}, {}
    for comp in order:
        members = C.nodes[comp]["members"]
        for v in members:
            a = nodes[v].get("a", 0.5)
            inc = []
            for u in G.predecessors(v):
                if u in members:          # inside a cycle: skip to avoid self-reinforcement
                    continue
                t = G.edges[u, v]["t"]
                if t == "DEL":
                    w = sl.discount(op[u], nodes[v].get("claim", sl.op(1, 0, 0, a)))
                else:
                    w = op[u].copy()
                w[3] = a
                inc.append((cap_op(w, nodes[u].get("cap", cap)), lineage[u]))
            prior = nodes[v].get("prior")
            if prior is not None:
                inc.append((cap_op(prior, nodes[v].get("cap", cap)), nodes[v].get("origin", {v})))
            if not inc:
                op[v] = sl.vacuous(a); lineage[v] = {v}; flags[v] = False; continue
            # conflict routing (R5)
            flag = any(sl.conflict(x[0], y[0]) > theta_c for i, x in enumerate(inc) for y in inc[i + 1:])
            # group dependent inputs by overlapping lineage (union-find on shared origins)
            groups = []
            for w, lin in inc:
                ng = {"ws": [w], "lin": set(lin)}; keep = []
                for g in groups:
                    if g["lin"] & ng["lin"]:
                        ng["ws"] += g["ws"]; ng["lin"] |= g["lin"]
                    else:
                        keep.append(g)
                groups = keep + [ng]
            fused = None
            for g in groups:
                wg = g["ws"][0] if len(g["ws"]) == 1 else avg_fuse(g["ws"])
                fused = wg if fused is None else sl.fuse(fused, wg)
            fused[3] = a
            op[v] = fused; lineage[v] = set().union(*[g["lin"] for g in groups]) | {v}; flags[v] = flag
    return op, flags


def naive_evaluate(nodes, edges):
    """Same graph, plain cumulative fusion of all inputs (no lineage, no caps)."""
    G = nx.DiGraph(); G.add_nodes_from(nodes); G.add_edges_from([(u, v, {"t": t}) for u, v, t in edges])
    op = {}
    for v in nx.topological_sort(G):
        a = nodes[v].get("a", 0.5); inc = []
        for u in G.predecessors(v):
            t = G.edges[u, v]["t"]
            w = sl.discount(op[u], nodes[v].get("claim", sl.op(1, 0, 0, a))) if t == "DEL" else op[u].copy()
            w[3] = a; inc.append(w)
        if nodes[v].get("prior") is not None:
            inc.append(nodes[v]["prior"])
        if not inc:
            op[v] = sl.vacuous(a); continue
        f = inc[0]
        for w in inc[1:]:
            f = sl.fuse(f, w)
        f[3] = a; op[v] = f
    return op
