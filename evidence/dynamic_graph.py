"""Dynamic hypergraph: verified claims and their evidence are added to the main hypergraph.

  python3 -m evidence.dynamic_graph --trace runs/a/trace_controller.jsonl \
      --src expr/2WikiMultiHopQA --dst expr/2WikiMultiHopQA_dyn

A claim is a required retrieve obligation of a CERTIFIED question, written out with its values (for example
"The director of F is Anna Berg"). Its evidence is the passages that support it. Each claim becomes one hyperedge of the
main hypergraph, written exactly the way Graph-R1 writes the hyperedges it extracts (graphr1/operate.py):

  hyperedge node   name "<hyperedge>" + '"claim"', role "hyperedge", weight, source_id (chunk ids of its evidence)
  edges            hyperedge - entity, for the entities that already exist and are named in the claim
  kv store         kv_store_hyperedges.json, key "rel-" + md5(name), {"content": name, "hyperedge_name": name}
  FAISS            one vector appended to index_hyperedge.bin, in the same order as the kv store
  provenance       the evidence article titles of the claim, so the audit does not count the claim as a new source

Retrieval is untouched: the dual-path search (entity path and hyperedge path, rank fusion) finds the new hyperedges the
same way it finds the original ones. Nothing is written into --src; the enriched graph lives in --dst, a copy, and the
retrieval API is started on it with --data_source <dst name>. Running a trace again does not add its claims twice.

Limits to know:
  - A certified answer is not always right, and its claims are then added too. The report of every run lists how many
    claims came from questions whose answer was wrong, when the trace holds that information.
  - Questions that produced the claims must not be used to measure the benefit: their own claims answer them. Evaluate
    on questions that are not in the trace (evidence.pipeline --exclude).
  - A claim is a restatement of evidence from the original corpus; it adds no new source, only a verified, compact
    form that is linked to the entities and chunks of that evidence.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field

from evidence.provenance import SEP, ProvenanceIndex, clean_hyperedge, norm

GRAPH = "graph_chunk_entity_relation.graphml"
HYPEREDGES = "kv_store_hyperedges.json"
HYPEREDGE_INDEX = "index_hyperedge.bin"
PROVENANCE = "provenance_index.json"
LOG = "dynamic_graph.json"
PREFIX = "<hyperedge>"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_WEIGHT = 1.0    # the weight Graph-R1 gives a hyper-relation that comes without a score (operate.py)


@dataclass
class Claim:
    text: str
    support: list = field(default_factory=list)     # texts of the passages that support the claim
    questions: list = field(default_factory=list)
    wrong: int = 0                                  # how many of its questions were answered wrongly (em == 0)


def md5_id(content, prefix=""):
    return prefix + hashlib.md5(content.encode()).hexdigest()    # same as graphr1.utils.compute_mdhash_id


def claim_key(text):
    return norm(text)


def claims_from_traces(paths):
    """Claims of the certified questions of one or more controller traces, one per distinct text."""
    claims = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line) if line.strip() else None
                if not r or not r.get("certified"):
                    continue
                for o in r.get("obligations", []):
                    text = (o.get("text") or "").strip()
                    if o.get("kind") != "retrieve" or not o.get("covered") or not text or "?" in text:
                        continue
                    if not o.get("support"):
                        continue          # a trace written before supports were recorded cannot be used
                    c = claims.setdefault(claim_key(text), Claim(text))
                    c.support = list(dict.fromkeys(c.support + o["support"]))
                    c.questions.append(r.get("question", ""))
                    c.wrong += int(r.get("em", 1.0) == 0.0)
    return list(claims.values())


def split_ids(value):
    return [s for s in str(value or "").split(SEP) if s]


def entity_name_form(node):
    return node.strip().strip('"').lower()


def named_in(claim_text, entity):
    """Is the entity (a graph node name) mentioned in the claim? Whole words, case-insensitive."""
    e = entity_name_form(entity)
    return bool(e) and re.search(r"(?<!\w)" + re.escape(e) + r"(?!\w)", claim_text.lower()) is not None


def bge_embedder(device="cpu"):
    """The embedding model Graph-R1 uses for its FAISS indexes (script_build.py), as a function texts -> float32."""
    import numpy as np
    from FlagEmbedding import FlagAutoModel

    model = FlagAutoModel.from_finetuned(
        EMBED_MODEL, query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        devices=device)
    return lambda texts: np.asarray(model.encode_corpus(list(texts)), dtype="float32")


def copy_graph(src, dst):
    """Copy the graph folder (without the large .npy embedding dumps, which the API never reads)."""
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("--dst must differ from --src: the base graph is never modified")
    if os.path.exists(dst) and not os.path.exists(os.path.join(dst, LOG)):
        raise ValueError(f"{dst} exists and is not an enriched copy made by this tool")
    if not os.path.exists(dst):
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.npy"))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path, **kw):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, **kw)
    os.replace(tmp, path)


def enrich(src, dst, claims, embed=None, weight=DEFAULT_WEIGHT, dry_run=False):
    """Add the claims to the copy of the graph in dst. Returns a Counter of what happened.

    embed: function texts -> float32 matrix; required unless dry_run."""
    import networkx as nx

    stats = Counter()
    if not dry_run:
        copy_graph(src, dst)
    base = src if dry_run else dst
    G = nx.read_graphml(os.path.join(base, GRAPH))
    kv = read_json(os.path.join(base, HYPEREDGES))
    prov = ProvenanceIndex.load(os.path.join(base, PROVENANCE)) if os.path.exists(os.path.join(base, PROVENANCE)) \
        else ProvenanceIndex({})
    log_path = os.path.join(base, LOG)
    applied = set(read_json(log_path)["claims"]) if os.path.exists(log_path) else set()
    stats["nodes_before"], stats["edges_before"], stats["hyperedges_before"] = \
        G.number_of_nodes(), G.number_of_edges(), len(kv)

    by_text = {norm(clean_hyperedge(n)): n for n, a in G.nodes(data=True) if a.get("role") == "hyperedge"}
    new_names = []
    for c in claims:
        stats["claims"] += 1
        stats["from_wrong_answers"] += int(c.wrong > 0)
        if claim_key(c.text) in applied:
            stats["already_added"] += 1
            continue
        nodes = list(dict.fromkeys(by_text[norm(s)] for s in c.support if norm(s) in by_text))
        if not nodes:
            stats["skipped_evidence_not_in_graph"] += 1
            continue
        source_ids = sorted({i for n in nodes for i in split_ids(G.nodes[n].get("source_id"))})
        members = sorted({e for n in nodes for e in G.neighbors(n)
                          if G.nodes[e].get("role") == "entity" and named_in(c.text, e)})
        name = f'{PREFIX}"{c.text}"'
        if G.has_node(name):
            old = G.nodes[name]
            G.nodes[name]["weight"] = float(old.get("weight", 0.0)) + weight
            G.nodes[name]["source_id"] = SEP.join(sorted(set(split_ids(old.get("source_id"))) | set(source_ids)))
            stats["merged_into_existing"] += 1
        else:
            G.add_node(name, role="hyperedge", weight=float(weight), source_id=SEP.join(source_ids))
            key = md5_id(name, "rel-")
            if key not in kv:
                kv[key] = {"content": name, "hyperedge_name": name}
                new_names.append(name)
            stats["added"] += 1
        for e in members:
            if G.has_edge(name, e):
                d = G.edges[name, e]
                d["weight"] = float(d.get("weight", 0.0)) + weight
                d["source_id"] = SEP.join(sorted(set(split_ids(d.get("source_id"))) | set(source_ids)))
            else:
                G.add_edge(name, e, weight=float(weight), source_id=SEP.join(source_ids))
        stats["edges_added"] += len(members)
        stats["without_entity"] += int(not members)
        titles = sorted({t for s in c.support for t in prov.titles(s)})
        entry = prov.data.setdefault(claim_key(c.text), {"titles": [], "chunks": [], "how": "claim"})
        entry["titles"] = sorted(set(entry["titles"]) | set(titles))
        entry["chunks"] = sorted(set(entry["chunks"]) | set(source_ids))
        applied.add(claim_key(c.text))

    stats["nodes_after"], stats["edges_after"], stats["hyperedges_after"] = \
        G.number_of_nodes(), G.number_of_edges(), len(kv)
    if dry_run:
        return stats

    import faiss
    import numpy as np

    index_path = os.path.join(dst, HYPEREDGE_INDEX)
    index = faiss.read_index(index_path)
    if index.ntotal != stats["hyperedges_before"]:
        raise ValueError(f"{HYPEREDGE_INDEX} holds {index.ntotal} vectors but {HYPEREDGES} holds "
                         f"{stats['hyperedges_before']} entries: refusing to extend an inconsistent pair")
    if new_names:
        vectors = np.asarray(embed(new_names), dtype="float32")
        if vectors.shape != (len(new_names), index.d):
            raise ValueError(f"embedding has shape {vectors.shape}, expected ({len(new_names)}, {index.d})")
        index.add(vectors)
    stats["faiss_after"] = index.ntotal
    # nothing is written before the embeddings succeeded; the index and the kv store are written together
    faiss.write_index(index, index_path)
    write_json(kv, os.path.join(dst, HYPEREDGES), indent=2)
    nx.write_graphml(G, os.path.join(dst, GRAPH))
    write_json(prov.data, os.path.join(dst, PROVENANCE))
    write_json({"claims": sorted(applied)}, log_path)
    return stats


def show(stats):
    labels = [("claims", "claims in the traces (certified questions)"), ("added", "hyperedges added"),
              ("merged_into_existing", "merged into an existing hyperedge"), ("already_added", "added by an earlier run"),
              ("skipped_evidence_not_in_graph", "skipped: evidence text not found in the graph"),
              ("without_entity", "added with no entity link (reachable by the hyperedge path only)"),
              ("edges_added", "hyperedge-entity edges written"),
              ("from_wrong_answers", "claims from questions whose answer was wrong")]
    for key, label in labels:
        print(f"{label}: {stats[key]}")
    for what in ("nodes", "edges", "hyperedges"):
        print(f"{what}: {stats[what + '_before']} -> {stats[what + '_after']}")
    if "faiss_after" in stats:
        print(f"{HYPEREDGE_INDEX}: {stats['hyperedges_before']} -> {stats['faiss_after']} vectors")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", nargs="+", required=True, help="controller trace(s) written by evidence.controller")
    ap.add_argument("--src", default=None, help="graph folder to read (default: EVIDENCE_DATA_DIR)")
    ap.add_argument("--dst", required=True, help="folder of the enriched copy (created, or updated if made by this tool)")
    ap.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    ap.add_argument("--device", default="cpu", help="device of the embedding model")
    ap.add_argument("--dry-run", action="store_true", help="count what would be added; write nothing, load no model")
    args = ap.parse_args()
    if args.src is None:
        from evidence.common import DATA_DIR
        args.src = DATA_DIR
    claims = claims_from_traces(args.trace)
    if not claims:
        ap.error("no usable claims: the traces need certified questions and the 'support' field (written by this version)")
    embed = None if args.dry_run else bge_embedder(args.device)
    show(enrich(args.src, args.dst, claims, embed, args.weight, args.dry_run))
    if not args.dry_run:
        print(f"enriched graph: {args.dst}   start the API on it with: python3 script_api.py --data_source "
              f"{os.path.basename(os.path.normpath(args.dst))}   (it reads expr/<data_source>)")


if __name__ == "__main__":
    main()
