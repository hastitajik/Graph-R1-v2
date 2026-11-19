"""Dynamic hypergraph: verified claims are added to a copy of the main hypergraph (no network, no LLM, no GPU)."""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import networkx as nx
import numpy as np

try:
    import faiss
except ImportError:      # the retrieval image has it; a bare environment skips these tests
    faiss = None

from evidence import controller
from evidence import dynamic_graph as D
from evidence.provenance import ProvenanceIndex, norm
from tests.test_evidence_controller import CHAIN, FULL, P, QUESTION, TEXT, FakeLLM, args, make_search, prov

DIM = 8


def fake_embed(texts):
    """Deterministic vectors: the same text always gives the same vector."""
    out = []
    for t in texts:
        h = hashlib.md5(t.encode()).digest()
        v = np.frombuffer(h[:DIM], dtype=np.uint8).astype("float32") + 1.0
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_file(path, data, mode="w", **kw):
    with open(path, mode) as f:
        if mode == "w":
            json.dump(data, f, **kw)
        else:
            f.write(data)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def hyperedge(text):
    return f'{D.PREFIX}"{text}"'


def make_graph_dir(path, wrong_index=False):
    """A small Graph-R1 folder: graph, hyperedge kv store, FAISS index, provenance index."""
    os.makedirs(path)
    G = nx.Graph()
    kv = {}
    for key in ("dir_a", "dir_b", "mom_a", "mom_b"):
        name = hyperedge(TEXT[key])
        G.add_node(name, role="hyperedge", weight=8.0, source_id=f"chunk-{key}")
        kv[D.md5_id(name, "rel-")] = {"content": name, "hyperedge_name": name}
    for ent in ('"BLUE RIVER"', '"ANN LEE"', '"MARY LEE"'):
        G.add_node(ent, role="entity", entity_type="X", description="d", source_id="chunk-e")
    for key, ents in {"dir_a": ('"BLUE RIVER"', '"ANN LEE"'), "dir_b": ('"BLUE RIVER"', '"ANN LEE"'),
                      "mom_a": ('"ANN LEE"', '"MARY LEE"'), "mom_b": ('"ANN LEE"', '"MARY LEE"')}.items():
        for e in ents:
            G.add_edge(hyperedge(TEXT[key]), e, weight=8.0, source_id=f"chunk-{key}")
    nx.write_graphml(G, os.path.join(path, D.GRAPH))
    write_file(os.path.join(path, D.HYPEREDGES), kv, indent=2)
    names = [v["content"] for v in kv.values()]
    index = faiss.index_factory(DIM, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(fake_embed(names[:-1] if wrong_index else names))
    faiss.write_index(index, os.path.join(path, D.HYPEREDGE_INDEX))
    data = {norm(t): {"titles": [title], "chunks": [f"chunk-{k}"], "how": "exact"}
            for k, (t, title, _) in P.items()}
    write_file(os.path.join(path, D.PROVENANCE), data)
    write_file(os.path.join(path, "corpus.npy"), b"large", "wb")     # must not be copied


def solve_one(answer="Mary Lee", question=QUESTION):
    rec = {"question": question, "gold": ["Mary Lee"], "obligations": json.loads(json.dumps(CHAIN))}

    async def go():
        with mock.patch.object(controller, "retrieve_sync", make_search(FULL)), \
                mock.patch.object(controller.asyncio, "sleep", mock.AsyncMock()):
            return await controller.solve(FakeLLM(answer=answer), args(), prov(), rec)
    return asyncio.run(go())


def write_trace(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def checksum(folder):
    return {f: hashlib.md5(read_bytes(os.path.join(folder, f))).hexdigest() for f in sorted(os.listdir(folder))}


@unittest.skipIf(faiss is None, "faiss is not installed")
class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "src")
        self.dst = os.path.join(self.tmp.name, "dst")
        make_graph_dir(self.src)
        self.trace = os.path.join(self.tmp.name, "trace.jsonl")
        self.record = solve_one()
        write_trace(self.trace, [self.record])

    def claims(self):
        return D.claims_from_traces([self.trace])

    def graph(self, folder=None):
        return nx.read_graphml(os.path.join(folder or self.dst, D.GRAPH))

    def kv(self, folder=None):
        return read_json(os.path.join(folder or self.dst, D.HYPEREDGES))

    def index(self, folder=None):
        return faiss.read_index(os.path.join(folder or self.dst, D.HYPEREDGE_INDEX))


class ClaimsFromTrace(Base):
    def test_the_trace_of_a_certified_question_carries_the_support(self):
        self.assertTrue(self.record["certified"])
        o = {x["id"]: x for x in self.record["obligations"]}
        self.assertIn(TEXT["dir_a"], o["o1"]["support"])
        self.assertIn(TEXT["mom_a"], o["o2"]["support"])

    def test_claims_are_the_instantiated_retrieve_obligations(self):
        texts = sorted(c.text for c in self.claims())
        self.assertEqual(texts, ["The director of Blue River is Ann Lee", "The mother of Ann Lee is Mary Lee"])

    def test_uncertified_questions_give_no_claims(self):
        r = dict(self.record, certified=False)
        write_trace(self.trace, [r])
        self.assertEqual(self.claims(), [])

    def test_a_trace_without_support_is_not_usable(self):
        r = json.loads(json.dumps(self.record))
        for o in r["obligations"]:
            o.pop("support")
        write_trace(self.trace, [r])
        self.assertEqual(self.claims(), [])

    def test_uncovered_and_unbound_obligations_are_skipped(self):
        r = json.loads(json.dumps(self.record))
        r["obligations"][1]["covered"] = False
        r["obligations"][0]["text"] = "The director of Blue River is ?D"
        write_trace(self.trace, [r])
        self.assertEqual(self.claims(), [])

    def test_the_same_claim_from_two_questions_is_one_claim_and_wrong_answers_are_counted(self):
        write_trace(self.trace, [self.record, dict(self.record, question="other", em=0.0)])
        claims = {c.text: c for c in self.claims()}
        self.assertEqual(len(claims), 2)
        c = claims["The director of Blue River is Ann Lee"]
        self.assertEqual((len(c.questions), c.wrong), (2, 1))


class Enrich(Base):
    def run_enrich(self, **kw):
        return D.enrich(self.src, self.dst, self.claims(), fake_embed, **kw)

    def test_hyperedges_are_written_the_way_graph_r1_writes_them(self):
        stats = self.run_enrich()
        self.assertEqual((stats["added"], stats["edges_added"]), (2, 4))
        name = hyperedge("The director of Blue River is Ann Lee")
        G = self.graph()
        a = G.nodes[name]
        self.assertEqual((a["role"], a["weight"]), ("hyperedge", D.DEFAULT_WEIGHT))
        self.assertEqual(set(a["source_id"].split(D.SEP)), {"chunk-dir_a", "chunk-dir_b"})
        self.assertEqual(set(G.neighbors(name)), {'"BLUE RIVER"', '"ANN LEE"'})
        self.assertEqual(G.edges[name, '"ANN LEE"']["weight"], D.DEFAULT_WEIGHT)
        self.assertEqual(self.kv()[D.md5_id(name, "rel-")], {"content": name, "hyperedge_name": name})

    def test_the_index_and_the_kv_store_stay_in_step(self):
        self.run_enrich()
        kv, index = self.kv(), self.index()
        self.assertEqual((len(kv), index.ntotal), (6, 6))
        names = [v["content"] for v in kv.values()]
        for name in (hyperedge("The director of Blue River is Ann Lee"), hyperedge("The mother of Ann Lee is Mary Lee")):
            _, ids = index.search(fake_embed([name]), 1)                # the API maps an id to the kv position
            self.assertEqual(names[ids[0][0]], name)

    def test_the_new_hyperedge_is_reached_by_the_entity_path_and_the_hyperedge_path(self):
        self.run_enrich()
        G, name = self.graph(), hyperedge("The mother of Ann Lee is Mary Lee")
        self.assertIn(name, G.neighbors('"MARY LEE"'))               # entity path: entity -> its hyperedges
        _, ids = self.index().search(fake_embed([name]), 3)           # hyperedge path: nearest hyperedge vectors
        self.assertIn(name, [list(self.kv().values())[i]["content"] for i in ids[0]])

    def test_the_base_graph_is_never_modified_and_dumps_are_not_copied(self):
        before = checksum(self.src)
        self.run_enrich()
        self.assertEqual(checksum(self.src), before)
        self.assertFalse(os.path.exists(os.path.join(self.dst, "corpus.npy")))

    def test_provenance_of_a_claim_is_the_article_of_its_evidence(self):
        self.run_enrich()
        idx = ProvenanceIndex.load(os.path.join(self.dst, D.PROVENANCE))
        self.assertEqual(idx.titles("The director of Blue River is Ann Lee"), ["Ann Lee (director)", "Blue River"])
        self.assertEqual(idx.entry("The mother of Ann Lee is Mary Lee")["how"], "claim")
        self.assertEqual(idx.titles(TEXT["dir_a"]), ["Blue River"])            # the original entries are kept

    def test_running_the_same_trace_twice_adds_nothing(self):
        self.run_enrich()
        before = checksum(self.dst)
        stats = self.run_enrich()
        self.assertEqual((stats["already_added"], stats["added"]), (2, 0))
        self.assertEqual(checksum(self.dst), before)

    def test_a_claim_equal_to_an_existing_hyperedge_is_merged_not_duplicated(self):
        c = D.Claim("Blue River was directed by Ann Lee.", [TEXT["dir_a"]])
        # the stored name is the claim text wrapped like every hyperedge, so it already exists
        stats = D.enrich(self.src, self.dst, [c], fake_embed)
        self.assertEqual((stats["merged_into_existing"], stats["added"]), (1, 0))
        self.assertEqual(self.graph().nodes[hyperedge(c.text)]["weight"], 8.0 + D.DEFAULT_WEIGHT)
        self.assertEqual((len(self.kv()), self.index().ntotal), (4, 4))

    def test_evidence_that_is_not_in_the_graph_is_skipped(self):
        c = D.Claim("X was born in Y", ["A passage that the graph does not have."])
        stats = D.enrich(self.src, self.dst, [c], fake_embed)
        self.assertEqual((stats["skipped_evidence_not_in_graph"], stats["added"]), (1, 0))
        self.assertEqual(self.index().ntotal, 4)

    def test_a_claim_naming_no_known_entity_is_added_without_edges(self):
        c = D.Claim("It was released in 1999", [TEXT["dir_a"]])
        stats = D.enrich(self.src, self.dst, [c], fake_embed)
        self.assertEqual((stats["added"], stats["without_entity"]), (1, 1))
        self.assertEqual(list(self.graph().neighbors(hyperedge(c.text))), [])

    def test_a_short_entity_name_must_match_a_whole_word(self):
        self.assertTrue(D.named_in("The director of F is Ann", '"F"'))
        self.assertFalse(D.named_in("The film is Fine", '"F"'))
        self.assertFalse(D.named_in("anything", '""'))

    def test_dry_run_writes_nothing(self):
        stats = self.run_enrich(dry_run=True)
        self.assertEqual(stats["added"], 2)
        self.assertFalse(os.path.exists(self.dst))

    def test_wrong_answers_are_reported(self):
        write_trace(self.trace, [dict(self.record, em=0.0)])
        self.assertEqual(self.run_enrich(dry_run=True)["from_wrong_answers"], 2)


class Refusals(Base):
    def test_the_destination_must_not_be_the_source(self):
        with self.assertRaises(ValueError):
            D.enrich(self.src, self.src, self.claims(), fake_embed)

    def test_a_foreign_existing_folder_is_not_overwritten(self):
        os.makedirs(self.dst)
        with self.assertRaises(ValueError):
            D.enrich(self.src, self.dst, self.claims(), fake_embed)

    def test_an_inconsistent_index_is_refused_and_nothing_is_written(self):
        shutil_src = os.path.join(self.tmp.name, "bad")
        make_graph_dir(shutil_src, wrong_index=True)
        with self.assertRaises(ValueError):
            D.enrich(shutil_src, self.dst, self.claims(), fake_embed)
        self.assertEqual(len(self.kv()), 4)
        self.assertEqual(len(self.graph().nodes), 7)                  # still the copy of the source

    def test_an_embedding_of_the_wrong_shape_is_refused_and_nothing_is_written(self):
        with self.assertRaises(ValueError):
            D.enrich(self.src, self.dst, self.claims(), lambda texts: np.zeros((len(texts), DIM + 1), "float32"))
        self.assertEqual((len(self.kv()), self.index().ntotal), (4, 4))
        self.assertEqual(len(self.graph().nodes), 7)
        self.assertFalse(os.path.exists(os.path.join(self.dst, D.LOG)) and
                         read_json(os.path.join(self.dst, D.LOG))["claims"])


class CommandLine(Base):
    def test_help_and_dry_run(self):
        p = subprocess.run([sys.executable, "-m", "evidence.dynamic_graph", "--help"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--dry-run", p.stdout)
        p = subprocess.run([sys.executable, "-m", "evidence.dynamic_graph", "--trace", self.trace, "--src", self.src,
                            "--dst", self.dst, "--dry-run"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("hyperedges added: 2", p.stdout)
        self.assertFalse(os.path.exists(self.dst))

    def test_a_trace_without_claims_is_an_error(self):
        write_trace(self.trace, [dict(self.record, certified=False)])
        p = subprocess.run([sys.executable, "-m", "evidence.dynamic_graph", "--trace", self.trace, "--src", self.src,
                            "--dst", self.dst], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("no usable claims", p.stderr)


if __name__ == "__main__":
    unittest.main()
