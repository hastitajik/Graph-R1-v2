"""Navigator policy: prompt, reply validation and the controller branch that uses them (no network, no LLM)."""
import asyncio
import json
import subprocess
import sys
import types
import unittest
from unittest import mock

from evidence import controller as C
from evidence import navigator as N
from evidence.audit import audit
from tests.test_evidence_guards import hold, mk

QUESTION = "Where was the director of F born?"
OBLIGATIONS = [{"id": "o1", "text": "The director of F is ?D", "depends_on": [], "kind": "retrieve"},
               {"id": "o2", "text": "?D was born in ?P", "depends_on": ["o1"], "kind": "retrieve"}]


class ScriptClient:
    """Returns the scripted replies in order; an Exception in the script is raised instead of returned."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        if not self.replies:
            raise RuntimeError("no scripted reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        msg = types.SimpleNamespace(content=reply if isinstance(reply, str) else json.dumps(reply))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def make_args(policy="navigator", max_steps=5):
    return types.SimpleNamespace(model="m", search_url="x", max_searches=10, max_steps=max_steps, policy=policy)


def episode(client, policy="navigator", covered_first=False, max_steps=5):
    """An episode over the two-hop question; with covered_first, o1 is already covered by one passage."""
    st = mk(OBLIGATIONS, "o2")
    if covered_first:
        hold(st, "a", "F", "Anna Berg directed F", "o1", {"?D": "Anna Berg"})
    return C.Episode(st, QUESTION, client, make_args(policy, max_steps), C.Weights())


def candidates(ep):
    cert = audit(ep.state)
    return cert, ep.propose(cert, ep.monitor(cert))


def by_kind(acts, kind, oid):
    return next(a for a in acts if a.kind == kind and a.oid == oid)


def run(coro):
    with mock.patch("evidence.controller.asyncio.sleep", new=mock.AsyncMock()):
        return asyncio.run(coro)


class PureFunctions(unittest.TestCase):
    def test_weakness_of_each_action(self):
        ep = episode(None, covered_first=True)
        _, acts = candidates(ep)
        self.assertEqual(N.weakness(by_kind(acts, "FillGap", "o2")), "uncovered")
        self.assertEqual(N.weakness(by_kind(acts, "Verify", "o1")), "corroborate")
        self.assertEqual(N.weakness(by_kind(acts, "Diversify", "o1")), "corroborate")
        self.assertEqual(N.weakness(C.Action("Diversify", "o1", [], None, 0, "tied conflict")), "resolve")
        self.assertEqual(N.weakness(C.Action("Challenge", "o1", [], None, 0, "conflict")), "resolve")
        self.assertEqual(N.weakness(C.Action("Challenge", "o1", [], None, 0, "confirmation bias")), "corroborate")
        self.assertEqual(N.weakness(C.Action("Bridge", "o2", [], None, 0, "upstream hop not found")), "uncovered")

    def test_prompt_describes_state_and_candidates(self):
        ep = episode(None, covered_first=True)
        ep.state.note_query("o2", "Anna Berg born")
        cert, acts = candidates(ep)
        system, user = N.build_prompt(QUESTION, ep.state, cert, acts)
        self.assertIn("Navigator", system)
        self.assertIn(f"Question: {QUESTION}", user)
        self.assertIn("o1 | The director of F is Anna Berg | covered; sources: F; independent sources: 1; checked: no", user)
        self.assertIn("o2 | Anna Berg was born in ?P | NOT covered", user)
        self.assertIn("Certificate: not complete", user)
        self.assertIn("[F] Anna Berg directed F", user)
        self.assertIn("A1 | corroborate | Verify on o1", user)
        self.assertIn("uncovered | FillGap on o2", user)
        self.assertIn("already tried: ['Anna Berg born']", user)

    def test_blocked_obligation_is_marked(self):
        ep = episode(None)
        cert, acts = candidates(ep)
        user = N.build_prompt(QUESTION, ep.state, cert, acts)[1]
        self.assertIn("o2 | ?D was born in ?P | NOT covered; cannot be searched yet", user)

    def test_complete_certificate_is_reported(self):
        st = mk([{"id": "o1", "text": "X was born in ?P", "depends_on": [], "kind": "retrieve"}], "o1")
        hold(st, "a", "X", "X was born in Rome", "o1", {"?P": "Rome"})
        st.verified[("o1", "a")] = True
        text = N.render_state(st, audit(st))
        self.assertIn("checked: yes", text)
        self.assertIn("Certificate: complete", text)

    def test_snippets_are_limited(self):
        st = mk(OBLIGATIONS, "o2")
        for i in range(10):
            hold(st, f"k{i}", f"T{i}", "x" * 500)
        text = N.render_state(st, audit(st))
        shown = [ln for ln in text.splitlines() if ln.startswith("[T")]
        self.assertEqual(len(shown), N.SNIPPETS)
        self.assertTrue(all(len(ln) <= len("[T0] ") + N.SNIPPET_CHARS + 2 for ln in shown))


class ParseChoice(unittest.TestCase):
    def setUp(self):
        self.ep = episode(None, covered_first=True)
        self.cert, self.acts = candidates(self.ep)

    def test_valid_choice_with_queries(self):
        r = N.parse_choice({"action": "A3", "queries": ["  Anna Berg birthplace ", "", 5]}, self.acts, False)
        self.assertIsNone(r["error"])
        self.assertIs(r["action"], self.acts[2])
        self.assertEqual(r["queries"], ["Anna Berg birthplace"])

    def test_queries_are_optional_and_may_be_malformed(self):
        for obj in ({"action": "A1"}, {"action": "A1", "queries": "text"}, {"action": "A1", "queries": None}):
            r = N.parse_choice(obj, self.acts, False)
            self.assertIsNone(r["error"])
            self.assertEqual(r["queries"], [])

    def test_unknown_or_missing_action_is_an_error(self):
        for obj in ({"action": "A9"}, {"action": "FillGap"}, {}, {"action": None}, {"action": "A0"}):
            r = N.parse_choice(obj, self.acts, False)
            self.assertTrue(r["error"], obj)
            self.assertIsNone(r["action"])

    def test_not_an_object_is_an_error(self):
        for obj in (None, [], "A1", 3):
            self.assertTrue(N.parse_choice(obj, self.acts, False)["error"])

    def test_end_only_when_the_certificate_is_complete(self):
        self.assertTrue(N.parse_choice({"end": True}, self.acts, True)["end"])
        r = N.parse_choice({"end": True}, self.acts, False)
        self.assertFalse(r["end"])
        self.assertIn("not complete", r["error"])

    def test_end_must_be_the_boolean_true(self):
        self.assertTrue(N.parse_choice({"end": "true"}, self.acts, True)["error"])


class Navigate(unittest.TestCase):
    def test_no_candidate_means_no_call(self):
        client = ScriptClient()
        ep = episode(client)
        self.assertEqual(run(ep.navigate(audit(ep.state), [])), (None, False, None))
        self.assertEqual(client.calls, [])

    def test_the_models_choice_is_run_with_its_queries(self):
        client = ScriptClient({"action": "A3", "queries": ["Anna Berg birthplace"]})
        ep = episode(client, covered_first=True)
        cert, acts = candidates(ep)
        a, end, note = run(ep.navigate(cert, acts))
        self.assertEqual((a.kind, a.oid, end), ("FillGap", "o2", False))
        self.assertEqual(a.det, ["Anna Berg birthplace", "Anna Berg was born in", "Anna Berg born", "Anna Berg"])
        self.assertIsNone(a.llm_mode)
        self.assertEqual(a.n_search, min(C.Weights().max_queries, 4))
        self.assertEqual(note["weakness"], "uncovered")
        self.assertEqual(ep.nav["chosen"], 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(ep.n_llm, 1)

    def test_queries_already_tried_are_dropped(self):
        ep = episode(ScriptClient({"action": "A3", "queries": ["Anna Berg born", "new query"]}), covered_first=True)
        cert, acts = candidates(ep)
        ep.state.note_query("o2", "Anna Berg born")
        a, _, note = run(ep.navigate(cert, acts))
        self.assertIn("new query", a.det)
        self.assertNotIn("Anna Berg born", a.det)

    def test_without_queries_the_default_queries_stay(self):
        ep = episode(ScriptClient({"action": "A3"}), covered_first=True)
        cert, acts = candidates(ep)
        expected = list(by_kind(acts, "FillGap", "o2").det)
        a, _, note = run(ep.navigate(cert, acts))
        self.assertEqual(a.det, expected)
        self.assertNotIn("queries", note)

    def test_verify_ignores_queries(self):
        ep = episode(ScriptClient({"action": "A1", "queries": ["something"]}), covered_first=True)
        cert, acts = candidates(ep)
        a, _, note = run(ep.navigate(cert, acts))
        self.assertEqual(a.kind, "Verify")
        self.assertEqual(a.det, [])
        self.assertNotIn("queries", note)

    def test_end_is_refused_while_the_certificate_is_incomplete(self):
        ep = episode(ScriptClient({"end": True}), covered_first=True)
        cert, acts = candidates(ep)
        a, end, note = run(ep.navigate(cert, acts))
        self.assertFalse(end)
        self.assertIs(a, max(acts, key=lambda x: x.vne))          # the VNE choice
        self.assertIn("end refused", note["fallback"])
        self.assertEqual(ep.nav["fallback"], 1)

    def test_end_is_accepted_when_the_certificate_is_complete(self):
        st = mk([{"id": "o1", "text": "X was born in ?P", "depends_on": [], "kind": "retrieve"}], "o1")
        hold(st, "a", "X", "X was born in Rome", "o1", {"?P": "Rome"})
        st.verified[("o1", "a")] = True
        client = ScriptClient({"end": True})
        ep = C.Episode(st, "q", client, make_args(), C.Weights())
        cert, acts = candidates(ep)
        self.assertTrue(cert.complete)
        a, end, note = run(ep.navigate(cert, acts or [C.Action("Diversify", "o1", ["q"], None, 0.3, "single source")]))
        self.assertEqual((a, end), (None, True))
        self.assertEqual(ep.nav["end"], 1)

    def test_unusable_replies_fall_back_to_the_vne_choice(self):
        for reply in ("not json at all", {"action": "A9"}, ["A1"]):
            ep = episode(ScriptClient(reply, reply, reply), covered_first=True)
            cert, acts = candidates(ep)
            a, end, note = run(ep.navigate(cert, acts))
            self.assertIs(a, max(acts, key=lambda x: x.vne), reply)
            self.assertFalse(end)
            self.assertIn("fallback", note)

    def test_a_failed_call_falls_back_and_is_counted_once(self):
        err = RuntimeError("server down")
        ep = episode(ScriptClient(err, err, err), covered_first=True)
        cert, acts = candidates(ep)
        a, end, note = run(ep.navigate(cert, acts))
        self.assertIsNotNone(a)
        self.assertIn("RuntimeError", note["fallback"])
        self.assertEqual(ep.nav["fallback"], 1)

    def test_fallback_without_positive_value_returns_no_action(self):
        ep = episode(ScriptClient("garbage", "garbage", "garbage"), covered_first=True)
        cert, acts = candidates(ep)
        for a in acts:
            a.vne = -1.0
        self.assertEqual(run(ep.navigate(cert, acts)), (None, False, {"fallback": mock.ANY}))


class RunLoop(unittest.TestCase):
    def search(self):
        return mock.patch("evidence.controller.retrieve_sync", return_value="[]")

    def test_navigator_policy_runs_the_chosen_action_and_records_it(self):
        client = ScriptClient({"action": "A1", "queries": ["director of F"]})
        ep = episode(client, max_steps=1)
        with self.search() as retrieve:
            stop, cert = run(ep.run())
        self.assertEqual(stop, "max_steps")
        step = ep.steps[0]
        self.assertEqual((step["action"], step["oid"]), ("FillGap", "o1"))
        self.assertEqual(step["navigator"]["weakness"], "uncovered")
        self.assertEqual(step["queries"][0], "director of F")
        self.assertEqual(retrieve.call_args_list[0].args[1], "director of F")
        self.assertEqual(ep.nav["chosen"], 1)

    def test_default_policy_never_calls_the_navigator(self):
        client = ScriptClient()
        ep = episode(client, policy="vne", max_steps=1)
        with self.search():
            run(ep.run())
        self.assertEqual(client.calls, [])
        self.assertNotIn("navigator", ep.steps[0])
        self.assertEqual(sum(ep.nav.values()), 0)

    def test_a_failed_navigator_call_does_not_stop_the_question(self):
        err = RuntimeError("server down")
        ep = episode(ScriptClient(err, err, err), max_steps=1)
        with self.search():
            stop, _ = run(ep.run())
        self.assertEqual(stop, "max_steps")
        self.assertEqual(ep.steps[0]["action"], "FillGap")
        self.assertIn("fallback", ep.steps[0]["navigator"])

    def test_the_model_cannot_end_an_uncertified_question(self):
        ep = episode(ScriptClient({"end": True}), max_steps=1)
        with self.search():
            stop, cert = run(ep.run())
        self.assertNotEqual(stop, "certified")
        self.assertFalse(cert.complete)

    def test_the_budget_still_applies(self):
        ep = episode(ScriptClient({"action": "A1", "queries": ["a", "b", "c"]}), max_steps=3)
        ep.args.max_searches = 1
        with self.search() as retrieve:
            stop, _ = run(ep.run())
        self.assertEqual(stop, "budget")
        retrieve.assert_not_called()


class CommandLine(unittest.TestCase):
    def help_of(self, module):
        p = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout

    def test_flags_are_exposed(self):
        self.assertIn("--policy", self.help_of("evidence.controller"))
        self.assertIn("--navigator", self.help_of("evidence.pipeline"))


if __name__ == "__main__":
    unittest.main()
