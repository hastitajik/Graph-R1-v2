"""Closed-book (no search) reference system and the model-knowledge section of the report (no network, no LLM)."""
import asyncio
import subprocess
import sys
import types
import unittest
from unittest import mock

from evidence import baseline as B
from evidence.compare import closed_book_section, keyed_records, paired, report


def fake_client(replies):
    """A chat client that returns the given replies in order and records every call."""
    it = iter(replies)
    seen = []

    async def create(**kw):
        seen.append(kw)
        msg = types.SimpleNamespace(content=next(it))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return client, seen


def closed_args():
    return types.SimpleNamespace(model="m", search_url="", max_turns=4, info_chars=1000, short=True, no_search=True)


def rec(q, em, **kw):
    return {"question": q, "em": em, "f1": em, "searches": 0, "turns": 1, **kw}


class ClosedBookRun(unittest.TestCase):
    def test_answers_without_any_search(self):
        client, seen = fake_client(["<think>x</think>\n<answer>\nLiverpool\n</answer>"])

        def no_search(*a, **k):
            raise AssertionError("the closed-book system must not search")

        with mock.patch.object(B, "retrieve_sync", no_search):
            r = asyncio.run(B.run_one(client, closed_args(), {"question": "Where was X born?", "gold": ["Liverpool"]}))
        self.assertEqual((r["answer"], r["em"], r["searches"], r["turns"], r["mode"]),
                         ("Liverpool", 1.0, 0, 1, "closed_book"))
        self.assertNotIn("grounded", r)
        self.assertEqual(len(seen), 1)
        prompt = seen[0]["messages"][0]["content"]
        self.assertIn("cannot search", prompt)
        self.assertNotIn("<query>", prompt)

    def test_reply_without_answer_tag_is_used_as_the_answer(self):
        client, _ = fake_client(["Liverpool"])
        r = asyncio.run(B.run_one(client, closed_args(), {"question": "q", "gold": ["Liverpool"]}))
        self.assertEqual((r["answer"], r["em"]), ("Liverpool", 1.0))

    def test_a_failed_call_is_recorded_not_raised(self):
        async def create(**kw):
            raise RuntimeError("server down")

        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        with mock.patch("evidence.baseline.asyncio.sleep", new=mock.AsyncMock()):
            r = asyncio.run(B.run_one(client, closed_args(), {"question": "q", "gold": ["x"]}))
        self.assertTrue(r["error"].startswith("RuntimeError"))
        self.assertEqual((r["answer"], r["em"]), ("", 0.0))

    def test_retrieval_baseline_is_unchanged_without_the_flag(self):
        client, _ = fake_client(["<think>y</think>\n<answer>\nAnna Berg\n</answer>"])
        args = types.SimpleNamespace(model="m", search_url="x", max_turns=4, info_chars=1000, short=False)
        r = asyncio.run(B.run_one(client, args, {"question": "Who directed F?", "gold": ["Anna Berg"]}))
        self.assertIn("grounded", r)
        self.assertNotIn("mode", r)


class ModelKnowledge(unittest.TestCase):
    def runs(self):
        closed = [rec("a", 1.0, mode="closed_book"), rec("b", 0.0, mode="closed_book"),
                  rec("c", 1.0, mode="closed_book"), rec("d", 0.0, mode="closed_book")]
        base = [rec("a", 1.0, grounded=True), rec("b", 1.0, grounded=True),
                rec("c", 0.0, grounded=False), rec("d", 0.0, grounded=False)]
        ctrl = [rec("a", 1.0, certified=True), rec("b", 1.0, certified=True),
                rec("c", 0.0, certified=False), rec("d", 0.0, certified=False)]
        return {"baseline": base, "controller": ctrl, "closed_book": closed}

    def test_two_by_two_table_and_share(self):
        text = "\n".join(closed_book_section(self.runs()))
        self.assertIn("| closed-book right | 1 | 1 |", text)
        self.assertIn("| closed-book wrong | 1 | 1 |", text)
        self.assertIn("baseline is right on 2 of 4 questions; 1 of those were already right without any search.", text)

    def test_certified_answers_are_counted_for_controller_runs(self):
        text = "\n".join(closed_book_section(self.runs()))
        self.assertIn("Certified and right: 2; 1 of those were already right without any search.", text)

    def test_report_contains_the_section_and_no_grounded_row_for_closed_book(self):
        text = report(self.runs())
        self.assertIn("## Model knowledge", text)
        self.assertIn("| baseline | 2/4 |", text)          # grounded table lists the retrieval baseline
        self.assertNotIn("| closed_book | 0/", text)         # and not the closed-book run

    def test_no_section_without_a_closed_book_run(self):
        runs = self.runs()
        del runs["closed_book"]
        self.assertEqual(closed_book_section(runs), [])
        self.assertNotIn("Model knowledge", report(runs))


class KeyedRecords(unittest.TestCase):
    def test_repeated_question_is_matched_occurrence_by_occurrence(self):
        rs = [rec("q", 1.0), rec("q", 0.0)]
        self.assertEqual(sorted(keyed_records(rs)), [("q", 1), ("q", 2)])
        c = paired(rs, rs)
        self.assertEqual((c["n"], c["wins"], c["losses"]), (2, 0, 0))


class CommandLine(unittest.TestCase):
    def help_of(self, module):
        p = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout

    def test_flags_are_exposed(self):
        self.assertIn("--no-search", self.help_of("evidence.baseline"))
        self.assertIn("--closed-book", self.help_of("evidence.pipeline"))


if __name__ == "__main__":
    unittest.main()
