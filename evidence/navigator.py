"""Training-free Navigator: a frozen LLM that chooses the next action of the controller.

It plays the role of the Navigator in Argus (Section 2 of that paper): it looks at the whole evidence state and names
the weakness to fix next. Argus trains its Navigator with reinforcement learning; here nothing is trained, the model is
only prompted, so it can be expected to decide worse than a trained one. Weaknesses map to the controller's actions:

  uncovered    a required obligation has no supporting passage         FillGap, Bridge
  corroborate  a covered fact rests on one source or an unchecked one   Diversify, Challenge, Verify
  resolve      a variable has conflicting values that cannot be ranked  Diversify, Challenge

The model only PROPOSES. The candidate actions come from Episode.propose, so an action that is banned, exhausted or
impossible cannot be chosen. Passages are still accepted or refused by the guards and the certificate is still issued
by the audit, never by the model. The model may write the search queries of the chosen action, and it may say that the
evidence is sufficient; that ends the question only when the certificate is complete. A reply that is invalid, or a
failed call, falls back to the choice of the VNE rule and is recorded in the trace.

This module has no network and no controller import: it builds the prompt and validates the reply. The call itself
is made by Episode.navigate.
"""
from evidence.audit import is_verified, required_retrieve

SNIPPETS = 4          # passages shown to the model (best coherence first)
SNIPPET_CHARS = 160

NAVIGATOR_SYSTEM = """You are the Navigator of an evidence-based question answering system.
A question was split into obligations: facts that must each be shown by a retrieved passage before the answer can be certified. You see which obligations are covered and by which sources, any conflicting values, and whether the certificate is complete.
Each candidate action fixes one weakness of the evidence:
- uncovered: a required obligation has no supporting passage yet
- corroborate: a covered fact rests on one source or on a passage that was not checked
- resolve: a variable has conflicting values that the evidence cannot yet rank
Choose the single action to run next. You may write the search queries of that action: short, self-contained, entity names exactly as written, never a query that was already tried. If the certificate is complete and no further action is worth its cost, answer end.
Return ONLY JSON: {"action": "A2", "queries": ["..."]} or {"end": true}"""


def weakness(action):
    """The Argus weakness an action addresses."""
    if action.kind in ("FillGap", "Bridge"):
        return "uncovered"
    if action.why in ("tied conflict", "conflict"):
        return "resolve"
    return "corroborate"


def render_state(state, cert):
    """The evidence state as text: obligations with status and sources, open conflicts, certificate status, and the
    best passages retrieved so far."""
    lines = ["Obligations:"]
    for oid in required_retrieve(state):
        text = state.instantiate(oid)
        if state.covered(oid):
            titles = sorted({t for k in state.live_support(oid) for t in state.items[k].titles})
            status = (f"covered; sources: {', '.join(titles) or 'unknown'}; independent sources: "
                      f"{cert.kappa.get(oid, 0)}; checked: {'yes' if is_verified(state, oid) else 'no'}")
        elif oid in cert.blocked:
            status = "NOT covered; cannot be searched yet, an earlier obligation is missing"
        else:
            status = "NOT covered"
        lines.append(f"{oid} | {text} | {status}")
    open_conflicts = [c for c in cert.conflicts if not c["resolved"]]
    if open_conflicts:
        lines.append("Conflicts:")
        lines += [f"{c['var']} has the values {c['values']}" for c in open_conflicts]
    lines.append("Certificate: complete" if cert.complete else "Certificate: not complete, " + "; ".join(cert.reasons))
    best = sorted(state.items.values(), key=lambda it: -it.coherence)[:SNIPPETS]
    if best:
        lines.append("Best passages so far:")
        lines += [f"[{' | '.join(it.titles) or '?'}] {it.text[:SNIPPET_CHARS]}" for it in best]
    return "\n".join(lines)


def render_candidates(state, acts):
    """One line per candidate action, with the queries that are ready and the ones already tried."""
    lines = ["Candidate actions:"]
    for i, a in enumerate(acts, start=1):
        line = f"A{i} | {weakness(a)} | {a.kind} on {a.oid}: {state.instantiate(a.oid)} | {a.why}"
        if a.det:
            line += f" | queries ready: {list(a.det)}"
        sent = list(state.queries.get(a.oid, []))      # as written, in the order they were sent
        if sent:
            line += f" | already tried: {sent}"
        lines.append(line)
    return "\n".join(lines)


def build_prompt(question, state, cert, acts):
    """(system, user) messages of one navigation step."""
    user = f"Question: {question}\n\n{render_state(state, cert)}\n\n{render_candidates(state, acts)}"
    return NAVIGATOR_SYSTEM, user


def parse_choice(obj, acts, complete):
    """Validate the model's reply.

    Returns {"end": bool, "action": Action or None, "queries": [str], "error": str or None}. An error means the
    reply cannot be used and the caller falls back to the VNE choice. "end" is accepted only when the certificate is
    complete, because the certificate, not the model, decides when a question is finished."""
    def bad(why):
        return {"end": False, "action": None, "queries": [], "error": why}

    if not isinstance(obj, dict):
        return bad("the reply is not a JSON object")
    if obj.get("end") is True:
        return {"end": True, "action": None, "queries": [], "error": None} if complete \
            else bad("end refused: the certificate is not complete")
    chosen = {f"A{i}": a for i, a in enumerate(acts, start=1)}.get(str(obj.get("action", "")).strip())
    if chosen is None:
        return bad(f"unknown action {obj.get('action')!r}")
    qs = obj.get("queries")
    queries = [q.strip() for q in qs if isinstance(q, str) and q.strip()] if isinstance(qs, list) else []
    return {"end": False, "action": chosen, "queries": queries, "error": None}
