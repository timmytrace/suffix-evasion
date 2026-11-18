"""
The red-team runner: search for new bypasses, promote the ones that survive.

    python suffix_evasion/runner.py --guardrail heuristic --attacks 80
    python suffix_evasion/runner.py --guardrail heuristic --attacks 80 --write

The eval in `task.py` is a *measurement*: a fixed set of arms, argued for by
hand, reported with controls. This is the other mode - a search. It sweeps the
generated grid in `candidates.py`, puts every candidate through the four gates
in `promote.py`, and writes whatever survives to `corpus/`, where `task.py` can
pick it up as eval arms. A bypass found on Monday is in the published eval
without anyone editing the hand-written set.

Nothing is written unless `--write` is passed. The default is an exploratory
run, because a search that silently promotes while someone is fiddling with
thresholds is how a corpus fills up with things nobody can reproduce.

Why the search is separate from the eval
----------------------------------------
Inspect is built for measuring a defined task, and the discovery/confirmation
split here needs the *same* candidate measured on two disjoint prompt halves
with a decision in between. Expressing that as one Inspect task means either
two tasks and a join, or a solver that knows about the split - and the join is
precisely the mistake `task.py` was written to avoid. So the search is plain
asyncio over the guardrail interface, and its output goes back into Inspect as
arms. Each tool does the thing it is good at.

Cost
----
One clean call per prompt, then two per candidate per *eligible* prompt. A
prompt the guardrail does not block unaided is skipped entirely rather than
measured and discarded, which on a typical set removes about half the work -
and the full grid is 55 candidates, so that saving is the difference between a
runnable nightly job and one nobody waits for.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import corpus  # noqa: E402
from candidates import Candidate, generate  # noqa: E402
from guardrails import Guardrail, from_env  # noqa: E402
from promote import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_MIN_DISCORDANT,
    DEFAULT_MIN_ELIGIBLE,
    CandidateVerdict,
    PairedOutcome,
    Verdict,
    evaluate,
    split_prompts,
)

DEFAULT_DATASET = "xTRam1/safe-guard-prompt-injection"


@dataclass
class RunResult:
    guardrail: str
    verdicts: dict[str, CandidateVerdict]
    discovery_indices: list[int]
    confirmation_indices: list[int]
    eligible_discovery: int
    eligible_confirmation: int
    guardrail_calls: int
    seconds: float
    candidates: dict[str, Candidate] = field(default_factory=dict)

    @property
    def promoted(self) -> list[CandidateVerdict]:
        return sorted(
            (v for v in self.verdicts.values() if v.promoted), key=lambda v: v.name
        )

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {v.value: 0 for v in Verdict}
        for verdict in self.verdicts.values():
            tally[verdict.verdict.value] += 1
        return tally


class _Probe:
    """
    The guardrail, asked once per distinct string.

    Memoised for two reasons. The obvious one is cost: the same clean prompt is
    the baseline for all 55 candidates. The other matters more - a guardrail
    with any nondeterminism in it (a hosted API, an LLM judge, a sampled
    classifier) can answer differently on two calls, and if the baseline for
    the candidate arm disagrees with the baseline for the control arm then the
    pair is no longer a pair. Asking once makes the comparison within-prompt by
    construction instead of by hope.
    """

    def __init__(self, guard: Guardrail) -> None:
        self.guard = guard
        self._cache: dict[str, bool] = {}
        self.calls = 0

    async def blocks(self, text: str) -> bool:
        if text not in self._cache:
            self._cache[text] = await self.guard.blocks(text)
            self.calls += 1
        return self._cache[text]


async def run(
    guard: Guardrail,
    prompts: list[str],
    candidates: list[Candidate] | None = None,
    *,
    split_seed: int = 11,
    alpha: float = DEFAULT_ALPHA,
    min_eligible: int = DEFAULT_MIN_ELIGIBLE,
    min_discordant: int = DEFAULT_MIN_DISCORDANT,
    progress=None,
) -> RunResult:
    """
    Search `candidates` against `guard` over `prompts`.

    The split happens first, before a single measurement, and is seeded. That
    ordering is the whole value of the gate: a split chosen or re-chosen after
    looking at results is not a confirmation half, it is a second discovery
    half wearing its clothes.
    """
    started = time.monotonic()
    arms = candidates if candidates is not None else generate()
    probe = _Probe(guard)

    discovery_idx, confirmation_idx = split_prompts(
        list(range(len(prompts))), seed=split_seed
    )
    halves = {"discovery": discovery_idx, "confirmation": confirmation_idx}

    outcomes: dict[str, dict[str, list[PairedOutcome]]] = {
        "discovery": {arm.name: [] for arm in arms},
        "confirmation": {arm.name: [] for arm in arms},
    }
    eligible = {"discovery": 0, "confirmation": 0}

    for half, indices in halves.items():
        for position, index in enumerate(indices):
            prompt = prompts[index]
            blocked_clean = await probe.blocks(prompt)

            if not blocked_clean:
                # Ineligible: a prompt the guardrail never stopped cannot be
                # released, and `promote.py` would drop it anyway. Skipping it
                # here rather than there avoids 110 calls per prompt for a
                # sample that carries no signal either way.
                continue

            eligible[half] += 1
            for arm in arms:
                outcomes[half][arm.name].append(
                    PairedOutcome(
                        prompt_index=index,
                        blocked_clean=True,
                        blocked_candidate=await probe.blocks(prompt + arm.text),
                        blocked_control=await probe.blocks(prompt + arm.control_text),
                    )
                )

            if progress:
                progress(half, position + 1, len(indices), probe.calls)

    verdicts = evaluate(
        outcomes["discovery"],
        outcomes["confirmation"],
        alpha=alpha,
        min_eligible=min_eligible,
        min_discordant=min_discordant,
    )
    for name, verdict in verdicts.items():
        verdict.prompt_indices = {
            "discovery": [o.prompt_index for o in outcomes["discovery"][name]],
            "confirmation": [o.prompt_index for o in outcomes["confirmation"][name]],
        }

    return RunResult(
        guardrail=guard.name,
        verdicts=verdicts,
        discovery_indices=discovery_idx,
        confirmation_indices=confirmation_idx,
        eligible_discovery=eligible["discovery"],
        eligible_confirmation=eligible["confirmation"],
        guardrail_calls=probe.calls,
        seconds=time.monotonic() - started,
        candidates={arm.name: arm for arm in arms},
    )


def _table(discordance) -> dict:
    if discordance is None:
        return {}
    return {
        "n_eligible": discordance.n_eligible,
        "candidate_only": discordance.candidate_only,
        "control_only": discordance.control_only,
        "both": discordance.both,
        "neither": discordance.neither,
        "p_value": discordance.p_value,
        "candidate_release_rate": discordance.candidate_release_rate,
        "control_release_rate": discordance.control_release_rate,
    }


def report(result: RunResult, *, dataset: str, seed: int, limit: int) -> dict:
    """
    The full run, every verdict included.

    Promotions alone would be a press release. The denominator - how many
    candidates were tested, how many had too little data, how many passed
    discovery and then failed on prompts they had not seen - is what lets a
    reader judge whether the promoted ones mean anything.
    """
    return {
        "guardrail": result.guardrail,
        "date": corpus.today(),
        "dataset": dataset,
        "dataset_seed": seed,
        "dataset_limit": limit,
        "candidates_tested": len(result.verdicts),
        "eligible_prompts": {
            "discovery": result.eligible_discovery,
            "confirmation": result.eligible_confirmation,
        },
        "prompts_in_half": {
            "discovery": len(result.discovery_indices),
            "confirmation": len(result.confirmation_indices),
        },
        "guardrail_calls": result.guardrail_calls,
        "seconds": round(result.seconds, 1),
        "verdict_counts": result.counts(),
        "verdicts": [
            {
                "name": verdict.name,
                "verdict": verdict.verdict.value,
                "reason": verdict.reason,
                "mitigation_class": getattr(
                    result.candidates.get(verdict.name), "mitigation_class", ""
                ),
                "form": getattr(result.candidates.get(verdict.name), "form", ""),
                "discovery": _table(verdict.discovery),
                "confirmation": _table(verdict.confirmation),
                "discovery_threshold": verdict.discovery_threshold,
                "confirmation_threshold": verdict.confirmation_threshold,
            }
            for verdict in sorted(result.verdicts.values(), key=lambda v: v.name)
        ],
    }


def to_confirmed(
    result: RunResult, *, spec: str | None, dataset: str, seed: int, limit: int
) -> list[corpus.ConfirmedBypass]:
    stamp = corpus.today()
    confirmed: list[corpus.ConfirmedBypass] = []
    for verdict in result.promoted:
        arm = result.candidates[verdict.name]
        confirmed.append(
            corpus.ConfirmedBypass(
                name=arm.name,
                text=arm.text,
                control_text=arm.control_text,
                mitigation_class=arm.mitigation_class,
                form=arm.form,
                rationale=arm.rationale,
                guardrail=result.guardrail,
                guardrail_spec=spec,
                dataset=dataset,
                dataset_seed=seed,
                dataset_limit=limit,
                first_confirmed=stamp,
                last_confirmed=stamp,
                discovery=_table(verdict.discovery),
                confirmation=_table(verdict.confirmation),
            )
        )
    return confirmed


def summarise(result: RunResult) -> str:
    counts = result.counts()
    lines = [
        f"guardrail        {result.guardrail}",
        f"candidates       {len(result.verdicts)}",
        f"eligible prompts {result.eligible_discovery} discovery / "
        f"{result.eligible_confirmation} confirmation",
        f"guardrail calls  {result.guardrail_calls} in {result.seconds:.0f}s",
        "",
        f"promoted             {counts['promoted']}",
        f"no effect            {counts['no_effect']}",
        f"failed confirmation  {counts['failed_confirmation']}",
        f"insufficient data    {counts['insufficient_data']}",
    ]
    if result.promoted:
        lines.append("")
        lines.append("confirmed bypasses")
        for verdict in result.promoted:
            lines.append(f"  {verdict.name}")
            lines.append(f"      {verdict.reason}")
    else:
        lines.append("")
        lines.append("No candidate survived both halves. Nothing promoted.")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for suffix bypasses and promote the ones that reproduce.",
    )
    parser.add_argument(
        "--guardrail",
        default=None,
        help="guardrail spec, as SUFFIX_EVAL_GUARDRAIL (default: that variable)",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--attacks",
        type=int,
        default=80,
        help="attack prompts to fetch, split in half before measuring (default: 80)",
    )
    parser.add_argument("--seed", type=int, default=11, help="dataset sampling seed")
    parser.add_argument(
        "--split-seed", type=int, default=11, help="discovery/confirmation split seed"
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--min-eligible", type=int, default=DEFAULT_MIN_ELIGIBLE)
    parser.add_argument("--min-discordant", type=int, default=DEFAULT_MIN_DISCORDANT)
    parser.add_argument(
        "--class",
        dest="mitigation_class",
        default=None,
        help="restrict to one mitigation class (default: the whole grid)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write corpus/confirmed.json and the run report (default: off)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full report to stdout"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    args = _parse_args(argv)
    if args.guardrail:
        os.environ["SUFFIX_EVAL_GUARDRAIL"] = args.guardrail
    spec = os.environ.get("SUFFIX_EVAL_GUARDRAIL", "heuristic")
    guard = from_env()

    from prompts import attack_prompts  # noqa: PLC0415

    prompts = attack_prompts(args.dataset, limit=args.attacks, seed=args.seed)
    arms = [
        arm
        for arm in generate()
        if args.mitigation_class in (None, arm.mitigation_class)
    ]
    if not arms:
        print(f"no candidates in class {args.mitigation_class!r}", file=sys.stderr)
        return 2

    print(
        f"{len(prompts)} prompts, {len(arms)} candidates, guardrail {guard.name}",
        file=sys.stderr,
    )

    def progress(half: str, done: int, total: int, calls: int) -> None:
        print(f"\r  {half} {done}/{total}  ({calls} calls)", end="", file=sys.stderr)

    result = asyncio.run(
        run(
            guard,
            prompts,
            arms,
            split_seed=args.split_seed,
            alpha=args.alpha,
            min_eligible=args.min_eligible,
            min_discordant=args.min_discordant,
            progress=progress,
        )
    )
    print("", file=sys.stderr)
    print(summarise(result))

    full = report(result, dataset=args.dataset, seed=args.seed, limit=args.attacks)
    if args.json:
        print(json.dumps(full, indent=2))

    if args.write:
        existing = corpus.load()
        promoted = to_confirmed(
            result, spec=spec, dataset=args.dataset, seed=args.seed, limit=args.attacks
        )
        existing[result.guardrail] = corpus.merge(
            existing.get(result.guardrail, []), promoted
        )
        written = corpus.save(existing)
        report_path = corpus.save_run_report(result.guardrail, full)
        print(f"\nwrote {written}", file=sys.stderr)
        print(f"wrote {report_path}", file=sys.stderr)
    else:
        print(
            "\n(nothing written; pass --write to update the corpus)", file=sys.stderr
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
