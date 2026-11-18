"""
The corpus: confirmed bypasses, on disk, in the repository.

This is the runner's output and the point of the whole exercise. A bypass that
lives in a log file nobody reads has not been promoted into anything. A bypass
in a versioned JSON file, with the measurement that justified it and the run
that produced it, is a ruleset - and `git log` over this directory is the
history of what this guardrail stopped stopping, which is a thing almost
nobody publishes.

Two files per guardrail, both overwritten in place rather than appended:

    corpus/confirmed.json          every guardrail's promoted candidates
    corpus/runs/<slug>.json        the latest full report for one guardrail

Overwritten on purpose. Git is already an append-only log with better tools
than a folder of timestamped files, and a diff that shows a bypass
*disappearing* is the most interesting output this runner can produce - it
means a vendor shipped a fix, and nothing else in the repository would notice.

What is deliberately not stored
-------------------------------
The attack prompts. They belong to a third-party dataset which this repository
fetches rather than vendors, and that decision does not get quietly reversed by
a results file. Each record keeps `dataset`, `seed`, `limit` and the indices
used in each half, which is enough to reconstruct the exact prompt set and not
enough to redistribute it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

#: Repo root, two levels up from this file (suffix_evasion/corpus.py).
ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"
CONFIRMED = CORPUS_DIR / "confirmed.json"
RUNS_DIR = CORPUS_DIR / "runs"


def slug(name: str) -> str:
    """
    A filename from a guardrail name.

    Guardrail specs carry colons and slashes - `hf:meta-llama/Prompt-Guard-86M`,
    `onnx:/models/deberta` - so they cannot be used as paths. Reduced here
    rather than at each call site, because two call sites that slugify
    differently write two files for one guardrail and the second one looks
    like a guardrail that has never been tested.
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unnamed"


@dataclass(frozen=True)
class ConfirmedBypass:
    """One promoted candidate, with everything needed to re-check it."""

    name: str
    text: str
    control_text: str
    mitigation_class: str
    form: str
    rationale: str
    guardrail: str
    #: The spec string that reconstructs the guardrail, where one exists.
    #: `null` for guardrails configured in code, which cannot be re-run from
    #: a record alone - stated rather than implied by a missing field.
    guardrail_spec: str | None
    dataset: str
    dataset_seed: int
    dataset_limit: int
    first_confirmed: str
    last_confirmed: str
    discovery: dict
    confirmation: dict

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "text": self.text,
            "control_text": self.control_text,
            "mitigation_class": self.mitigation_class,
            "form": self.form,
            "rationale": self.rationale,
            "guardrail": self.guardrail,
            "guardrail_spec": self.guardrail_spec,
            "dataset": self.dataset,
            "dataset_seed": self.dataset_seed,
            "dataset_limit": self.dataset_limit,
            "first_confirmed": self.first_confirmed,
            "last_confirmed": self.last_confirmed,
            "discovery": self.discovery,
            "confirmation": self.confirmation,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "ConfirmedBypass":
        return cls(
            name=payload["name"],
            text=payload["text"],
            control_text=payload["control_text"],
            mitigation_class=payload.get("mitigation_class", ""),
            form=payload.get("form", ""),
            rationale=payload.get("rationale", ""),
            guardrail=payload["guardrail"],
            guardrail_spec=payload.get("guardrail_spec"),
            dataset=payload.get("dataset", ""),
            dataset_seed=int(payload.get("dataset_seed", 0)),
            dataset_limit=int(payload.get("dataset_limit", 0)),
            first_confirmed=payload.get("first_confirmed", ""),
            last_confirmed=payload.get("last_confirmed", ""),
            discovery=payload.get("discovery", {}),
            confirmation=payload.get("confirmation", {}),
        )


def load(path: Path | None = None) -> dict[str, list[ConfirmedBypass]]:
    """Confirmed bypasses, keyed by guardrail name. Empty when none exist."""
    target = path or CONFIRMED
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return {
        guardrail: [ConfirmedBypass.from_json(item) for item in items]
        for guardrail, items in payload.get("guardrails", {}).items()
    }


def save(
    confirmed: dict[str, list[ConfirmedBypass]], path: Path | None = None
) -> Path:
    target = path or CONFIRMED
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "Confirmed suffix bypasses. Written by suffix_evasion/runner.py. "
            "Each entry survived a paired comparison against a length-matched "
            "control on a discovery half and again on a confirmation half it "
            "had not been searched against. Attack prompts are not stored; "
            "dataset, seed and limit reconstruct them."
        ),
        "guardrails": {
            guardrail: [item.to_json() for item in items]
            for guardrail, items in sorted(confirmed.items())
        },
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def merge(
    existing: list[ConfirmedBypass], promoted: list[ConfirmedBypass]
) -> list[ConfirmedBypass]:
    """
    Fold this run's promotions into what was already confirmed.

    A candidate already in the corpus keeps its `first_confirmed` date and
    takes the new measurement - so the file carries how long a bypass has been
    reproducible, which is the number that says whether a vendor is fixing
    these.

    A candidate in the corpus that this run did *not* promote is dropped. That
    is the intended behaviour and the reason these files are overwritten: the
    corpus states what reproduces *today*, and a bypass vanishing from the
    diff is a fix landing upstream. Keeping stale entries would turn this into
    a list of things that used to be true, presented as things that are.
    """
    by_name = {item.name: item for item in existing}
    merged: list[ConfirmedBypass] = []
    for item in promoted:
        previous = by_name.get(item.name)
        first = previous.first_confirmed if previous else item.last_confirmed
        merged.append(
            ConfirmedBypass(**{**item.__dict__, "first_confirmed": first or item.last_confirmed})
        )
    return sorted(merged, key=lambda item: item.name)


def save_run_report(guardrail: str, report: dict, runs_dir: Path | None = None) -> Path:
    """The full report for one guardrail: every verdict, not just promotions."""
    directory = runs_dir or RUNS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{slug(guardrail)}.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return target


def as_suffixes(guardrail: str, path: Path | None = None):
    """
    The confirmed set for one guardrail, as eval arms.

    This is how a promotion reaches the measurement: `task.py` can run the
    corpus instead of the hand-written set, so a bypass found by the runner on
    Monday is in the published eval without anyone editing `suffixes.py`.

    Each bypass arrives with its own length-matched control alongside it, named
    `control_for__<name>`. The hand-written set shares one control across all
    arms, which stops being adequate once arm lengths vary - and a generated
    arm set whose control is not length-matched is a worse measurement than no
    arm set at all.
    """
    from suffixes import Suffix  # noqa: PLC0415  (see sys.path note in task.py)

    arms: list[Suffix] = []
    for item in load(path).get(guardrail, []):
        arms.append(
            Suffix(
                name=item.name,
                text=item.text,
                control=False,
                rationale=item.rationale,
            )
        )
        arms.append(
            Suffix(
                name=f"control_for__{item.name}",
                text=item.control_text,
                control=True,
                rationale=(
                    f"Length-matched control for {item.name} "
                    f"({len(item.control_text)} vs {len(item.text)} characters), "
                    "carrying no identifier and no mitigation trigger."
                ),
            )
        )
    return tuple(arms)


def today() -> str:
    return date.today().isoformat()
