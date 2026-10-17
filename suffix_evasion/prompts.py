"""
Attack prompts, fetched rather than vendored.

Two paths to the same rows. `hf_dataset` is the idiomatic Inspect route and is
used when the `datasets` package is present. When it is not, the Hugging Face
dataset-viewer API returns the same records over plain HTTP with no heavy
dependency chain.

The fallback is not a convenience. An eval that only runs on a machine with
several gigabytes of ML tooling installed is an eval most people will read
rather than run, and the point of publishing it is that somebody else can
check the numbers.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

PAGE = 100
_RETRYABLE = frozenset({429, 500, 502, 503, 504})


def _get(url: str, attempts: int = 6) -> dict:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE or attempt == attempts - 1:
                raise
            time.sleep(min(30, 2**attempt))
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(30, 2**attempt))
    raise RuntimeError("unreachable")


def _viewer_rows(dataset: str, split: str, want: int) -> list[dict]:
    """Page the dataset-viewer API until `want` attack rows have been seen."""
    rows: list[dict] = []
    offset = 0
    while len(rows) < want:
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {"dataset": dataset, "config": "default", "split": split,
             "offset": offset, "length": PAGE}
        )
        batch = _get(url).get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        offset += len(batch)
        time.sleep(0.3)
    return rows


def attack_prompts(
    dataset: str,
    *,
    limit: int,
    split: str = "train",
    text_field: str = "text",
    label_field: str = "label",
    is_attack=lambda value: int(value) == 1,
    seed: int = 11,
) -> list[str]:
    """
    A seeded random sample of attack-labelled prompts, de-duplicated.

    Attacks only: a prompt the guardrail was never going to block carries no
    information about evasion, and padding the set with benign rows would make
    every suffix look harmless by dilution.

    Sampled rather than taken in dataset order. These sets arrive grouped by
    source and technique, so the first sixty attacks can be sixty variations of
    one family - fine for comparing two guardrails against each other on
    identical prompts, and not a basis for a number anyone should quote about
    either of them. The seed keeps runs comparable.
    """
    import random
    try:
        from datasets import load_dataset  # noqa: PLC0415

        rows = load_dataset(dataset, split=split)
        records = ({text_field: r[text_field], label_field: r[label_field]} for r in rows)
    except ImportError:
        # Over-fetch: roughly a third of these sets are attacks, so asking for
        # `limit` rows would usually come back short.
        records = iter(_viewer_rows(dataset, split, want=max(limit * 10, 600)))

    seen: set[str] = set()
    prompts: list[str] = []
    for record in records:
        text = (record.get(text_field) or "").strip()
        if not text or text in seen:
            continue
        if not is_attack(record.get(label_field)):
            continue
        seen.add(text)
        prompts.append(text)

    random.Random(seed).shuffle(prompts)
    return prompts[:limit]
