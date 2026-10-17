"""
Run the eval against a guardrail behind HTTP, with auth and a custom verdict.

`inspect eval task.py` covers the simple cases through SUFFIX_EVAL_GUARDRAIL.
A deployed gateway usually needs more than a URL: a bearer token, a nested
response, and a verdict that is not a single boolean. This is the worked
example for that, and it is deliberately a plain script rather than more
configuration - the thing you have to get right is *which field means blocked*,
and that is a judgement, not a setting.

    GATEWAY_URL=http://127.0.0.1:8011 \
    GATEWAY_EMAIL=... GATEWAY_PASSWORD=... \
    python suffix_evasion/run_http.py --attacks 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inspect_ai import Task, eval as inspect_eval  # noqa: E402
from inspect_ai.scorer import grouped  # noqa: E402

from guardrails import HttpGuardrail  # noqa: E402
from prompts import attack_prompts  # noqa: E402
from suffixes import ALL  # noqa: E402
from task import (  # noqa: E402
    DEFAULT_DATASET,
    _cross,
    baseline_block_rate,
    eligible_samples,
    held_under_suffix,
    probe_guardrail,
    release_rate,
)


def login(base: str, email: str, password: str) -> str:
    request = urllib.request.Request(
        base.rstrip("/") + "/api/v1/auth/login",
        data=json.dumps({"email": email, "password": password}).encode(),
        method="POST",
    )
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["access_token"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--base", default=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8011"))
    ap.add_argument("--attacks", type=int, default=60)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--token", default=os.environ.get("GATEWAY_TOKEN"))
    args = ap.parse_args()

    token = args.token
    if not token:
        email = os.environ.get("GATEWAY_EMAIL")
        password = os.environ.get("GATEWAY_PASSWORD")
        if not (email and password):
            print("set GATEWAY_TOKEN, or GATEWAY_EMAIL and GATEWAY_PASSWORD", file=sys.stderr)
            return 2
        token = login(args.base, email, password)

    guardrail = HttpGuardrail(
        name="timlin-gateway",
        url=args.base.rstrip("/") + "/api/v1/gateway/scan/input",
        headers={"Authorization": f"Bearer {token}"},
        # NOT is_safe. That field is `injection_clean AND no_pii_found`, so a
        # card number in the suffix flips it to "unsafe" for a data-loss reason
        # while the injection underneath is released - which would score this
        # arm as a perfect block and hide the finding. blocked_reason is set by
        # injection detection alone, which is the question being asked.
        blocked_if=lambda response: response.get("blocked_reason") is not None,
    )

    task = Task(
        dataset=_cross(attack_prompts(args.dataset, limit=args.attacks), ALL),
        solver=probe_guardrail(guardrail),
        scorer=held_under_suffix(),
        metrics=[
            grouped(release_rate(), "suffix_name"),
            baseline_block_rate(),
            eligible_samples(),
        ],
        name="suffix_evasion_http",
    )

    inspect_eval(task, model="none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
