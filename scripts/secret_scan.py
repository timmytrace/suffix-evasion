#!/usr/bin/env python3
"""Scan for credentials before they reach the repository.

Why this exists
---------------
A review of the gateway repository found a live SECRET_KEY and the admin password
committed in `.env.example`, in a repository that was public at the time. Both
values were valid. The fix was not a grep for one credential class - that does
not generalise - but a check for all of them on every commit, so nobody has to
remember.

This copy exists because the repository with the scanner was the *private* one.
The public repositories had no scanning at all, which is the wrong way round.

Shared file
-----------
Byte-identical copies live in three repositories:

    timlin-gateway/scripts/secret_scan.py
    timlin-connect/scripts/secret_scan.py
    suffix-evasion/scripts/secret_scan.py

Identical on purpose: three variants would drift, and the drift would be
invisible. `diff` between any two copies should be empty, and a patch belongs
in all three. There is no shared package because three repositories with no
shared build is not enough to justify one - but that is a trade, not a
tidiness claim, and if these ever disagree the oldest one is probably wrong.

Design constraint
-----------------
A detector that cannot be quiet gets switched off. The DLP engine's own tests
contain *deliberately fake* secrets (`AKIAIOSFODNN7EXAMPLE`, `sk-abcdef...`, a
dummy RSA header), and application code is full of lines that look like
assignments but are references (`process.env.GEMINI_API_KEY`,
`settings.OPENAI_API_KEY`). Findings are therefore filtered by:

  1. a known-example allowlist (AWS's published documentation key, etc.)
  2. recognition of code references and variable interpolation
  3. Shannon entropy - real credentials are random, `abcdefghijklmnop` is not
  4. an inline escape hatch: put `secret-scan: allow` on the line

Usage
-----
    python scripts/secret_scan.py --staged     # pre-commit hook
    python scripts/secret_scan.py --all        # tracked files (CI)
    python scripts/secret_scan.py path ...     # explicit paths

Exit code 1 if anything is found.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys

ALLOW_MARKER = "secret-scan: allow"

# Values that are published examples or obviously synthetic. Matching is
# case-insensitive and substring-based.
KNOWN_EXAMPLES = (
    "akiaiosfodnn7example",          # AWS documentation key
    "wjalrxutnfemi/k7mdeng",         # AWS documentation secret
    "4111111111111111",              # test Visa
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789abcdef",
    "your-",
    "example",
    "changeme",
    "placeholder",
    "dummy",
    "<token>",
    "xxxxx",
    "strongpassword",
)

# Values that are conventional defaults or non-secrets by construction. Matched
# exactly (case-insensitive) rather than as substrings, so a real password that
# merely contains "admin" is still reported.
COMMON_DEFAULTS = {
    "postgres", "password", "passwd", "root", "admin", "secret", "test",
    "changeme", "changeme123", "changeme123!", "mysecretpassword", "guest",
}

# Prefixes that mean "this is a reference to a secret, not a secret".
#
# Without this, every line that reads a credential from the environment is a
# finding: `apiKey = process.env.GEMINI_API_KEY` is 26 unbroken characters
# assigned to a secret-shaped name. A history scan of these repositories
# produced thirteen such hits and one real leak, and a ratio like that is how a
# scanner gets ignored - so the references are recognised rather than tuned
# away with a length threshold that would also hide short real passwords.
CODE_REFERENCE_PREFIXES = (
    "process.env", "import.meta.env", "os.environ", "os.getenv", "getenv(",
    "settings.", "self.", "cls.", "config.", "conf.", "env.", "Deno.env",
    "secrets.", "vault.", "credentials.", "this.",
)

#: A Python/pydantic type annotation rather than a value. See is_known_example.
TYPE_ANNOTATION = re.compile(
    r"^(Optional|Union|List|Dict|Set|Tuple|Any|Literal|SecretStr|Field"
    r"|str|int|float|bool|bytes|None)\b"
)

# (name, pattern, minimum entropy required to report)
PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), 3.0),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"), 3.0),
    ("openai_key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_\-]{20,}\b"), 3.2),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"), 3.2),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 2.8),
    ("github_token", re.compile(r"\b(ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}\b"), 3.0),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b"), 3.0),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), 3.0),
    # NVIDIA NIM / build.nvidia.com. Added because a key of this form was in
    # use on this machine while the eval was being run against NVIDIA-hosted
    # models, and the original scanner did not know the prefix - so the one
    # credential most likely to be pasted into a repository by accident was
    # the one credential it could not have caught.
    ("nvidia_key", re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}\b"), 3.2),
    # Netlify personal access token. Same reasoning: the marketing site is
    # deployed there, so this is a token that exists.
    ("netlify_token", re.compile(r"\bnfp_[A-Za-z0-9]{20,}\b"), 3.0),
    # Both prefixes: keys issued as msk_ before the rename are still valid,
    # because verification hashes the whole token and never checks the prefix.
    # A scanner that only knew the new one would miss every legacy key.
    ("timlin_api_key", re.compile(r"\b(?:msk|tlk)_[A-Za-z0-9_\-]{30,}\b"), 3.5),
    # Require real key material after the header. A bare header followed by
    # prose ("----BEGIN RSA PRIVATE KEY----- some data") is a test fixture, not
    # a key; real keys carry a long base64 body.
    ("private_key_block",
     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\r\n\s]*[A-Za-z0-9+/=]{60,}"),
     0.0),
    # Assignment of a long, random-looking value to a secret-shaped name.
    # Threshold note: hexadecimal cannot exceed 4.0 bits/char (log2 of 16), and
    # a random 64-char hex key measures ~3.88. A 4.0 threshold was therefore
    # impossible to meet for hex secrets - which is precisely the format of the
    # SECRET_KEY that leaked. 3.3 admits hex and base64 while still rejecting
    # words and sequential filler.
    ("secret_assignment",
     re.compile(r"(?i)\b(secret_key|api_key|password|passwd|token|access_key)\s*[=:]\s*['\"]?([A-Za-z0-9/+_\-]{24,})['\"]?"),
     3.3),
    # Named credentials that must never carry a real value in a tracked file.
    # Generic short-password detection is unusable (far too many false
    # positives), but these specific names are unambiguous, and one of them is
    # how the admin password leaked. Any value that is not empty and not an
    # obvious placeholder or reference is reported regardless of length, which
    # is the only way to catch a *short* real credential.
    #
    # Deliberately NOT listed here: OPENAI_API_KEY, ANTHROPIC_API_KEY,
    # GEMINI_API_KEY, NVIDIA_API_KEY, HF_TOKEN. Every one of those has a
    # distinctive prefix with its own pattern above, so listing the variable
    # name as well adds no coverage - and because this rule has no entropy
    # floor, it does add noise: it fires on `OPENAI_API_KEY: Optional[str]`,
    # on `OPENAI_API_KEY="sk-test"` in a unit test, and on
    # `ANTHROPIC_API_KEY=sk-ant-...` in a README. Eight findings, none real.
    # The names kept below are the ones with no recognisable format, where
    # matching the name is the only way to see the value at all.
    ("known_credential_var",
     re.compile(
         r"(?i)\b(FIRST_ADMIN_PASSWORD|ADMIN_PASSWORD|POSTGRES_PASSWORD|DB_PASSWORD"
         r"|ROOT_PASSWORD|MYSQL_ROOT_PASSWORD|CANARY_LOG_KEY|NETLIFY_AUTH_TOKEN"
         r"|LOG_ENCRYPTION_KEY|SECRET_KEY_PREVIOUS)"
         r"\s*[=:]\s*['\"]?([^\s'\"#]{6,})['\"]?"),
     0.0),
]

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".next",
             "dist", "build", ".pytest_cache", "deberta-guard-v1", ".netlify",
             "logs", "coverage"}
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".woff", ".woff2",
            ".zip", ".gz", ".bin", ".safetensors", ".pt", ".onnx", ".lock",
            ".eval"}


def entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def is_known_example(text: str) -> bool:
    stripped = text.strip()
    # A SOPS-encrypted value is ciphertext, not a secret: ENC[AES256_GCM,...].
    # The whole point of committing secrets.enc.env is that these are safe.
    if stripped.startswith("ENC["):
        return True
    low = stripped.lower()
    # A variable reference or a documentation placeholder is not a value:
    # ${POSTGRES_PASSWORD}, $DB_PASS, {{ vault_secret }}, <openssl rand -hex 24>
    # carry nothing to leak. Angle brackets are the usual fill-in-the-blank
    # convention in setup docs, and no real credential begins with "<".
    if low.startswith(("$", "{{", "<", "%(")) or low in ("''", '""'):
        return True
    # A read from the environment or a settings object, in any language.
    if any(low.startswith(p.lower()) for p in CODE_REFERENCE_PREFIXES):
        return True
    # A Python type annotation, not an assignment. `SECRET_KEY_PREVIOUS:
    # Optional[str] = None` contains a colon followed by thirteen unbroken
    # characters, which is indistinguishable from `KEY: abcdefghijklm` to a
    # regex that accepts either separator - and accepting both separators is
    # what lets this catch YAML.
    if TYPE_ANNOTATION.match(stripped):
        return True
    # A documentation placeholder that trails off: `sk-ant-...`, `nvapi-...`.
    # The convention is near-universal in setup instructions and no real
    # credential ends in an ellipsis.
    if stripped.endswith(("...", "…")):
        return True
    if low in COMMON_DEFAULTS:
        return True
    return any(ex in low for ex in KNOWN_EXAMPLES)


def scan_text(path: str, text: str) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARKER in line:
            continue
        for name, rx, min_entropy in PATTERNS:
            for m in rx.finditer(line):
                # Where a pattern captures (name, value), judge the *value*.
                # Matching on the whole span would mean a reference such as
                # POSTGRES_PASSWORD=${POSTGRES_PASSWORD} never looks like a
                # variable reference, because the span starts with the var name.
                value = m.group(2) if (m.lastindex or 0) >= 2 else m.group(0)
                if is_known_example(value):
                    continue
                if min_entropy and entropy(value) < min_entropy:
                    continue
                redacted = value[:6] + "..." if len(value) > 10 else value
                hits.append((lineno, name, redacted))
    return hits


def iter_files(paths: list[str]):
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for f in files:
                    if os.path.splitext(f)[1].lower() not in SKIP_EXT:
                        yield os.path.join(root, f)
        elif os.path.isfile(p):
            if os.path.splitext(p)[1].lower() not in SKIP_EXT:
                yield p


def tracked_files() -> list[str]:
    """Files git actually tracks.

    Used for --all so the scan matches what can reach the remote. Scanning
    ignored files would flag deploy/.env.prod, which is intentionally local and
    intentionally full of real secrets.
    """
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=False
    ).stdout
    return [
        f for f in out.splitlines()
        if f.strip()
        and os.path.exists(f)
        and os.path.splitext(f)[1].lower() not in SKIP_EXT
        and not any(part in SKIP_DIRS for part in f.split("/"))
    ]


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=False,
    ).stdout
    return [f for f in out.splitlines() if f.strip() and os.path.exists(f)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--staged", action="store_true", help="scan staged files")
    ap.add_argument("--all", action="store_true", help="scan tracked files")
    args = ap.parse_args()

    if args.staged:
        targets = staged_files()
    elif args.all or not args.paths:
        targets = tracked_files()
    else:
        targets = list(iter_files(args.paths))

    findings = []
    for path in targets:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        for lineno, name, redacted in scan_text(path, text):
            findings.append((path, lineno, name, redacted))

    if not findings:
        print(f"secret-scan: clean ({len(targets)} files)")
        return 0

    print("secret-scan: POTENTIAL CREDENTIALS FOUND\n", file=sys.stderr)
    for path, lineno, name, redacted in findings:
        print(f"  {path}:{lineno}  [{name}]  {redacted}", file=sys.stderr)
    print(
        "\nIf a finding is a false positive, append '# secret-scan: allow' to the line.\n"
        "If it is real: do NOT just delete it and commit. Rotate the credential first,\n"
        "because it is already in your reflog and in any clone taken since.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
