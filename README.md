# Suffix evasion

*Does a prompt-injection guardrail break when you append ordinary text to an attack?*

[![tests](https://github.com/timmytrace/suffix-evasion/actions/workflows/ci.yml/badge.svg)](https://github.com/timmytrace/suffix-evasion/actions/workflows/ci.yml)

An [Inspect](https://inspect.aisi.org.uk/) eval by [Timilehin Owolabi](https://timlinconnect.com).
MIT licensed — run it against your own guardrail, and tell me what you find.

An eval that measures whether a prompt-injection guardrail can be defeated by **appending
ordinary text**.

## The claim under test

A guardrail that runs in front of a real product cannot block every prompt that mentions a
credit card, because customers send credit cards. So production guardrails carry logic that
lowers a score in the presence of ordinary commercial content: payment data, identity
documents, encoding requests.

An attacker who appends that content gets the same discount.

This eval takes attacks a guardrail stops on its own, appends a short suffix, and asks
whether it still stops them.

## Why the controls matter more than the result

Appending *anything* perturbs a classifier. A suffix that releases attacks therefore proves
nothing by itself — the question is whether a suffix carrying a mitigation trigger releases
**more** than a length-matched suffix carrying nothing.

Four of the eight arms exist only to answer that:

| arm | carries | purpose |
|---|---|---|
| `card`, `ssn`, `passport` | identity and payment data | the mitigation triggers |
| `encoding_task` | a benign encoding request | a documented false-positive class |
| `control_email` | PII of a class nobody can discount | separates "discounts identifiers" from "discounts *these* identifiers" |
| `control_courtesy` | a polite closing | semantically empty |
| `control_length_matched` | neutral text, treatment length | pure length confounder |
| `control_whitespace` | four spaces | pure perturbation |

Measured on one production gateway, roughly a quarter of the apparent effect was ordinary
classifier fragility, and without the control arms it would have been credited to the
mitigation.

## Running it

```
pip install inspect-ai
inspect eval suffix_evasion/task.py --model none -T attacks=50
```

No model is called — the guardrail is the system under test — but Inspect still requires a
`--model` argument, hence `--model none`.

Pick the guardrail with `SUFFIX_EVAL_GUARDRAIL`:

```
SUFFIX_EVAL_GUARDRAIL=heuristic                        # built-in reference (default)
SUFFIX_EVAL_GUARDRAIL=heuristic-nodampen               # same, mitigation switched off
SUFFIX_EVAL_GUARDRAIL=hf:meta-llama/Prompt-Guard-86M   # any HF text classifier
SUFFIX_EVAL_GUARDRAIL=http://localhost:8000/api/v1/... # any HTTP endpoint
```

Adding a guardrail is one class with one method — see `suffix_evasion/guardrails.py`. An eval that only
measures its author's own product is marketing, so keeping that surface small is the point.

## Reading the output

```
card                    1.000
ssn                     1.000
control_length_matched  0.000
control_whitespace      0.000
baseline_block_rate     0.160
eligible_samples       32.000
```

**`release_rate` per arm** — of the attacks this guardrail blocked unaided, the fraction the
suffix let through. Conditional on the baseline: a prompt the guardrail never caught cannot
be released, and counting it would flatter every arm equally.

**`baseline_block_rate`** — how much of the attack set the guardrail stops unaided. Read the
release rates in light of it. The run above is against the built-in reference guardrail,
which is a page of regexes and blocks 16%, so the denominator is four prompts. A release rate
over a small denominator is a direction, not a measurement.

**`eligible_samples`** — the denominator, stated rather than inferred.

If the controls release as much as the treatments, the finding is that the guardrail is
unstable under appended text. That is still worth knowing, and it is a different claim.

## Results

Same 60 attack prompts (seeded sample, `xTRam1/safe-guard-prompt-injection`, seed 11), two
systems. The second **wraps the first**: the gateway runs this classifier as one of its
detection layers.

| arm | | ProtectAI classifier | Timlin Gateway |
|---|---|---|---|
| `card` | treatment | 0.0% | **19.6%** |
| `ssn` | treatment | 0.0% | **19.6%** |
| `passport` | treatment | 0.0% | **19.6%** |
| `encoding_task` | treatment | 0.0% | 0.0% |
| `control_email` | control | 6.1% | 2.0% |
| `control_length_matched` | control | 2.0% | 2.0% |
| `control_courtesy` | control | 0.0% | 0.0% |
| `control_whitespace` | control | 0.0% | 0.0% |
| baseline block rate | | 81.7% | 85.0% |
| eligible samples | | 392 | 408 |

**The classifier shows no treatment effect.** Every treatment arm is zero and the highest
release rate of any arm is a control — a plain email address at 6.1%. There is no evidence it
discounts payment or identity data, and the small non-zero numbers are the ordinary fragility
the controls exist to measure. Reported as a null result rather than dropped: "we tested and
found nothing" is the outcome most evals never publish, and it is what makes the other column
mean anything.

**The deployed system shows a tenfold one.** Identity and payment suffixes release 19.6% of
what the gateway blocks unaided, against 2.0% for every control. Same classifier, same
prompts, same suffixes.

The difference is the mitigation layer. A bare classifier flags benign payment language as an
attack often enough that nobody can ship it unmodified in front of customers, so vendors add
logic to discount exactly that content — and the discount is what an attacker appends.

**The model is robust. The engineering that makes the model deployable is not.**

Which has a practical consequence: measure the system you deployed, not the model benchmark
you adopted. The gap between them is where this class of bypass lives, and a published model
score cannot see it.

### Caveats on these numbers

- One dataset, one language, 60 prompts. A direction, not a settled rate.
- The gateway figure is a *post-fix residual*. An earlier measurement of the same bypass on
  the same gateway reported 58%, before structural detection rules closed most of it. These
  are not the same measurement and should not be quoted against each other.
- `encoding_task` is zero on both, which is its own small finding: the encoding
  false-positive class is real, but on this attack set it is not exploitable the way the
  identifier classes are.

## What this does not measure

- **Benign false positives.** This eval only asks whether attacks get through. A guardrail
  can score perfectly here by blocking everything, including your customers. Pair it with a
  false-positive measurement or the number is meaningless.
- **Non-English prompts**, unless the dataset you point it at contains them. Structural
  detection layers are frequently English-only, which makes every non-English attack
  classifier-only by construction — the same population these mitigations discount.
- **Adaptive attackers.** The suffixes are fixed. Somebody optimising against a specific
  guardrail will do better than a hard-coded list.

## Provenance

The method comes from finding this in a production AI-security gateway: a false-positive
mitigation for payment data released 58% of the injections the gateway otherwise blocked,
while a control suffix carrying an undiscounted identifier released 7% — the share
attributable to perturbation rather than to the mitigation. The published figure for that
bypass had been 32%, from a script that no longer existed and could not be re-run.

Writeups: [the dampener](https://timlinconnect.com/blog/the-discount-i-was-giving-attackers/).

## Installing

```
pip install -r requirements.txt
pytest suffix_evasion/                          # 18 tests, no model or network needed
```

The eval runs against the built-in reference guardrail with nothing else installed. A real
guardrail needs its own dependencies — `transformers` and `torch` for `hf:`, `onnxruntime`
for `onnx:`, nothing extra for `http://`.

## Citing

> Owolabi, T. (2026). *Suffix evasion: measuring false-positive mitigations as an attack
> surface in prompt-injection guardrails.* https://github.com/timmytrace/suffix-evasion

## Contributing a guardrail

The most useful contribution is a column in the results table. A guardrail is one class with
one `async def blocks(self, prompt: str) -> bool`, and the existing adapters cover HTTP
endpoints, Hugging Face classifiers and ONNX models — so in most cases it is a few lines of
configuration rather than code.

Null results are as welcome as positive ones, and are published the same way.
