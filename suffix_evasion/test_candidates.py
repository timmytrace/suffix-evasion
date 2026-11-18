"""
Tests for the candidate generator.

Almost all of these are about the *controls*, because a generated arm set is
only as good as the thing it is compared against. If a length-matched control
quietly picks up a digit or the word "card", it stops being a control, every
paired comparison in the run loses its denominator, and the runner reports
promotions that mean nothing. The generator asserts this at construction; these
tests check the assertion is actually reachable and actually holds.

    pytest suffix_evasion/test_candidates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from candidates import (  # noqa: E402
    BENIGN_CLAUSES,
    CLASSES,
    FORMS,
    _assemble_filler,
    by_class,
    generate,
)

#: The widest tolerated miss between a candidate and its control, in
#: characters. The filler is assembled from whole clauses, so it lands near the
#: target rather than on it - but a control that is thirty characters shorter
#: than its treatment is not length-matched, and the confounder the control
#: exists to rule out walks straight back in.
MAX_LENGTH_DELTA = 12


def test_the_grid_is_the_full_cross_product():
    assert len(generate()) == len(CLASSES) * len(FORMS)


def test_names_are_unique():
    names = [c.name for c in generate()]
    assert len(names) == len(set(names))


def test_generation_is_deterministic():
    # The discovery/confirmation split is only meaningful if the same candidate
    # name means the same text on both halves - and across runs, or a promoted
    # bypass in the corpus cannot be re-checked.
    assert [c.text for c in generate()] == [c.text for c in generate()]
    assert [c.control_text for c in generate()] == [c.control_text for c in generate()]


@pytest.mark.parametrize("candidate", generate(), ids=lambda c: c.name)
def test_every_control_is_length_matched(candidate):
    assert abs(candidate.length_delta) <= MAX_LENGTH_DELTA, (
        f"{candidate.name}: control is {candidate.length_delta:+d} characters "
        f"off ({len(candidate.control_text)} vs {len(candidate.text)})"
    )


@pytest.mark.parametrize("candidate", generate(), ids=lambda c: c.name)
def test_no_control_carries_a_digit(candidate):
    # The reference guardrail's dampener keys on digit count. Any digit here
    # and the control shares the treatment's trigger.
    assert not any(ch.isdigit() for ch in candidate.control_text)


@pytest.mark.parametrize("candidate", generate(), ids=lambda c: c.name)
def test_no_control_carries_an_identifier_word(candidate):
    lowered = candidate.control_text.lower()
    for word in ("card", "ssn", "passport", "account"):
        assert word not in lowered


@pytest.mark.parametrize("candidate", generate(), ids=lambda c: c.name)
def test_control_differs_from_the_candidate(candidate):
    assert candidate.control_text != candidate.text


def test_clause_pool_is_itself_clean():
    # The generator's assertion is a backstop. The pool should never have been
    # able to violate it in the first place.
    for clause in BENIGN_CLAUSES:
        assert not any(ch.isdigit() for ch in clause)
        lowered = clause.lower()
        for word in ("card", "ssn", "passport", "account"):
            assert word not in lowered


def test_filler_never_exceeds_its_target():
    # Overshooting would make the control *longer* than the treatment, which
    # is the same confounder pointed the other way.
    for target in range(0, 240):
        assert len(_assemble_filler(target, seed_key=f"t{target}")) <= target


def test_filler_gets_close_once_there_is_room_for_the_shortest_clause():
    shortest = min(len(clause) for clause in BENIGN_CLAUSES)
    for target in range(shortest, 240):
        filler = _assemble_filler(target, seed_key=f"t{target}")
        assert target - len(filler) <= MAX_LENGTH_DELTA, (
            f"target {target} produced {len(filler)} characters"
        )


def test_by_class_filters():
    assert {c.mitigation_class for c in by_class("payment")} == {"payment"}
    assert len(by_class("payment")) == len(FORMS)


def test_every_class_states_a_rationale():
    # A promoted bypass is written into the corpus with this attached. An empty
    # rationale makes the corpus a list of strings instead of a set of claims.
    for klass in CLASSES:
        assert len(klass.rationale) > 40
