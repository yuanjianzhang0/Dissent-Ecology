from fractions import Fraction

import pytest

from dissent_ecology.protocol import Problem, number, parse_action
from dissent_ecology.verification import TypedVerifier, evaluate


@pytest.mark.parametrize("text, kind", [
    ("KEEP", "KEEP"), ("The sum is wrong.\nREPLACE #### 12", "REPLACE"),
    ("REPLACE #### 1/2", "REPLACE"), ("REPLACE #### 1/0", "INVALID"),
    ("KEEP\nREPLACE #### 4", "INVALID"), ("KEEP\nExplanation", "INVALID"),
    ("REPLACE #### nan", "INVALID"), ("REPLACE #### 1,2", "INVALID"),
    ("I would KEEP", "INVALID"), ("", "INVALID"),
])
def test_action_parsing(text, kind):
    assert parse_action(text).kind == kind


def test_exact_numeric_and_expression():
    assert number("12.5%") == Fraction(1, 8)
    assert number("1,024") == 1024
    assert evaluate("0.1 + 0.2", {}) == Fraction(3, 10)
    with pytest.raises(ValueError):
        evaluate("__import__('os').system('id')", {})
    with pytest.raises(ValueError):
        evaluate("10 ** 1000000", {})


@pytest.mark.parametrize("critic, spec, correct", [
    ("arithmetic", {"values": {"x": 7, "y": 5}, "expression": "x+y"}, "12"),
    ("ratio_basis", {"values": {"part": 3, "base": 12}, "numerator": "part", "denominator": "base", "multiplier": 100}, "25"),
    ("ledger", {"values": {}, "balances": [{"opening": "10", "incoming": "3", "outgoing": "5", "closing": "8"}], "answer_expression": "8"}, "8"),
    ("constraint", {"constraints": [{"left": "answer * 2", "op": "eq", "right": "12"}, {"left": "answer", "op": "gt", "right": "0"}]}, "6"),
])
def test_typed_checks(critic, spec, correct):
    problem = Problem("p", "Problem", "Solution", "0", checks={critic: spec})
    verifier = TypedVerifier()
    assert verifier.verify(critic, problem, correct)
    assert not verifier.verify(critic, problem, "99")


def test_fail_closed_and_countermodel_direct_acceptance():
    verifier = TypedVerifier()
    problem = Problem("p", "Problem", "Solution", "0")
    assert not verifier.verify("arithmetic", problem, "10")
    assert verifier.verify("countermodel", problem, "10")
    problem = Problem("p", "", "", "0", checks={"constraint": {"constraints": []}})
    assert not verifier.verify("constraint", problem, "10")
    problem = Problem("p", "", "", "0", checks={"ratio_basis": {"numerator": "1", "denominator": "0"}})
    assert not verifier.verify("ratio_basis", problem, "1")
