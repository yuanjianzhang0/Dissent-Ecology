import pytest

from dissent_ecology.backend import AuditResult
from dissent_ecology.engine import ActiveSuccession, RunConfig
from dissent_ecology.protocol import Problem


class Router:
    def __init__(self):
        self.states = []

    def scores(self, text, history, consumed_ratio, candidates):
        self.states.append((list(history), consumed_ratio))
        return {name: {"ledger": 0.9, "arithmetic": .8, "countermodel": .1}[name] for name in candidates}


class Backend:
    def __init__(self, actions):
        self.actions, self.calls = actions, []

    def audit(self, problem, critic, max_tokens):
        self.calls.append((problem.direct_answer, critic, max_tokens))
        return AuditResult(self.actions[critic], min(10, max_tokens), 20)


def problem():
    return Problem("test", "What is 2+2?", "2+2=3", "3", 5, {"arithmetic": {"expression": "2+2"}})


def test_rejection_returns_to_router_with_original_solution():
    router = Router()
    backend = Backend({"ledger": "REPLACE #### 99", "arithmetic": "REPLACE #### 4"})
    result = ActiveSuccession(["ledger", "arithmetic"], router, backend, RunConfig(stop_probability=0)).run(problem())
    assert result.answer == "4" and result.reason == "accepted_replacement"
    assert result.total_tokens == 25 and result.audit_tokens == 20 and result.prompt_tokens == 40
    assert result.trace[0]["verified"] is False
    assert [call[0] for call in backend.calls] == ["3", "3"]
    assert router.states[1][0][0]["action"] == "REPLACE"


def test_keep_stop_and_continuation_are_separate_from_rank():
    backend = Backend({"ledger": "KEEP", "arithmetic": "REPLACE #### 4"})
    result = ActiveSuccession(["ledger", "arithmetic"], Router(), backend, RunConfig(stop_probability=1)).run(problem())
    assert result.answer == "3" and result.reason == "keep_stop" and len(result.trace) == 1
    result = ActiveSuccession(["ledger", "arithmetic"], Router(), backend, RunConfig(stop_probability=0)).run(problem())
    assert result.answer == "4"


def test_budget_cap_and_critic_cap():
    backend = Backend({"ledger": "KEEP", "arithmetic": "KEEP"})
    result = ActiveSuccession(["ledger", "arithmetic"], Router(), backend,
                             RunConfig(token_budget=11, stop_probability=0)).run(problem())
    assert result.total_tokens == 11 and result.reason == "token_budget"
    assert backend.calls == [("3", "ledger", 6)]
    result = ActiveSuccession(["ledger", "arithmetic"], Router(), backend,
                             RunConfig(max_critics=1, stop_probability=0)).run(problem())
    assert result.reason == "critic_budget"
    with pytest.raises(ValueError):
        ActiveSuccession([], Router(), backend, RunConfig(token_budget=1)).run(problem())


def test_countermodel_and_invalid_action():
    backend = Backend({"ledger": "unparseable", "countermodel": "REPLACE #### 4"})
    result = ActiveSuccession(["ledger", "countermodel"], Router(), backend, RunConfig(stop_probability=1)).run(problem())
    assert result.answer == "4"
    assert result.trace[0]["action"] == "INVALID" and "stop_draw" not in result.trace[0]


def test_seeded_stopping_reproducibility():
    def sequence():
        engine = ActiveSuccession(["ledger", "arithmetic"], Router(), Backend({"ledger": "KEEP", "arithmetic": "KEEP"}),
                                  RunConfig(seed=42, stop_probability=.5))
        return [engine.run(problem()).to_dict() for _ in range(10)]
    assert sequence() == sequence()
