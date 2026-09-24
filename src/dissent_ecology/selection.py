"""Exact selection using union coverage and asymmetric intervention costs."""

from dataclasses import asdict, dataclass
from itertools import combinations
import math

from .protocol import Action, numeric_equal

CRITICS = ("ledger", "ratio_basis", "constraint", "arithmetic", "countermodel")


@dataclass(frozen=True)
class Observation:
    problem_id: str
    critic: str
    direct_correct: bool
    catch: bool
    harm: bool
    futile: bool
    tokens: int

    def __post_init__(self):
        flags = (self.direct_correct, self.catch, self.harm, self.futile)
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("Outcome flags must be booleans")
        if type(self.tokens) is not int or self.tokens < 0:
            raise ValueError("Tokens must be nonnegative integers")
        if sum((self.catch, self.harm, self.futile)) > 1:
            raise ValueError("An individual audit has at most one outcome")
        if (self.direct_correct and (self.catch or self.futile)) or (not self.direct_correct and self.harm):
            raise ValueError("Outcome contradicts direct correctness")

    @classmethod
    def from_action(cls, problem_id, critic, direct_answer, reference, action: Action, tokens, equal=numeric_equal):
        direct = equal(direct_answer, reference)
        replaces = action.kind == "REPLACE"
        correct = replaces and equal(action.replacement, reference)
        return cls(problem_id, critic, direct, not direct and correct,
                   direct and replaces and not correct,
                   not direct and replaces and not correct, tokens)


@dataclass(frozen=True)
class Selection:
    critics: tuple[str, ...]
    utility: float
    harm_weight: float
    repair_count: int
    harm_count: int
    futile_count: int
    tokens: int
    shared_credit: dict[str, float]
    niches: dict[str, float]
    evaluated_subsets: int

    def to_dict(self):
        return asdict(self)


class CommunityObjective:
    def __init__(self, observations, harm_weight=None, mu=0.25, tau=0.01):
        self.rows = tuple(observations)
        if not self.rows:
            raise ValueError("Development observations are empty")
        self.critics = tuple(sorted({row.critic for row in self.rows}))
        self.problems = {row.problem_id for row in self.rows}
        pairs = {(row.problem_id, row.critic) for row in self.rows}
        if len(pairs) != len(self.rows) or len(pairs) != len(self.critics) * len(self.problems):
            raise ValueError("Expected one independent audit per problem and critic")
        direct = {}
        for row in self.rows:
            if row.problem_id in direct and direct[row.problem_id] != row.direct_correct:
                raise ValueError("Inconsistent direct correctness")
            direct[row.problem_id] = row.direct_correct
        errors = sum(not correct for correct in direct.values())
        # With no direct errors, no critic can earn repair reward. Infinite harm
        # aversion makes the empty community optimal without a division by zero.
        self.harm_weight = (max(4.0, (len(direct) - errors) / errors)
                            if errors else math.inf) if harm_weight is None else harm_weight
        for value in (mu, tau):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Penalties must be finite and nonnegative")
        if math.isnan(self.harm_weight) or self.harm_weight < 0:
            raise ValueError("Harm weight must be nonnegative")
        self.mu, self.tau = mu, tau

    def stats(self, community):
        rows = [row for row in self.rows if row.critic in community]
        return (len({r.problem_id for r in rows if r.catch}),
                len({r.problem_id for r in rows if r.harm}),
                len({r.problem_id for r in rows if r.futile}),
                sum(r.tokens for r in rows))

    def utility(self, community):
        repair, harm, futile, tokens = self.stats(community)
        return repair - (self.harm_weight * harm if harm else 0) - self.mu * futile - self.tau * tokens / 1000


def select_community(observations, max_critics=4, harm_weight=None, mu=0.25, tau=0.01):
    if not 0 <= max_critics <= 5:
        raise ValueError("max_critics must be between zero and five")
    objective = CommunityObjective(observations, harm_weight, mu, tau)
    if len(objective.critics) > 5:
        raise ValueError("Exact selection supports up to five candidate policies")
    candidates = [s for n in range(min(max_critics, len(objective.critics)) + 1)
                  for s in combinations(objective.critics, n)]
    # Prefer fewer critics, then lexical order when utilities tie.
    best = min(candidates, key=lambda s: (-objective.utility(s), len(s), s))
    while best:
        removable = [a for a in best if objective.utility(best) - objective.utility(tuple(b for b in best if b != a)) <= 0]
        if not removable:
            break
        best = tuple(a for a in best if a != removable[0])
    credit = dict.fromkeys(best, 0.0)
    for problem in objective.problems:
        successful = [r.critic for r in objective.rows if r.problem_id == problem and r.critic in best and r.catch]
        for critic in successful:
            credit[critic] += 1 / len(successful)
    niches = {a: objective.utility(best) - objective.utility(tuple(b for b in best if b != a)) for a in best}
    return Selection(best, objective.utility(best), objective.harm_weight,
                     *objective.stats(best), credit, niches, len(candidates))
