from itertools import combinations

import pytest

from dissent_ecology.protocol import Action
from dissent_ecology.selection import CommunityObjective, Observation, select_community


def test_union_overlap_credit_and_mixed_futile():
    rows = [Observation("p1", "a", False, True, False, False, 10),
            Observation("p1", "b", False, True, False, False, 10),
            Observation("p2", "a", False, True, False, False, 10),
            Observation("p2", "b", False, False, False, True, 10),
            Observation("p3", "a", False, False, False, False, 10),
            Observation("p3", "b", False, True, False, False, 10)]
    result = select_community(rows)
    assert result.critics == ("a", "b")
    assert result.repair_count == 3 and result.futile_count == 1
    assert result.utility == pytest.approx(3 - .25 - .01 * 60 / 1000)
    assert result.shared_credit == {"a": 1.5, "b": 1.5}
    assert sum(result.shared_credit.values()) == result.repair_count
    assert all(value > 0 for value in result.niches.values())


def test_harm_prior_and_exact_search():
    rows = [Observation(str(i), c, i > 0, i == 0, i == 1 and c == "b", False, 5)
            for i in range(11) for c in ("a", "b", "c", "d", "e")]
    result = select_community(rows)
    assert result.harm_weight == 10
    assert result.critics == ("a",)
    assert result.evaluated_subsets == 31
    objective = CommunityObjective(rows)
    assert result.utility == max(objective.utility(s) for n in range(5) for s in combinations(objective.critics, n))


def test_no_errors_and_zero_cost_redundancy():
    result = select_community([Observation("p", "a", True, False, True, False, 10)])
    assert result.critics == () and result.utility == 0
    result = select_community([Observation("p", c, False, True, False, False, 0) for c in ("a", "b")])
    assert result.critics == ("a",)


def test_invalid_matrix_and_counts():
    a = Observation("p", "a", False, True, False, False, 10)
    with pytest.raises(ValueError):
        select_community([a, a])
    with pytest.raises(ValueError):
        select_community([a, Observation("q", "b", False, False, False, False, 10)])
    with pytest.raises(ValueError):
        Observation("p", "a", True, True, False, False, 1)
    with pytest.raises(ValueError):
        Observation("p", "a", False, False, False, False, -1)


@pytest.mark.parametrize("direct, replacement, expected", [
    ("3", "4", (True, False, False)),
    ("4", "3", (False, True, False)),
    ("3", "5", (False, False, True)),
    ("4", "4", (False, False, False)),
])
def test_outcome_construction(direct, replacement, expected):
    row = Observation.from_action("p", "a", direct, "4", Action("REPLACE", replacement), 8)
    assert (row.catch, row.harm, row.futile) == expected
