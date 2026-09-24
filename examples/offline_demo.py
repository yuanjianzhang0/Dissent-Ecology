"""Run an illustrative, deterministic end-to-end example without a model server."""

import json
from pathlib import Path
import tempfile

from dissent_ecology.backend import AuditResult
from dissent_ecology.engine import ActiveSuccession, RunConfig
from dissent_ecology.protocol import Problem
from dissent_ecology.router import CatchRouter, RouterExample
from dissent_ecology.selection import CRITICS, Observation, select_community


def development_rows(prefix, repeats):
    examples, observations = [], []
    for i in range(repeats):
        for index, niche in enumerate(CRITICS):
            pid = f"{prefix}-{i}-{niche}"
            text = {
                "ledger": "How many remain after quantities were bought, sold, and received?",
                "ratio_basis": "Calculate the percentage discount using the original price as the base.",
                "constraint": "Find the positive integer satisfying exactly the stated bound.",
                "arithmetic": "Calculate the sum and product of these numbers.",
                "countermodel": "Is an alternative possible if the assumption changes?",
            }[niche]
            catches = {critic: critic == niche for critic in CRITICS}
            examples.append(RouterExample(pid, text, [], 0.1, catches))
            if i % 2:
                previous = CRITICS[(index + 1) % len(CRITICS)]
                examples.append(RouterExample(pid, text, [{"critic": previous, "action": "KEEP"}], 0.4, catches))
            for critic in CRITICS:
                # Only three policies repair this toy development population.
                observations.append(Observation(pid, critic, False,
                    critic == niche and critic in {"ledger", "ratio_basis", "arithmetic"},
                    False, False, 20))
    return examples, observations


class DemoBackend:
    def audit(self, problem, critic, max_tokens):
        text = "REPLACE #### 12" if critic == "arithmetic" else "KEEP"
        return AuditResult(text, min(8, max_tokens))


def main():
    training, observations = development_rows("fit", 8)
    calibration, _ = development_rows("calibration", 4)
    community = select_community(observations)
    router = CatchRouter.fit(training, calibration)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "router.json"
        router.save(path)
        router = CatchRouter.load(path)
    problem = Problem("demo-test", "Calculate the sum of 7 and 5.", "7 + 5 = 11", "11", 10,
                      {"arithmetic": {"values": {"a": 7, "b": 5}, "expression": "a + b"}})
    result = ActiveSuccession(community.critics, router, DemoBackend(),
                             RunConfig(token_budget=100, stop_probability=0)).run(problem)
    assert result.answer == "12" and result.reason == "accepted_replacement"
    print(json.dumps({"example_type": "synthetic_functional_demo", "community": community.to_dict(),
                      "result": result.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
