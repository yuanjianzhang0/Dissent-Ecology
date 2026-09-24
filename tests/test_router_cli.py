from dataclasses import asdict
import json

import pytest

from dissent_ecology.cli import main, write_jsonl
from dissent_ecology.router import CatchRouter, RouterExample
from dissent_ecology.selection import CRITICS


def rows(prefix):
    return [RouterExample(f"{prefix}-{i}-{critic}", f"audit {critic} quantity {i % 3}", [], .1,
                          {name: name == critic for name in CRITICS})
            for i in range(4) for critic in CRITICS]


def test_router_roundtrip_and_probability(tmp_path):
    router = CatchRouter.fit(rows("fit"), rows("cal"))
    path = tmp_path / "router.json"
    router.save(path)
    loaded = CatchRouter.load(path)
    expected = router.scores("audit arithmetic quantity", [], .2, CRITICS)
    assert loaded.scores("audit arithmetic quantity", [], .2, CRITICS) == expected
    assert all(0 <= score <= 1 for score in expected.values())
    assert max(expected, key=expected.get) == "arithmetic"


def test_router_rejects_leakage_and_degenerate_calibration():
    with pytest.raises(ValueError, match="disjoint"):
        CatchRouter.fit(rows("fit"), rows("fit"))
    with pytest.raises(ValueError, match="both catch classes"):
        CatchRouter.fit(rows("fit"), [RouterExample("cal", "text", [], 0, dict.fromkeys(CRITICS, False))])


def test_cli_train_and_evaluate(tmp_path):
    training, calibration, model = [tmp_path / name for name in ("train.jsonl", "cal.jsonl", "router.json")]
    write_jsonl(training, [asdict(r) for r in rows("fit")])
    write_jsonl(calibration, [asdict(r) for r in rows("cal")])
    main(["train-router", "--training", str(training), "--calibration", str(calibration), "--output", str(model)])
    assert len(CatchRouter.load(model).heads) == 5
    predictions, references, report = [tmp_path / name for name in ("pred.jsonl", "ref.jsonl", "report.json")]
    write_jsonl(predictions, [{"problem_id": "x", "answer": "0.5", "total_tokens": 10}])
    write_jsonl(references, [{"id": "x", "reference": "1/2"}])
    main(["evaluate", "--input", str(predictions), "--references", str(references), "--output", str(report)])
    assert json.loads(report.read_text())["accuracy"] == 1
    write_jsonl(references, [{"id": "y", "reference": "1/2"}])
    with pytest.raises(ValueError, match="matching"):
        main(["evaluate", "--input", str(predictions), "--references", str(references), "--output", str(report)])


def test_cli_collect_select_and_run(tmp_path, monkeypatch):
    from dissent_ecology.backend import AuditResult

    calls = []

    class Backend:
        def audit(self, problem, critic, max_tokens):
            calls.append((problem.id, critic))
            return AuditResult("REPLACE #### 4" if critic == "arithmetic" else "KEEP", 5)

    monkeypatch.setattr("dissent_ecology.cli.make_backend", lambda args: Backend())
    dev, audits, community, router_path, test, predictions = [tmp_path / name for name in
        ("dev.jsonl", "audits.jsonl", "community.json", "router.json", "test.jsonl", "predictions.jsonl")]
    row = {"id": "dev-1", "text": "audit arithmetic quantity", "direct_solution": "2 + 2 = 3",
           "direct_answer": "3", "direct_tokens": 3, "reference": "4",
           "checks": {"arithmetic": {"expression": "2+2"}}}
    write_jsonl(dev, [row])
    main(["collect", "--input", str(dev), "--output", str(audits)])
    assert len(calls) == 5
    main(["select", "--input", str(audits), "--output", str(community)])
    assert json.loads(community.read_text())["critics"] == ["arithmetic"]
    CatchRouter.fit(rows("fit"), rows("cal")).save(router_path)
    write_jsonl(test, [{**row, "id": "test-1"}])
    args = ["run", "--input", str(test), "--community", str(community), "--router", str(router_path),
            "--output", str(predictions)]
    main(args)
    assert json.loads(predictions.read_text())["answer"] == "4"
    write_jsonl(test, [row])
    with pytest.raises(ValueError, match="overlap"):
        main(args)
