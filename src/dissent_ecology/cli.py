"""File-oriented development and inference commands."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from .backend import ChatBackend
from .engine import ActiveSuccession, OUTPUT_LIMITS, STOP_PROBABILITIES, RunConfig
from .protocol import Problem, numeric_equal
from .router import CatchRouter, RouterExample
from .selection import CRITICS, Observation, select_community


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_backend(args):
    extra = json.loads(Path(args.request_options).read_text()) if args.request_options else None
    return ChatBackend(args.base_url, args.model, extra_body=extra)


def make_problem(row):
    return Problem(**{key: row[key] for key in Problem.__dataclass_fields__ if key in row})


def collect(args):
    backend = make_backend(args)
    rows = read_jsonl(args.input)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Problem IDs must be unique")
    with Path(args.output).open("w", encoding="utf-8") as stream:
        for row in rows:
            problem = make_problem(row)
            for critic in CRITICS:
                audit = backend.audit(problem, critic, args.output_limit)
                observation = Observation.from_action(problem.id, critic, problem.direct_answer,
                                                      row["reference"], audit.action, audit.tokens)
                stream.write(json.dumps({**asdict(observation), "action": asdict(audit.action),
                    "raw_response": audit.text, "prompt_tokens": audit.prompt_tokens}) + "\n")
                stream.flush()


def select(args):
    observations = [Observation(**{k: row[k] for k in Observation.__dataclass_fields__}) for row in read_jsonl(args.input)]
    result = select_community(observations, args.max_critics, args.harm_weight, args.mu, args.tau).to_dict()
    if math.isinf(result["harm_weight"]):
        result["harm_weight"] = "infinity"
    result.update({"mu": args.mu, "tau": args.tau, "max_critics": args.max_critics,
                   "development_ids": sorted({r.problem_id for r in observations})})
    write_json(args.output, result)


def train(args):
    training = [RouterExample(**row) for row in read_jsonl(args.training)]
    calibration = [RouterExample(**row) for row in read_jsonl(args.calibration)]
    CatchRouter.fit(training, calibration, seed=args.seed).save(args.output)


def run(args):
    selection = json.loads(Path(args.community).read_text())
    router = CatchRouter.load(args.router)
    rows = read_jsonl(args.input)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Problem IDs must be unique")
    development = set(router.fit_ids) | set(router.calibration_ids) | set(selection.get("development_ids", []))
    if development & set(ids):
        raise ValueError("Inference IDs overlap development data; use disjoint frozen evaluation IDs")
    config = RunConfig(args.token_budget, args.max_critics, args.output_limit or OUTPUT_LIMITS[args.task],
                       STOP_PROBABILITIES[args.task] if args.stop_probability is None else args.stop_probability,
                       args.seed)
    engine = ActiveSuccession(selection["critics"], router, make_backend(args), config)
    # Stream completed cases so an interrupted endpoint does not discard prior work.
    with Path(args.output).open("w", encoding="utf-8") as stream:
        for row in rows:
            result = engine.run(make_problem(row)).to_dict()
            result["config"] = asdict(config)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()


def evaluate(args):
    references = read_jsonl(args.references)
    results = read_jsonl(args.input)
    ref = {row["id"]: row["reference"] for row in references}
    ids = [row["problem_id"] for row in results]
    if not results or len(ref) != len(references) or len(ids) != len(set(ids)) or set(ids) != set(ref):
        raise ValueError("Expected nonempty, unique, exactly matching prediction and reference IDs")
    correct = sum(numeric_equal(row["answer"], ref[row["problem_id"]]) for row in results)
    write_json(args.output, {"examples": len(results), "correct": correct,
                            "accuracy": correct / len(results),
                            "mean_generated_tokens": sum(row["total_tokens"] for row in results) / len(results)})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select and route complementary critics under token budgets.")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("collect", help="Collect independent development audits")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--output-limit", type=int, default=1024)
    p.set_defaults(handler=collect)
    q = commands.add_parser("select", help="Select an exact critic community")
    q.add_argument("--input", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--max-critics", type=int, default=4)
    q.add_argument("--harm-weight", type=float)
    q.add_argument("--mu", type=float, default=0.25)
    q.add_argument("--tau", type=float, default=0.01)
    q.set_defaults(handler=select)
    q = commands.add_parser("train-router", help="Fit catch heads and held-out Platt scaling")
    q.add_argument("--training", required=True)
    q.add_argument("--calibration", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--seed", type=int, default=42)
    q.set_defaults(handler=train)
    r = commands.add_parser("run", help="Run a frozen community and router")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--community", required=True)
    r.add_argument("--router", required=True)
    r.add_argument("--task", choices=("gsm8k", "math", "math-500"), default="gsm8k")
    r.add_argument("--token-budget", type=int, default=4096)
    r.add_argument("--max-critics", type=int, default=4)
    r.add_argument("--output-limit", type=int)
    r.add_argument("--stop-probability", type=float)
    r.add_argument("--seed", type=int, default=42)
    r.set_defaults(handler=run)
    for q in (p, r):
        q.add_argument("--base-url", default="http://localhost:8000/v1")
        q.add_argument("--model", default="Qwen/Qwen3-8B")
        q.add_argument("--request-options", help="JSON file of additional endpoint request options")
    q = commands.add_parser("evaluate", help="Evaluate exact numeric answers")
    q.add_argument("--input", required=True)
    q.add_argument("--references", required=True)
    q.add_argument("--output", required=True)
    q.set_defaults(handler=evaluate)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
