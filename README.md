# Dissent Ecology

Select complementary critics and route their audits under an inference budget.

Dissent Ecology measures which errors each critic repairs, selects a compact community, and invokes its members according to calibrated catch probabilities. A proposed replacement passes a typed check before acceptance. KEEP can terminate the process or hand control back to the router.

## Install

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
python examples/offline_demo.py
```

The offline example trains a router on synthetic fixtures, selects a community, and corrects an arithmetic answer with a scripted backend. It requires no model download, API key, or GPU. Its output is a functional demonstration, not a benchmark result.

## Community selection

Five default policies audit entity ledgers, ratio bases, constraints, arithmetic, and independent countermodels. Each policy independently receives the same original problem and direct solution.

Development observations record a correct repair of an incorrect direct answer (`catch`), an incorrect replacement of a correct answer (`harm`), or an incorrect replacement of an already incorrect answer (`futile`). KEEP and malformed actions earn no repair credit. Token usage is charged for every audit, including malformed output.

For a candidate community `S`, selection maximizes:

```text
J(S) = |repair union| - lambda * |harm union| - mu * |futile union|
       - tau * sum(audit generated tokens) / 1000

lambda = max(4, correct direct answers / incorrect direct answers)
mu = 0.25
tau = 0.01
```

A problem can belong to both the repair union and the futile union when different critics propose different answers. Overlapping catches divide one unit of credit equally. A critic's removal utility is `J(S) - J(S without critic)`; members with nonpositive removal utility are removed. Shared credit is a diagnostic and does not replace union coverage in the objective.

The default cap is four critics, giving 31 candidate subsets including the empty set. Ties prefer smaller communities and then lexical order. With no direct errors, selection uses infinite harm aversion and returns an empty community. Set `--max-critics 5` to include the full candidate set.

## Development workflow

Supply JSONL problems with a precomputed direct solution. Direct solutions and all audits should use the same checkpoint, greedy decoding, and configured output cap. A local chat-completions endpoint is supported; the default model identifier is `Qwen/Qwen3-8B`.

```json
{"id":"dev-1","text":"What is 7 plus 5?","direct_solution":"7 + 5 = 11","direct_answer":"11","direct_tokens":10,"reference":"12"}
```

```bash
mkdir -p artifacts
dissent-ecology collect --input development.jsonl --output artifacts/audits.jsonl
dissent-ecology select --input artifacts/audits.jsonl --output artifacts/community.json
```

Use `--base-url` and `--model` to select an endpoint. `DISSENT_API_KEY` supplies authentication when needed. Additional server options can be supplied through `--request-options options.json`. Required decoding fields cannot be overridden. The server must return actual completion and prompt token usage; token counts are never estimated from text length.

The collection file contains both the outcome fields used for selection and each raw response. Audit outcomes use the reference answer only during development. Task-supplied verification facts and reference labels are not sent to the critic.

## Router fitting

Create two JSONL files with disjoint problem IDs: one for fitting and one for calibration. A row represents the state before the next invocation:

```json
{"problem_id":"fit-1","text":"Find the percentage discount.","history":[{"critic":"ledger","action":"KEEP","verified":null}],"consumed_ratio":0.2,"catches":{"ledger":false,"ratio_basis":true,"constraint":false,"arithmetic":false,"countermodel":false}}
```

Native-catch targets come from independent development audits. History contains only actions already observed in that state, never future critic outcomes or reference answers. The consumed ratio is generated tokens used so far divided by the total generated-token budget. Multiple states for one problem must stay in the same partition. Initial states use an empty history; fitting only initial states will not teach history effects.

```bash
dissent-ecology train-router \
  --training router-fit.jsonl \
  --calibration router-calibration.jsonl \
  --output artifacts/router.json
```

Each of five logistic heads uses word counts, error-type lexical indicators, prior actions and verification outcomes, and consumed-budget ratio. A second logistic model fits Platt scaling on held-out raw scores. Each critic needs positive and negative examples in both partitions; insufficient data raises an error. The feature vocabulary is fit exclusively on fitting examples. Router artifacts store ordinary JSON coefficients and development IDs, without executable serialization.

## Active inference

Add trusted verification specifications to inference problems. For example:

```json
{"id":"test-1","text":"What is 7 plus 5?","direct_solution":"7 + 5 = 11","direct_answer":"11","direct_tokens":10,"checks":{"arithmetic":{"values":{"a":7,"b":5},"expression":"a + b"}}}
```

```bash
dissent-ecology run \
  --input test.jsonl \
  --community artifacts/community.json \
  --router artifacts/router.json \
  --task gsm8k \
  --token-budget 4096 \
  --output artifacts/predictions.jsonl

dissent-ecology evaluate \
  --input artifacts/predictions.jsonl \
  --references references.jsonl \
  --output artifacts/metrics.json
```

Reference rows have `id` and `reference` fields. Evaluation requires exactly matching unique IDs. Inference rejects overlap with known selection, fitting, or calibration IDs. Maintain globally consistent dataset IDs to make these checks effective.

At each step the unused member with the highest calibrated score is called. Scores rank critics; they are not stopping thresholds. A KEEP stops with the configured probability. A failed replacement check or malformed action returns to routing. An accepted replacement terminates immediately. When the community or budget is exhausted, the original answer is retained. Every critic sees the original solution, even after another critic proposes a rejected change.

| Task preset | KEEP stop probability | Output cap |
| --- | ---: | ---: |
| GSM8K | 0.20 | 1,024 |
| MATH / MATH-500 | 0.32 | 1,024 |
| MBPP configuration constants | 0.35 | 768 |

The numerical CLI supports GSM8K and numerical MATH answers. Symbolic equivalence, code actions, code execution, and task-specific code policies require adapters. MBPP constants are provided for such adapters; the numerical CLI does not claim code-generation evaluation support.

The default stopping seed is 42; 13 and 20260903 can also be supplied. One random stream advances across problems in input order. A fresh engine with the same seed and order gives identical stopping draws.

## Typed verification

Checks use exact rational arithmetic and a restricted expression evaluator. Supported operations are `+`, `-`, `*`, `/`, parentheses, and named numeric values. Functions, attribute access, exponentiation, and arbitrary Python execution are rejected.

| Policy | Required check fields |
| --- | --- |
| `arithmetic` | `values`, `expression` matching the replacement |
| `ratio_basis` | `values`, `numerator`, nonzero `denominator`, optional `multiplier` |
| `ledger` | `values`, nonempty `balances` with `opening`, `incoming`, `outgoing`, `closing`, plus `answer_expression` |
| `constraint` | `values`, nonempty `constraints` with `left`, `op`, `right`; at least one equality references `answer` |
| `countermodel` | Direct acceptance of a valid numeric replacement |

Comparison operators are `eq`, `ne`, `lt`, `le`, `gt`, and `ge`. Ledger checks require `opening + incoming - outgoing == closing` for every balance. Ratio checks recompute `numerator / denominator * multiplier`. Constraint checks establish satisfaction of supplied constraints, not uniqueness of a solution.

The task adapter must derive these facts from the problem, independently of the proposed answer. Missing or malformed checks reject the replacement. The built-in verifier does not extract semantics from arbitrary natural-language problems. Countermodel follows an explicit direct-acceptance path, so it provides no deterministic correctness guarantee.

## Token accounting and integration

The budget counts generated tokens, including the supplied direct solution's generated tokens. Each invocation is capped at the smaller of the output limit and remaining budget. Audit prompt tokens are recorded separately and excluded from this budget and the selection token penalty. For billing or total-context budgets, provide a different accounting adapter. The default total cap of 4,096 is configurable.

`ActiveSuccession` accepts a router with `scores(text, history, consumed_ratio, candidates)`, a backend with `audit(problem, critic, max_tokens)`, and a verifier with `verify(critic, problem, replacement)`. Backend results expose `tokens`, `prompt_tokens`, and an `Action` through `.action`. Custom code backends may return string-valued replacements and supply an external execution verifier. The numerical parser remains strict about the `KEEP` / `REPLACE #### <number>` boundary.

No pretrained router, development traces, or benchmark measurements are bundled. Train and freeze the selection and router with your own development data before evaluation. The tests cover union accounting, exact selection, leakage checks, serialization, action parsing, all typed checks, rejection paths, stopping, token caps, and endpoint request behavior.
