"""Active Niche Succession with typed adjudication and seeded stopping."""

from dataclasses import asdict, dataclass, field
import random

from .verification import TypedVerifier

STOP_PROBABILITIES = {"gsm8k": 0.20, "math": 0.32, "math-500": 0.32, "mbpp": 0.35}
OUTPUT_LIMITS = {"gsm8k": 1024, "math": 1024, "math-500": 1024, "mbpp": 768}


@dataclass(frozen=True)
class RunConfig:
    token_budget: int = 4096
    max_critics: int = 4
    output_limit: int = 1024
    stop_probability: float = 0.20
    seed: int = 42

    def __post_init__(self):
        if self.token_budget <= 0 or self.output_limit <= 0 or not 0 <= self.max_critics <= 5:
            raise ValueError("Invalid inference budget")
        if not 0 <= self.stop_probability <= 1:
            raise ValueError("Stop probability must lie in [0, 1]")


@dataclass
class RunResult:
    problem_id: str
    answer: str
    reason: str
    total_tokens: int
    audit_tokens: int
    prompt_tokens: int
    trace: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class ActiveSuccession:
    def __init__(self, community, router, backend, config=None, verifier=None):
        self.community = tuple(community)
        if len(set(self.community)) != len(self.community):
            raise ValueError("Community members must be unique")
        self.router, self.backend = router, backend
        self.config = config or RunConfig()
        self.verifier = verifier or TypedVerifier()
        self.random = random.Random(self.config.seed)

    def run(self, problem):
        config = self.config
        result = RunResult(problem.id, problem.direct_answer, "community_exhausted", problem.direct_tokens, 0, 0)
        if problem.direct_tokens > config.token_budget:
            raise ValueError("Direct solution already exceeds the total token budget")
        remaining = list(self.community)
        while remaining and len(result.trace) < config.max_critics:
            available = config.token_budget - result.total_tokens
            if available <= 0:
                result.reason = "token_budget"
                return result
            scores = self.router.scores(problem.text, result.trace,
                                        result.total_tokens / config.token_budget, remaining)
            critic = min(remaining, key=lambda name: (-scores[name], name))
            remaining.remove(critic)
            cap = min(config.output_limit, available)
            audit = self.backend.audit(problem, critic, cap)
            if audit.tokens > cap:
                raise ValueError("Audit exceeded its output-token cap")
            result.audit_tokens += audit.tokens
            result.total_tokens += audit.tokens
            result.prompt_tokens += audit.prompt_tokens
            action = audit.action
            event = {"critic": critic, "action": action.kind, "replacement": action.replacement,
                     "tokens": audit.tokens, "prompt_tokens": audit.prompt_tokens,
                     "score": scores[critic], "verified": None}
            result.trace.append(event)
            if action.kind == "REPLACE":
                event["verified"] = bool(self.verifier.verify(critic, problem, action.replacement))
                if event["verified"]:
                    result.answer, result.reason = action.replacement, "accepted_replacement"
                    return result
            elif action.kind == "KEEP":
                event["stop_draw"] = self.random.random()
                if event["stop_draw"] < config.stop_probability:
                    result.reason = "keep_stop"
                    return result
        if remaining:
            result.reason = "critic_budget"
        return result
