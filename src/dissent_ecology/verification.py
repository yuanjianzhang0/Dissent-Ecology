"""Typed checks over task-supplied facts; critic assertions are never facts."""

import ast
import operator

from .protocol import number

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def evaluate(expression, values):
    if not isinstance(expression, str) or len(expression) > 2048:
        raise ValueError("Invalid arithmetic expression")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression is too complex")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in values:
            return number(str(values[node.id]))
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return number(ast.get_source_segment(expression, node))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("Only numeric names and basic arithmetic are allowed")

    return visit(tree)


class TypedVerifier:
    def verify(self, critic, problem, replacement):
        try:
            answer = number(replacement)
            if critic == "countermodel":
                return True
            spec = problem.checks.get(critic)
            if not isinstance(spec, dict):
                return False
            values = spec.get("values", {})
            if "answer" in values:
                return False
            if critic == "arithmetic":
                return evaluate(spec["expression"], values) == answer
            if critic == "ratio_basis":
                numerator = evaluate(spec["numerator"], values)
                denominator = evaluate(spec["denominator"], values)
                multiplier = number(str(spec.get("multiplier", 1)))
                return denominator != 0 and numerator / denominator * multiplier == answer
            if critic == "ledger":
                balances = spec["balances"]
                return bool(balances) and all(
                    evaluate(b["opening"], values) + evaluate(b["incoming"], values)
                    - evaluate(b["outgoing"], values) == evaluate(b["closing"], values)
                    for b in balances
                ) and evaluate(spec["answer_expression"], values) == answer
            if critic == "constraint":
                constraints = spec["constraints"]
                # At least one equality must involve the proposed answer;
                # vacuous or unrelated constraints cannot validate a proposal.
                anchored = any(c["op"] == "eq" and any(
                    isinstance(node, ast.Name) and node.id == "answer"
                    for expr in (c["left"], c["right"]) for node in ast.walk(ast.parse(expr, mode="eval"))
                ) for c in constraints)
                comparison = {"eq": operator.eq, "lt": operator.lt, "le": operator.le,
                              "gt": operator.gt, "ge": operator.ge, "ne": operator.ne}
                bound = {**values, "answer": str(answer)}
                return anchored and all(comparison[c["op"]](evaluate(c["left"], bound), evaluate(c["right"], bound))
                                        for c in constraints)
            return False
        except (ValueError, KeyError, TypeError, ZeroDivisionError, SyntaxError, OverflowError, RecursionError):
            return False
