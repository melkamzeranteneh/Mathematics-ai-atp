from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Operation = dict[str, object]


class ActionTargetError(ValueError):
    """A structured action trace cannot be compiled without guessing."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class CompiledActionTarget:
    """Unambiguous prefix operations for one tactic's structured payload."""

    operations: tuple[Operation, ...]
    term_count: int
    syntax_argument_count: int

    @property
    def has_payload(self) -> bool:
        return self.term_count > 0 or self.syntax_argument_count > 0


def _tokenize_sexp(source: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(source):
        character = source[index]
        if character.isspace():
            index += 1
            continue
        if character in "()":
            tokens.append(character)
            index += 1
            continue
        if character == '"':
            start = index
            index += 1
            escaped = False
            while index < len(source):
                current = source[index]
                index += 1
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    break
            else:
                raise ActionTargetError(
                    "Unterminated string in action S-expression.",
                    code="invalid_sexp",
                )
            tokens.append(source[start:index])
            continue
        start = index
        while (
            index < len(source)
            and not source[index].isspace()
            and source[index] not in "()"
        ):
            index += 1
        tokens.append(source[start:index])
    return tokens


def _parse_sexp(source: str) -> object:
    tokens = _tokenize_sexp(source)
    position = 0

    def read() -> object:
        nonlocal position
        if position >= len(tokens):
            raise ActionTargetError(
                "Unexpected end of action S-expression.", code="invalid_sexp"
            )
        token = tokens[position]
        position += 1
        if token == "(":
            values: list[object] = []
            while position < len(tokens) and tokens[position] != ")":
                values.append(read())
            if position >= len(tokens):
                raise ActionTargetError(
                    "Unclosed action S-expression.", code="invalid_sexp"
                )
            position += 1
            return values
        if token == ")":
            raise ActionTargetError(
                "Unexpected ')' in action S-expression.", code="invalid_sexp"
            )
        return token

    parsed = read()
    if position != len(tokens):
        raise ActionTargetError(
            "Extra tokens after action S-expression.", code="invalid_sexp"
        )
    return parsed


def _expect_atom(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise ActionTargetError(
            f"Expected atom for {context}.", code="invalid_expression_shape"
        )
    return value


def _expect_shape(node: list[object], size: int, tag: str) -> None:
    if len(node) != size:
        raise ActionTargetError(
            f"{tag} expects {size - 1} fields, received {len(node) - 1}.",
            code="invalid_expression_shape",
        )


def _parse_nat(value: object, *, context: str) -> int:
    atom = _expect_atom(value, context=context)
    try:
        result = int(atom)
    except ValueError as exc:
        raise ActionTargetError(
            f"Expected natural number for {context}: {atom!r}.",
            code="invalid_expression_shape",
        ) from exc
    if result < 0:
        raise ActionTargetError(
            f"Expected natural number for {context}: {atom!r}.",
            code="invalid_expression_shape",
        )
    return result


def _compile_expression(
    parsed: object,
    *,
    local_indices: set[int],
    output: list[Operation],
) -> None:
    if not isinstance(parsed, list) or not parsed:
        raise ActionTargetError(
            "Action expression must be a non-empty list.",
            code="invalid_expression_shape",
        )
    tag = _expect_atom(parsed[0], context="expression tag")

    if tag == ":local":
        _expect_shape(parsed, 2, tag)
        reference = _expect_atom(parsed[1], context="local reference")
        if not reference.startswith("FV") or not reference[2:].isdigit():
            raise ActionTargetError(
                f"Unresolved local reference: {reference!r}.",
                code="unresolved_local_reference",
            )
        context_index = int(reference[2:])
        if context_index not in local_indices:
            raise ActionTargetError(
                f"Local reference FV{context_index} is absent from local_context.",
                code="unknown_local_reference",
            )
        output.append({"op": "LOCAL", "context_index": context_index})
        return

    if tag in {":global", ":ctor"}:
        _expect_shape(parsed, 2, tag)
        output.append(
            {
                "op": "GLOBAL" if tag == ":global" else "CONSTRUCTOR",
                "name": _expect_atom(parsed[1], context=f"{tag} name"),
            }
        )
        return

    if tag == ":app":
        if len(parsed) < 2:
            raise ActionTargetError(
                ":app requires a function expression.",
                code="invalid_expression_shape",
            )
        # The function is the first child; arity counts only explicit arguments.
        output.append({"op": "APP", "arity": len(parsed) - 2})
        for child in parsed[1:]:
            _compile_expression(child, local_indices=local_indices, output=output)
        return

    if tag == ":bound":
        _expect_shape(parsed, 2, tag)
        output.append(
            {"op": "BOUND", "index": _parse_nat(parsed[1], context="bound index")}
        )
        return

    if tag == ":metavar":
        _expect_shape(parsed, 1, tag)
        output.append({"op": "METAVAR"})
        return

    if tag == ":sort":
        _expect_shape(parsed, 2, tag)
        output.append(
            {"op": "SORT", "level": _expect_atom(parsed[1], context="sort level")}
        )
        return

    if tag == ":lit":
        _expect_shape(parsed, 2, tag)
        output.append(
            {"op": "LITERAL", "value": _expect_atom(parsed[1], context="literal")}
        )
        return

    if tag in {":lambda", ":forall"}:
        _expect_shape(parsed, 4, tag)
        output.append(
            {
                "op": "LAMBDA" if tag == ":lambda" else "FORALL",
                "binder_name": _expect_atom(parsed[1], context=f"{tag} binder"),
            }
        )
        _compile_expression(parsed[2], local_indices=local_indices, output=output)
        _compile_expression(parsed[3], local_indices=local_indices, output=output)
        return

    if tag == ":let":
        _expect_shape(parsed, 5, tag)
        output.append(
            {"op": "LET", "binder_name": _expect_atom(parsed[1], context="let binder")}
        )
        for child in parsed[2:]:
            _compile_expression(child, local_indices=local_indices, output=output)
        return

    if tag == ":proj":
        _expect_shape(parsed, 4, tag)
        output.append(
            {
                "op": "PROJECTION",
                "type_name": _expect_atom(parsed[1], context="projection type"),
                "index": _parse_nat(parsed[2], context="projection index"),
            }
        )
        _compile_expression(parsed[3], local_indices=local_indices, output=output)
        return

    raise ActionTargetError(
        f"Unsupported action expression tag: {tag!r}.",
        code="unsupported_expression_tag",
    )


def compile_action_trace(trace: dict[str, Any]) -> CompiledActionTarget:
    """Compile one validated action_trace_v2 payload into prefix operations."""
    terms = trace.get("terms")
    syntax_arguments = trace.get("syntax_args")
    local_context = trace.get("local_context")
    if not isinstance(terms, list):
        raise ActionTargetError("Trace terms must be a list.", code="invalid_trace")
    if not isinstance(syntax_arguments, list):
        raise ActionTargetError(
            "Trace syntax_args must be a list.", code="invalid_trace"
        )
    if not isinstance(local_context, list):
        raise ActionTargetError(
            "Trace local_context must be a list.", code="invalid_trace"
        )

    local_indices: set[int] = set()
    for variable in local_context:
        if not isinstance(variable, dict) or not isinstance(
            variable.get("context_index"), int
        ):
            raise ActionTargetError(
                "Every local_context entry requires an integer context_index.",
                code="invalid_local_context",
            )
        context_index = int(variable["context_index"])
        if context_index in local_indices:
            raise ActionTargetError(
                f"Duplicate local context index: {context_index}.",
                code="invalid_local_context",
            )
        local_indices.add(context_index)

    ordered_arguments: list[tuple[int, int, str, dict[str, object]]] = []
    for ordinal, term in enumerate(terms):
        if not isinstance(term, dict) or not isinstance(term.get("action_sexp"), str):
            raise ActionTargetError(
                "Every term requires an action_sexp string.", code="invalid_trace"
            )
        if not isinstance(term.get("source_start"), int):
            raise ActionTargetError(
                "Every term requires an integer source_start.", code="invalid_trace"
            )
        ordered_arguments.append((term["source_start"], ordinal, "term", term))

    for ordinal, argument in enumerate(syntax_arguments):
        if not isinstance(argument, dict) or not isinstance(argument.get("role"), str):
            raise ActionTargetError(
                "Every syntax argument requires a role.", code="invalid_trace"
            )
        if not isinstance(argument.get("source_start"), int):
            raise ActionTargetError(
                "Every syntax argument requires an integer source_start.",
                code="invalid_trace",
            )
        ordered_arguments.append(
            (argument["source_start"], len(terms) + ordinal, "syntax", argument)
        )

    operations: list[Operation] = []
    for _source_start, _ordinal, kind, argument in sorted(ordered_arguments):
        if kind == "term":
            operations.append({"op": "TERM_START"})
            _compile_expression(
                _parse_sexp(str(argument["action_sexp"])),
                local_indices=local_indices,
                output=operations,
            )
            operations.append({"op": "TERM_END"})
            continue

        role = argument["role"]
        if role != "fresh_name":
            raise ActionTargetError(
                f"Unsupported syntax argument role: {role!r}.",
                code="unsupported_syntax_role",
            )
        operations.append(
            {"op": "FRESH_NAME", "source": str(argument.get("source", ""))}
        )

    operations.append({"op": "STOP"})
    return CompiledActionTarget(
        operations=tuple(operations),
        term_count=len(terms),
        syntax_argument_count=len(syntax_arguments),
    )
