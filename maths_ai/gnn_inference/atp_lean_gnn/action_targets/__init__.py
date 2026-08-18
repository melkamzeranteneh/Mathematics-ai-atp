"""Lean-native structured targets for tactic argument generation."""

from .audit import TRACE_VERSIONS, ActionTargetAuditConfig, run_action_target_audit
from .compiler import ActionTargetError, CompiledActionTarget, compile_action_trace
from .source_syntax import CompiledSourceSyntaxTarget, compile_source_syntax_trace

__all__ = [
    "ActionTargetAuditConfig",
    "ActionTargetError",
    "CompiledActionTarget",
    "CompiledSourceSyntaxTarget",
    "TRACE_VERSIONS",
    "compile_action_trace",
    "compile_source_syntax_trace",
    "run_action_target_audit",
]
