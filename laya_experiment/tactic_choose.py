"""Canonical tactic-family choices used by the ATP/Laya experiment.

The names mirror the tactic vocabulary and gloss table in ``new_idea.ipynb``.
They are kept as data so reranking and retraining use the same labels.
"""

from __future__ import annotations

TACTIC_NAMES: tuple[str, ...] = (
    ".", "C_simp", "abel", "abel_nf", "ac_rfl", "aesop", "aesop_cat", "aesop_mat",
    "all_goals", "any_goals", "apply", "apply_assumption", "apply_fun", "apply_mod_cast",
    "apply_rules", "assumption", "assumption'", "assumption_mod_cast", "beta_reduce",
    "bitwise_assoc_tac", "borelize", "by_cases", "by_contra", "by_contra!", "calc",
    "cancel_denoms", "case", "case'", "cases", "cases'", "cases_type", "casesm", "change",
    "choose", "choose!", "classical", "clear", "clear!", "clear_value", "coherence",
    "compute_degree", "compute_degree!", "congr", "congr!", "congrm", "constructor",
    "continuity", "contradiction", "contrapose", "contrapose!", "conv", "conv_lhs", "conv_rhs",
    "convert", "convert_to", "decide", "delta", "derivative_simp", "done", "dsimp", "erw",
    "eta_expand", "eval_simp", "exact", "exact_mod_cast", "exacts", "exfalso", "exists",
    "existsi", "ext", "ext1", "fapply", "fconstructor", "field_simp", "filter_upwards",
    "fin_cases", "first", "focus", "fun_prop", "funext", "gcongr", "generalize",
    "generalize_proofs", "ghost_calc", "ghost_fun_tac", "ghost_simp", "group", "have", "have'",
    "haveI", "if", "induction", "induction'", "infer_instance", "infer_param", "inhabit",
    "init_ring", "injection", "injections", "interval_cases", "intro", "intros", "introv",
    "isBoundedDefault", "iterate", "left", "let", "letI", "lift", "lift_lets", "linarith",
    "linear_combination", "map_fun_tac", "map_simp", "match", "matrix_simp", "measurability",
    "mem_tac", "mfld_set_tac", "mono", "move_mul", "mv_bisim", "next", "nlinarith", "nofun",
    "nomatch", "noncomm_ring", "nontriviality", "norm_cast", "norm_cast0", "norm_num",
    "norm_num1", "nth_rewrite", "nth_rw", "obtain", "omega", "on_goal", "open", "pderiv_simp",
    "peel", "pgame_wf_tac", "pi_lower_bound", "pi_upper_bound", "pick_goal", "positivity",
    "push_cast", "push_neg", "qify", "rcases", "rcongr", "refine", "refine'", "rel", "rename",
    "rename'", "rename_i", "repeat", "repeat'", "replace", "revert", "rewrite", "rfl", "rify",
    "right", "ring", "ring!", "ring1", "ring_nf", "rintro", "rotate_left", "rsuffices",
    "run_tac", "rw", "rw_mod_cast", "rwa", "set", "set!", "set_option", "show", "simp",
    "simp!", "simp?", "simp_all", "simp_all!", "simp_arith", "simp_intro", "simp_rw", "simp_wf",
    "simpa", "simpa!", "simpa?", "slice_lhs", "slice_rhs", "solve_by_elim", "specialize",
    "split", "split_ifs", "subst", "subst_hom_lift", "subst_vars", "substs", "suffices", "swap",
    "swap_var", "symm", "sz_positivity", "tauto", "tfae_finish", "tfae_have", "to_encard_tac",
    "trans", "transfer", "transfer_rw", "transitivity", "trivial", "try", "unfold", "unfold_let",
    "unfold_projs", "unit_interval", "use", "use!", "valid", "with_unfolding_all",
    "witt_truncateFun_tac", "wlog", "zify",
)

TACTIC_SET = frozenset(TACTIC_NAMES)


def validate_tactic_names(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return unique tactic names and reject labels outside the canonical catalog."""
    unknown = sorted(set(names) - TACTIC_SET)
    if unknown:
        raise ValueError(f"Unknown tactic families: {', '.join(unknown)}")
    return tuple(dict.fromkeys(names))
