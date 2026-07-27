"""Comprehensive tests for the S-expression → DAG parser."""
import sys
sys.path.insert(0, ".")

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.graph import (
    BINDER_KIND_FORALL,
    BINDER_KIND_LAMBDA,
    BINDER_KIND_LET,
    BINDER_KIND_NONE,
    DAGBuilder,
    get_node_labels,
    proof_state_to_dag,
    sexp_to_dag,
)


# ---------------------------------------------------------------------------
# S-expression → DAG conversion
# ---------------------------------------------------------------------------

def test_simple_forall():
    dag = sexp_to_dag("(:forall q (:sort 0) 0)")
    labels = get_node_labels(dag)
    assert ":forall" in labels
    assert "Prop" in labels
    assert "q" in labels


def test_nested_forall():
    dag = sexp_to_dag("(:forall p (:sort 0) (:forall q (:sort 0) ((:c Or) 1 0)))")
    labels = get_node_labels(dag)
    assert labels.count(":forall") == 2
    assert "Or" in labels
    assert "p" in labels
    assert "q" in labels


def test_nested_forall_resolves_exact_debruijn_argument_order():
    dag = sexp_to_dag(
        "(:forall p (:sort 0) (:forall q (:sort 0) ((:c Or) 1 0)))"
    )
    p_node = next(node for node in dag.nodes if node.label == "p" and node.is_bound)
    q_node = next(node for node in dag.nodes if node.label == "q" and node.is_bound)
    application = next(node for node in dag.nodes if node.label == "App")

    assert [dag.nodes[node_id].label for node_id in application.children] == [
        "Or",
        "p",
        "q",
    ]
    assert application.children[1:] == (p_node.id, q_node.id)


def test_arrow():
    dag = sexp_to_dag("(:forall a (:c Nat) (:forall a (:c Nat) (:c Nat)))")
    labels = get_node_labels(dag)
    assert labels.count(":forall") == 2
    assert "Nat" in labels


def test_same_named_binders_have_distinct_scoped_nodes():
    dag = sexp_to_dag(
        "(:forall a (:sort 0) (:forall a (:sort 0) ((:c Eq) 1 0)))"
    )
    binders = [node for node in dag.nodes if node.label == "a" and node.is_bound]
    application = next(node for node in dag.nodes if node.label == "App")

    assert len(binders) == 2
    assert binders[0].id != binders[1].id
    assert [node.binder_depth for node in binders] == [1, 2]
    assert application.children[1:] == (binders[0].id, binders[1].id)


def test_constants():
    dag = sexp_to_dag("((:c Eq) (:c Nat) 0 0)")
    labels = get_node_labels(dag)
    assert "Eq" in labels
    assert "Nat" in labels


def test_lambda():
    dag = sexp_to_dag("(:lambda a 3 ((:c And) (3 0) ((:c Not) (2 0))))")
    labels = get_node_labels(dag)
    assert ":lambda" in labels
    assert "And" in labels
    assert "Not" in labels


@pytest.mark.parametrize("binder_tag", [":forall", ":lambda"])
def test_binder_type_uses_old_context_and_body_uses_new_context(binder_tag):
    dag = sexp_to_dag(
        f"(:forall alpha (:sort 1) "
        f"({binder_tag} x 0 ((:c Eq) 1 0)))"
    )
    alpha = next(node for node in dag.nodes if node.label == "alpha" and node.is_bound)
    x = next(node for node in dag.nodes if node.label == "x" and node.is_bound)
    inner_binder = next(
        node
        for node in dag.nodes
        if node.label == binder_tag and node.children[0] == x.id
    )
    application = next(node for node in dag.nodes if node.label == "App")

    assert inner_binder.children[1] == alpha.id
    assert application.children[1:] == (alpha.id, x.id)


def test_let_binder_scope_and_children_are_exact():
    dag = sexp_to_dag(
        "(:forall x (:c Nat) (:let x 0 0 ((:c Eq) (:c Nat) 1 0)))"
    )
    outer_x, inner_x = [
        node for node in dag.nodes if node.label == "x" and node.is_bound
    ]
    let_node = next(node for node in dag.nodes if node.label == ":let")
    body = dag.nodes[let_node.children[3]]

    assert let_node.label == ":let"
    assert let_node.node_type == "sbinder"
    assert inner_x.binder_kind == BINDER_KIND_LET
    assert let_node.children[:3] == (inner_x.id, outer_x.id, outer_x.id)
    assert body.label == "App"
    assert body.children[-2:] == (outer_x.id, inner_x.id)


def test_complex_expression():
    sexp = "(:forall U (:sort 1) (:forall P (:forall a 0 (:sort 0)) (:forall Q (:forall a 1 (:sort 0)) (:forall a (:forall x 2 (:forall a (2 0) (2 1))) ((:c Not) ((:c Exists) 3 (:lambda a 3 ((:c And) (3 0) ((:c Not) (2 0))))))))))"
    dag = sexp_to_dag(sexp)
    labels = get_node_labels(dag)
    assert "Not" in labels
    assert "Exists" in labels
    assert "And" in labels
    assert dag.num_nodes > 5
    assert dag.num_edges > 0


def test_hash_consing():
    dag = sexp_to_dag("((:c Or) 1 0)")
    dag2 = sexp_to_dag("((:c Or) 1 0)")
    assert dag.num_nodes == dag2.num_nodes


def test_node_count():
    dag = sexp_to_dag("(:forall n (:c Nat) ((:c Eq) (:c Nat) 0 0))")
    assert dag.num_nodes == 5


def test_node_count_no_sharing():
    dag = sexp_to_dag("(:forall n (:c Nat) ((:c Eq) (:c Nat) (:c Bool) 0))")
    assert dag.num_nodes == 6


# ---------------------------------------------------------------------------
# Binder annotations
# ---------------------------------------------------------------------------

def test_forall_binder_annotation():
    dag = sexp_to_dag("(:forall q (:sort 0) 0)")
    bound = [n for n in dag.nodes if n.is_bound == 1]
    assert len(bound) >= 1
    assert bound[0].binder_kind == BINDER_KIND_FORALL
    assert bound[0].binder_depth == 1


def test_nested_binder_depth():
    dag = sexp_to_dag("(:forall p (:sort 0) (:forall q (:sort 0) ((:c Or) 1 0)))")
    bound = [n for n in dag.nodes if n.is_bound == 1]
    depths = {n.label: n.binder_depth for n in bound}
    assert depths.get("p") == 1
    assert depths.get("q") == 2


def test_lambda_binder_annotation():
    dag = sexp_to_dag("(:lambda a 3 ((:c And) (3 0) ((:c Not) (2 0))))")
    bound = [n for n in dag.nodes if n.is_bound == 1]
    assert any(n.binder_kind == BINDER_KIND_LAMBDA for n in bound)


def test_context_variables_not_bound():
    dag = sexp_to_dag("((:c Or) ((:c Nat)) ((:c Bool)))")
    for n in dag.nodes:
        assert n.is_bound == BINDER_KIND_NONE


# ---------------------------------------------------------------------------
# De Bruijn index resolution
# ---------------------------------------------------------------------------

def test_debruijn_resolves_to_binder():
    dag = sexp_to_dag("(:forall n (:c Nat) ((:c Eq) (:c Nat) 0 0))")
    labels = get_node_labels(dag)
    # de Bruijn 0 should resolve to "n", not appear as a number
    assert "0" not in labels


def test_debruijn_multiple_indices():
    dag = sexp_to_dag("(:forall p (:sort 0) (:forall q (:sort 0) ((:c Or) 1 0)))")
    labels = get_node_labels(dag)
    assert "0" not in labels
    assert "1" not in labels
    assert "p" in labels
    assert "q" in labels


# ---------------------------------------------------------------------------
# Free variables and sorts
# ---------------------------------------------------------------------------

def test_free_variable():
    dag = sexp_to_dag("(:fv _uniq.28)")
    labels = get_node_labels(dag)
    assert "_uniq.28" in labels


def test_special_forms_preserve_semantic_node_types():
    cases = {
        "(:c Nat)": ("Nat", "const"),
        "(:fv _uniq.28)": ("_uniq.28", "var"),
        "(:sort 0)": ("Prop", "type"),
        "(:lit 42)": ("Lit:42", "const"),
        "(:lambda x (:c Nat) 0)": (":lambda", "sbinder"),
    }

    for sexp, (expected_label, expected_type) in cases.items():
        dag = sexp_to_dag(sexp)
        root = dag.nodes[dag.expression_root_id]
        assert (root.label, root.node_type) == (expected_label, expected_type)


def test_metadata_form_retains_metadata_and_expression_children():
    dag = sexp_to_dag("(:mdata (key value) (:c Nat))")
    root = dag.nodes[dag.expression_root_id]

    assert root.label == ":mdata"
    assert root.node_type == "meta"
    assert [dag.nodes[node_id].label for node_id in root.children] == [
        "Metadata:(key value)",
        "Nat",
    ]


def test_projection_form_retains_structure_index_and_expression():
    dag = sexp_to_dag("(:proj Prod 1 (:fv pair))")
    root = dag.nodes[dag.expression_root_id]

    assert root.label == ":proj"
    assert root.node_type == "sapp"
    assert [dag.nodes[node_id].label for node_id in root.children] == [
        "Prod",
        "Field:1",
        "pair",
    ]


def test_unknown_tagged_form_is_not_misparsed_as_application():
    dag = sexp_to_dag("(:custom (:c Nat) (:fv x))")
    root = dag.nodes[dag.expression_root_id]

    assert root.label == ":custom"
    assert root.node_type == "sconst"
    assert [dag.nodes[node_id].label for node_id in root.children] == ["Nat", "x"]


def test_model_application_preserves_argument_roles_and_positions():
    dag = sexp_to_dag(
        "(:app (:c Eq) "
        "(:arg :implicit-type 0 (:c Nat)) "
        "(:arg :explicit 1 (:fv FV0)) "
        "(:arg :proof 2 (:proof-of (:c True))))"
    )
    root = dag.nodes[dag.expression_root_id]
    arguments = [dag.nodes[node_id] for node_id in root.children[1:]]

    assert root.label == ":app"
    assert root.node_type == "sapp"
    assert [node.label for node in arguments] == [":arg", ":arg", ":arg"]
    assert [
        [dag.nodes[child].label for child in argument.children[:2]]
        for argument in arguments
    ] == [
        ["ArgRole::implicit-type", "ArgPosition:0"],
        ["ArgRole::explicit", "ArgPosition:1"],
        ["ArgRole::proof", "ArgPosition:2"],
    ]


def test_model_binder_preserves_role_and_debruijn_scope():
    dag = sexp_to_dag(
        "(:forall p :implicit (:sort Prop) "
        "(:app (:c Eq) (:arg :explicit 0 0)))"
    )
    root = dag.nodes[dag.expression_root_id]
    binder = dag.nodes[root.children[0]]
    body = dag.nodes[root.children[2]]

    assert dag.nodes[root.children[3]].label == "BinderRole::implicit"
    assert dag.nodes[dag.nodes[body.children[1]].children[2]].id == binder.id


def test_pantograph_raw_binder_role_is_accepted():
    dag = sexp_to_dag("(:lambda x (:c Nat) 0 :strictImplicit)")
    root = dag.nodes[dag.expression_root_id]
    assert dag.nodes[root.children[3]].label == "BinderRole::strictImplicit"


def test_pantograph_hypotheses_do_not_zip_with_text_branch_labels():
    dag = proof_state_to_dag(
        "case a.mk.mk\nx : Nat\nh : P x\n⊢ Q x",
        goal_sexp="(:c GoalType)",
        hyp_sexps=[("x", "(:c Nat)"), ("h", "(:c HypothesisType)")],
    )
    hypotheses = [node for node in dag.nodes if node.label == "Hyp"]

    assert len(hypotheses) == 2
    assert [
        [dag.nodes[child].label for child in hypothesis.children]
        for hypothesis in hypotheses
    ] == [["x", "Nat"], ["h", "HypothesisType"]]
    assert "case" not in get_node_labels(dag)


@pytest.mark.parametrize(
    "sexp, expected_tag",
    [
        ("(:forall x (:c Nat))", ":forall"),
        ("(:lambda x (:c Nat))", ":lambda"),
        ("(:let x (:c Nat) (:lit 1))", ":let"),
        ("(:c Nat extra)", ":c"),
        ("(:fv x extra)", ":fv"),
        ("(:sort 0 extra)", ":sort"),
        ("(:lit 1 extra)", ":lit"),
        ("(:mdata key)", ":mdata"),
        ("(:proj Prod 0)", ":proj"),
    ],
)
def test_malformed_known_forms_fail_explicitly(sexp, expected_tag):
    with pytest.raises(ValueError, match=expected_tag):
        sexp_to_dag(sexp)


def test_sort_prop():
    dag = sexp_to_dag("(:sort 0)")
    labels = get_node_labels(dag)
    assert "Prop" in labels


def test_sort_type():
    dag = sexp_to_dag("(:sort 1)")
    labels = get_node_labels(dag)
    assert "Type" in labels


# ---------------------------------------------------------------------------
# Edge structure
# ---------------------------------------------------------------------------

def test_edges_exist():
    dag = sexp_to_dag("((:c Or) 1 0)")
    assert len(dag.edges) > 0
    for child_id, parent_id in dag.edges:
        assert child_id < len(dag.nodes)
        assert parent_id < len(dag.nodes)


def test_root_nodes():
    dag = sexp_to_dag("(:forall q (:sort 0) 0)")
    roots = dag.root_nodes()
    assert len(roots) >= 1


def test_sexp_parser_records_the_explicit_expression_root():
    dag = sexp_to_dag("(:forall q (:sort 0) ((:c Or) 0 0))")

    assert dag.expression_root_id is not None
    assert dag.nodes[dag.expression_root_id].label == ":forall"


def test_leaf_nodes():
    dag = sexp_to_dag("(:forall q (:sort 0) 0)")
    leaves = dag.leaf_nodes()
    assert len(leaves) >= 1


# ---------------------------------------------------------------------------
# Proof state with sexp param
# ---------------------------------------------------------------------------

def test_proof_state_with_sexp():
    state = "h : Nat\n⊢ Nat"
    sexp = "(:forall a (:c Nat) (:c Nat))"
    dag = proof_state_to_dag(state, sexp=sexp)
    assert dag.num_nodes > 0
    labels = get_node_labels(dag)
    assert "Nat" in labels
    assert "h" in labels
    assert "Goal" in labels
    assert "Hyp" in labels
    assert "State" in labels

    goal = next(node for node in dag.nodes if node.label == "Goal")
    assert goal.children == (dag.expression_root_id,)


def test_proof_state_without_sexp_fallback():
    state = "h : Nat\n⊢ Nat"
    dag = proof_state_to_dag(state)
    assert dag.num_nodes > 0
    labels = get_node_labels(dag)
    assert "Nat" in labels


# ---------------------------------------------------------------------------
# DAGBuilder interface
# ---------------------------------------------------------------------------

def test_dag_builder_hash_consing():
    dag = DAGBuilder()
    id1 = dag.get_or_create("Nat", ())
    id2 = dag.get_or_create("Nat", ())
    assert id1 == id2
    assert dag.num_nodes == 1


def test_dag_builder_different_children():
    dag = DAGBuilder()
    nat = dag.get_or_create("Nat", ())
    bool_ = dag.get_or_create("Bool", ())
    id1 = dag.get_or_create("App", (nat, bool_))
    id2 = dag.get_or_create("App", (nat, nat))
    assert id1 != id2
    assert dag.num_nodes == 4  # Nat, Bool, App(nat,bool), App(nat,nat)


def test_dag_builder_stats():
    dag = sexp_to_dag("(:forall q (:sort 0) 0)")
    stats = dag.stats()
    assert stats.num_nodes > 0
    assert stats.num_edges > 0


if __name__ == "__main__":
    test_simple_forall()
    test_nested_forall()
    test_arrow()
    test_constants()
    test_lambda()
    test_complex_expression()
    test_hash_consing()
    test_node_count()
    test_node_count_no_sharing()
    test_forall_binder_annotation()
    test_nested_binder_depth()
    test_lambda_binder_annotation()
    test_context_variables_not_bound()
    test_debruijn_resolves_to_binder()
    test_debruijn_multiple_indices()
    test_free_variable()
    test_sort_prop()
    test_sort_type()
    test_edges_exist()
    test_root_nodes()
    test_leaf_nodes()
    test_proof_state_with_sexp()
    test_proof_state_without_sexp_fallback()
    test_dag_builder_hash_consing()
    test_dag_builder_different_children()
    test_dag_builder_stats()
    print("All tests passed!")
