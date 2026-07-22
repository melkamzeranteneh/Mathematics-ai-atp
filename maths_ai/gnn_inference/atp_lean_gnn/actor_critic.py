from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .argument_selector import ArgumentSelector, resolve_premise_mask
from .labels import get_tactic_arity
from .model import GraphSAGEStateClassifier
from .pyg import NODE_TYPE_TO_ID
from .reporting import console_print


@dataclass
class ActionSample:
    """One sampled action ``a = (tactic, args)`` with the quantities the RL update needs.

    Every tensor is batched over ``B`` states. ``tactic_logp`` and ``arg_logp`` carry
    gradient (they are the policy-gradient terms); ``value`` carries gradient (the critic
    output); ``tactic_action`` / ``arg_actions`` are the concrete sampled ids the environment
    consumes.
    """
    tactic_action: Tensor            # [B] sampled tactic ids
    tactic_logp: Tensor              # [B] log π(τ|s)
    tactic_entropy: Tensor           # [B] H(π(·|s))
    arg_actions: list[Tensor]        # per step, each [B] sampled node ids (padded index)
    arg_logp: Tensor                 # [B] Σ_k log π(u_k | s, τ, u_<k)
    value: Tensor                    # [B] V(s)
    tactic_logits: Tensor            # [B, num_tactics] (masked) — for entropy / BC downstream


class ActorHead(nn.Module):
    """Policy head: ``actor(h) = base(h) + residual(h)``.

    ``base`` is a single ``Linear(H → num_tactics)`` that inherits the supervised
    tactic classifier's weights at warm-start.  ``residual`` is a nonlinear
    correction whose final layer is initialized to zero, so ``residual(h) = 0``
    for every input at step 0 and ``actor(h) ≡ base(h) ≡ classifier(h)`` exactly.
    RL then grows the correction from zero, giving an exact behavioral warm-start
    while retaining the extra nonlinear capacity.
    """
    def __init__(self, hidden_dim: int, num_tactics: int, dropout: float = 0.2):
        super().__init__()
        self.base = nn.Linear(hidden_dim, num_tactics)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tactics),
        )
        # Zero the residual's output layer so the branch is silent at init.
        nn.init.zeros_(self.residual[3].weight)
        nn.init.zeros_(self.residual[3].bias)

    def forward(self, state_emb: Tensor, mask: Tensor | None = None) -> Tensor:
        # Returns [B, num_tactics] logits
        # If mask is provided, sets masked positions to -inf before returning
        logits = self.base(state_emb) + self.residual(state_emb)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class CriticHead(nn.Module):
    """Value head: state embedding → scalar V(s)."""
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_emb: Tensor) -> Tensor:
        # Returns [B, 1] value estimates
        return self.net(state_emb)


class ActorCriticWithArgsClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_node_labels: int,
        num_tactics: int,
        num_node_types: int = len(NODE_TYPE_TO_ID),
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_node_type: bool = True,
        max_args: int = 3,
    ):
        super().__init__()
        self.backbone = GraphSAGEStateClassifier(
            num_node_labels=num_node_labels,
            num_tactics=num_tactics,
            num_node_types=num_node_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_node_type=use_node_type,
        )
        self.actor = ActorHead(hidden_dim, num_tactics, dropout)
        self.critic = CriticHead(hidden_dim, dropout)
        self.tactic_embedding = nn.Embedding(num_tactics, hidden_dim)
        self.argument_selector = ArgumentSelector(hidden_dim)
        self.max_args = max_args
        self.hidden_dim = hidden_dim

    # ---- convenience accessors for the backbone ----
    @property
    def label_embedding(self) -> nn.Embedding:
        return self.backbone.label_embedding

    @property
    def node_type_embedding(self) -> nn.Embedding | None:
        return self.backbone.node_type_embedding

    def encode(
        self,
        data,
        *,
        tactic_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the shared GNN once. Return (node_embeddings, state_emb, tactic_logits, values).

        ``node_embeddings`` — [total_nodes, H]; ``state_emb`` — [B, H];
        ``tactic_logits`` — [B, num_tactics] (masked to legal tactics if ``tactic_mask``
        is given); ``values`` — [B, 1].
        """
        node_embeddings = self.backbone.encode_nodes(data)  # [total_nodes, H]
        state_emb = self.backbone.readout(node_embeddings, data)  # [B, H]
        tactic_logits = self.actor(state_emb, mask=tactic_mask)  # [B, num_tactics]
        values = self.critic(state_emb)  # [B, 1]
        return node_embeddings, state_emb, tactic_logits, values

    def select_arguments(
        self,
        data,
        node_embeddings: Tensor,
        state_emb: Tensor,
        tactic_ids: Tensor,
        *,
        tactic_names: list[str] | None = None,
    ) -> list[Tensor]:
        """Run only the pointer head, conditioned on ``tactic_ids``, reusing the
        already-computed encoder outputs (no second GNN pass)."""
        tactic_emb = self.tactic_embedding(tactic_ids)  # [B, H]

        # Determine how many argument steps to run
        if tactic_names is not None:
            n_steps = max(get_tactic_arity(name) for name in tactic_names)
            n_steps = min(n_steps, self.max_args)
        else:
            n_steps = self.max_args

        if n_steps == 0:
            return []

        # Autoregressive argument selection using the shared candidate mask —
        # identical to the supervised pointer's so warm-started argument
        # behavior transfers and RL-sampled indices stay valid at recompute.
        premise_mask = resolve_premise_mask(data, node_embeddings.device)
        batch_index = data.batch.to(device=node_embeddings.device)

        arg_logits_list: list[Tensor] = []
        prev_arg_emb: Tensor | None = None

        for _ in range(n_steps):
            scores, selected_emb = self.argument_selector(
                state_emb, tactic_emb, node_embeddings, premise_mask, batch_index,
                prev_arg_emb=prev_arg_emb,
            )
            arg_logits_list.append(scores)
            prev_arg_emb = selected_emb

        return arg_logits_list

    def act(
        self,
        data,
        *,
        tactic_mask: Tensor | None = None,
        id_to_tactic: dict[int, str] | None = None,
        greedy: bool = False,
    ) -> ActionSample:
        """Sample a full action ``(tactic, args)`` on-policy and return it with its
        log-probs, entropy, and value — the quantities the RL rollout/update consume (B2).

        The tactic and every argument are SAMPLED from the policy (unless ``greedy``), the
        arguments are fed back autoregressively using the SAMPLED node's embedding, and the
        summed argument log-prob ``Σ_k log π(u_k)`` is returned so the pointer is trained by
        the same advantage as the tactic (pointer-as-actor). ``id_to_tactic`` lets the number
        of argument steps follow the sampled tactic's arity; without it, ``max_args`` steps run.
        """
        node_embeddings, state_emb, tactic_logits, values = self.encode(data, tactic_mask=tactic_mask)

        tactic_dist = torch.distributions.Categorical(logits=tactic_logits)
        if greedy:
            tactic_action = tactic_logits.argmax(dim=1)
        else:
            tactic_action = tactic_dist.sample()
        tactic_logp = tactic_dist.log_prob(tactic_action)
        tactic_entropy = tactic_dist.entropy()

        batch_size = tactic_action.size(0)
        device = tactic_action.device
        arg_logp = torch.zeros(batch_size, device=device)
        arg_actions: list[Tensor] = []

        # Number of argument steps: follow the sampled tactic's arity if we can decode names.
        if id_to_tactic is not None:
            names = [id_to_tactic.get(int(a), "") for a in tactic_action.tolist()]
            n_steps = min(max((get_tactic_arity(n) for n in names), default=0), self.max_args)
        else:
            n_steps = self.max_args

        if n_steps > 0:
            tactic_emb = self.tactic_embedding(tactic_action)  # [B, H]
            premise_mask = resolve_premise_mask(data, node_embeddings.device)
            batch_index = data.batch.to(device=node_embeddings.device)

            prev_arg_emb: Tensor | None = None
            for _ in range(n_steps):
                scores, selected_idx, step_logp, selected_emb = self.argument_selector.sample_step(
                    state_emb, tactic_emb, node_embeddings, premise_mask, batch_index,
                    prev_arg_emb=prev_arg_emb,
                )
                # Rows with no valid premise (all -inf) yield NaN log-prob; zero them so a
                # degenerate state contributes no argument gradient/credit.
                valid = torch.isfinite(scores).any(dim=1)
                step_logp = torch.where(valid, step_logp, torch.zeros_like(step_logp))
                arg_logp = arg_logp + step_logp
                arg_actions.append(selected_idx)
                prev_arg_emb = selected_emb

        return ActionSample(
            tactic_action=tactic_action,
            tactic_logp=tactic_logp,
            tactic_entropy=tactic_entropy,
            arg_actions=arg_actions,
            arg_logp=arg_logp,
            value=values.squeeze(-1),
            tactic_logits=tactic_logits,
        )

    def evaluate_actions(
        self,
        data,
        tactic_ids: Tensor,          # [B] long — the tactics that were executed
        arg_indices: Tensor | None,  # [B, T] long, -1 padded — the argument nodes that were sampled
        *,
        tactic_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Recompute the executed actions' log-probs under the CURRENT parameters.

        The collect phase stores only integer actions (no autograd graph); the train phase
        calls this to rebuild ``log π(τ|s)`` and ``Σ_k log π(u_k|s,τ)`` with gradient, feeding
        each stored argument's embedding forward autoregressively via
        ``ArgumentSelector.forced_step``. Valid only under the one-optimizer-step-per-collect
        invariant (parameters unchanged between collect and train), otherwise the recomputed
        probabilities are off-policy and would need a PPO-style ratio correction.

        Return ``(tactic_logp [B], arg_logp [B], entropy [B], values [B], tactic_logits)``.
        """
        node_embeddings, state_emb, tactic_logits, values = self.encode(data, tactic_mask=tactic_mask)

        log_probs = torch.log_softmax(tactic_logits, dim=-1)
        tactic_logp = log_probs.gather(1, tactic_ids.unsqueeze(1)).squeeze(1)  # [B]
        entropy = torch.distributions.Categorical(logits=tactic_logits).entropy()  # [B]

        arg_logp = torch.zeros_like(tactic_logp)
        if arg_indices is not None and arg_indices.numel() > 0 and arg_indices.size(1) > 0:
            tactic_emb = self.tactic_embedding(tactic_ids)  # [B, H]
            premise_mask = resolve_premise_mask(data, node_embeddings.device)
            batch_index = data.batch.to(device=node_embeddings.device)

            prev_arg_emb: Tensor | None = None
            for t in range(arg_indices.size(1)):
                step_logp, selected_emb = self.argument_selector.forced_step(
                    state_emb, tactic_emb, node_embeddings, premise_mask, batch_index,
                    arg_indices[:, t], prev_arg_emb=prev_arg_emb,
                )
                arg_logp = arg_logp + step_logp
                prev_arg_emb = selected_emb

        return tactic_logp, arg_logp, entropy, values.squeeze(-1), tactic_logits

    def forward(
        self,
        data,
        *,
        tactic_mask: Tensor | None = None,
        teacher_tactic_ids: Tensor | None = None,
        tactic_names: list[str] | None = None,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Return (tactic_logits, value_estimates, arg_logits_list).

        Thin wrapper over ``encode`` + ``select_arguments`` for callers that want the
        full model in one call (inference, tests). The RL training loop calls the two
        methods directly so it can sample the tactic between them.
        """
        node_embeddings, state_emb, tactic_logits, values = self.encode(
            data, tactic_mask=tactic_mask
        )

        if teacher_tactic_ids is not None:
            tactic_ids = teacher_tactic_ids
        else:
            tactic_ids = tactic_logits.argmax(dim=1)  # [B]

        arg_logits_list = self.select_arguments(
            data, node_embeddings, state_emb, tactic_ids, tactic_names=tactic_names
        )
        return tactic_logits, values, arg_logits_list


def init_actor_from_supervised(model: ActorCriticWithArgsClassifier) -> None:
    """Copy backbone.classifier weights/biases into the actor's inherited ``base`` layer.

    After this call ``actor(h) ≡ classifier(h)`` because the residual branch is
    zero at initialization.
    """
    with torch.no_grad():
        model.actor.base.weight.copy_(model.backbone.classifier.weight)
        if model.backbone.classifier.bias is not None:
            model.actor.base.bias.copy_(model.backbone.classifier.bias)


def load_from_pointer_checkpoint(
    model: ActorCriticWithArgsClassifier,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """Load TacticWithArgsClassifier checkpoint weights into ActorCriticWithArgsClassifier.

    Raises ``ValueError`` on any intended-transfer key whose checkpoint shape differs
    from the model's — ``load_state_dict(strict=False)`` would otherwise drop such keys
    silently (e.g. a hidden_dim mismatch between checkpoint and config), leaving the
    model at random init while reporting a successful warm-start.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # We will build a new state dict compatible with ActorCriticWithArgsClassifier
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_state_dict[k] = v
        elif k.startswith("tactic_embedding."):
            new_state_dict[k] = v
        elif k.startswith("argument_selector."):
            new_state_dict[k] = v

    # Guard: a shape-mismatched key is dropped by strict=False without appearing in
    # missing_keys, so diff shapes explicitly before loading.
    model_sd = model.state_dict()
    mismatched = [
        (k, tuple(v.shape), tuple(model_sd[k].shape))
        for k, v in new_state_dict.items()
        if k in model_sd and v.shape != model_sd[k].shape
    ]
    if mismatched:
        raise ValueError(
            "Warm-start shape mismatch (hidden_dim disagreement between checkpoint "
            f"and model config?): {mismatched}"
        )

    result = model.load_state_dict(new_state_dict, strict=False)
    console_print(
        f"Warm-start: loaded {len(new_state_dict)} tensors from pointer checkpoint; "
        f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
    )
    init_actor_from_supervised(model)
