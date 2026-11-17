#!/usr/bin/env python3
"""Unit tests for the SPACeR anchor (spacer-anchor proposal D1-D7 + review musts 1-5).

Covers the four review-hard-gated behaviors plus the proposal's config rules:
- test_anchor_disabled_bit_identical (D4): β=α=0 keys leave the rollout buffers
  and the torch RNG stream bit-identical to a config that never carried the
  spacer keys (same pattern as test_selfplay_off_rng_bit_equivalence).
- test_kl_term_exact_value (D2 + review §5-1): hand-computed forward
  KL(π_θ‖π_ref) on toy logits, exact loss contribution β·KL through the real
  _ppo_loss, and the direction asymmetry (reverse KL differs on the same
  fixture).
- test_anchor_reward_floored (D3 + review F1): α·clamp(logπ_ref(a|s), -floor, 0)
  injected into rewards with a logπ_ref = -8 nats action (floor 3.0 bites),
  a mid-range action untouched by the clamp, and masked rows excluded.
- test_normalize_rejected (review F2 / must 3): spacer on + normalize_rewards
  true fails at construction with ValueError and at schema validation.
- test_ref_shared_single_load (D1): the mixer's ref and the anchor ref are the
  same object; the anchor works independently with self-play off.
- test_spacer_config_validation (proposal §4): path required, use_rnn rejected,
  floor must be positive, defaults validate clean.
"""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pufferlib
from pufferlib.config_schema import validate_puffer_drive_config
from pufferlib.pufferl import (
    PuffeRL,
    SelfplayMixer,
    build_frozen_ref_policy,
    load_config,
    spacer_forward_kl,
)

OBS_DIM = 6
HORIZON = 4
NUM_AGENTS = 8


class StubPolicy(torch.nn.Module):
    """Tiny actor-critic whose logits are exactly ``action_bias`` when
    ``zero_init=True`` (all Linear weights/biases zeroed), so tests can assert
    hand-computed softmax values against the real training path."""

    def __init__(self, num_actions=4, action_bias=None, zero_init=True):
        super().__init__()
        self.backbone = torch.nn.Linear(OBS_DIM, 8)
        self.actor_head = torch.nn.Linear(8, num_actions)
        self.critic_head = torch.nn.Linear(8, 1)
        self.is_continuous = False
        self.action_bias = torch.nn.Parameter(torch.zeros(num_actions))
        if action_bias is not None:
            with torch.no_grad():
                self.action_bias.copy_(torch.as_tensor(action_bias, dtype=torch.float32))
        if zero_init:
            for module in (self.backbone, self.actor_head, self.critic_head):
                torch.nn.init.zeros_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, observations, state=None):
        hidden = torch.tanh(self.backbone(observations))
        return self.actor_head(hidden) + self.action_bias, self.critic_head(hidden)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


class StubVecenv:
    """Deterministic vecenv stand-in (same contract as test_selfplay_mixer)."""

    def __init__(self, num_agents=NUM_AGENTS, horizon=HORIZON):
        self.num_agents = num_agents
        self.horizon = horizon
        rng = np.random.RandomState(7)
        self.observations = rng.randn(num_agents, OBS_DIM).astype(np.float32)
        self.single_observation_space = SimpleNamespace(shape=(OBS_DIM,), dtype=np.dtype("float32"))
        self.single_action_space = SimpleNamespace(shape=(), dtype=np.dtype("int64"))
        self.action_space = SimpleNamespace(low=-np.ones(1), high=np.ones(1))
        self.driver_env = SimpleNamespace(obs_stats_feature_mask=np.ones(OBS_DIM, dtype=bool))

    def async_reset(self, seed=None):
        pass

    def recv(self):
        rows = self.num_agents
        env_id = np.arange(rows)
        reward = np.zeros(rows, dtype=np.float32)
        done = np.zeros(rows, dtype=np.float32)
        truncated = np.zeros(rows, dtype=np.float32)
        info = [{} for _ in range(rows)]
        mask = np.arange(rows) % 4 != 3
        return self.observations.copy(), reward, done, truncated, info, env_id, mask

    def send(self, action):
        pass

    def close(self):
        pass


def _train_config(**overrides):
    config = {
        "env": "stub",
        "run_name": "spacer_unit_test",
        "tf32": False,
        "torch_deterministic": False,
        "seed": 42,
        "batch_size": NUM_AGENTS * HORIZON,
        "bptt_horizon": HORIZON,
        "device": "cpu",
        "precision": "float32",
        "amp": False,
        "rollout_dtype": "float32",
        "cpu_offload": False,
        "use_rnn": False,
        "minibatch_size": NUM_AGENTS * HORIZON,
        "max_minibatch_size": NUM_AGENTS * HORIZON,
        "update_epochs": 1,
        "compile": False,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "adam_weight_decay": 0.0,
        "separate_grad_clip": False,
        "total_timesteps": 10_000_000,
        "normalize_rewards": False,
        "anneal_lr": False,
        "gamma": 0.999,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "vf_coef": 0.5,
        "vf_clip_coef": None,
        "max_grad_norm": 0.5,
        "ent_coef": 0.01,
        "ent_coef_anneal": False,
        "ent_coef_final": 0.001,
        "ent_coef_anneal_start_frac": 0.75,
        "use_value_bootstrapping": False,
        "adv_filter_enabled": False,
        "min_batch_size": None,
        "checkpoint_interval": 10_000_000,
        # v4 self-play keys (defaults = pure v3)
        "selfplay_p_learner": 1.0,
        "selfplay_p_ref": 0.0,
        "selfplay_p_snapshot": 0.0,
        "selfplay_ref_model_path": None,
        "selfplay_snapshot_pool_size": 4,
        "selfplay_snapshot_interval": 50,
        # SPACeR anchor keys (defaults = pure no-op; review must 3: floor 3.0)
        "spacer_kl_coef": 0.0,
        "spacer_kl_coef_anneal": False,
        "spacer_kl_coef_final": 0.0,
        "spacer_kl_coef_anneal_start_frac": 0.75,
        "spacer_reward_coef": 0.0,
        "spacer_logprob_floor": 3.0,
    }
    config.update(overrides)
    return config


def _seed_torch(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _save_ref(tmp_path, name, num_actions, action_bias):
    ref = StubPolicy(num_actions=num_actions, action_bias=action_bias)
    ref_path = tmp_path / f"{name}.pt"
    torch.save(ref.state_dict(), ref_path)
    return str(ref_path)


# ------------------------------------------------------------ D4 bit identity --


def _run_spacer_off_rollout(with_spacer_keys):
    _seed_torch(1234)
    vecenv = StubVecenv()
    policy = StubPolicy(num_actions=4, action_bias=[0.5, 0.2, -0.1, 0.0])
    config = _train_config()
    if not with_spacer_keys:
        for key in [k for k in config if k.startswith("spacer_")]:
            del config[key]
    rng_after_policy_init = torch.random.get_rng_state()

    trainer = PuffeRL(config, vecenv, policy)
    try:
        # Disabled spacer keys build no ref and consume no RNG (D4).
        assert trainer.anchor_ref is None
        assert torch.equal(torch.random.get_rng_state(), rng_after_policy_init)
        trainer.evaluate()
        trainer.last_log_time -= 10  # force the logging branch in train()
        trainer.train()
        snapshot = {
            "actions": trainer.actions.clone(),
            "rewards": trainer.rewards.clone(),
            "logprobs": trainer.logprobs.clone(),
            "values": trainer.values.clone(),
            "rng": torch.random.get_rng_state(),
        }
    finally:
        trainer.utilization.stop()
    return snapshot


def test_anchor_disabled_bit_identical():
    """D4: explicit β=α=0 spacer keys keep the rollout buffers and RNG stream
    bit-identical to a config that never carried the spacer keys."""
    v3 = _run_spacer_off_rollout(with_spacer_keys=False)
    off = _run_spacer_off_rollout(with_spacer_keys=True)

    assert torch.equal(v3["actions"], off["actions"])
    assert torch.equal(v3["rewards"], off["rewards"])
    assert torch.equal(v3["logprobs"], off["logprobs"])
    assert torch.equal(v3["values"], off["values"])
    assert torch.equal(v3["rng"], off["rng"])
    assert (v3["rewards"] == 0).all()  # stub env rewards are zero: no anchor leakage


# ------------------------------------------------------------- D2 KL exactness --


def test_kl_term_exact_value(tmp_path):
    """Forward KL hand value on toy logits, exact β·KL through _ppo_loss, and
    direction asymmetry (review §5-1: a flipped term silently mode-seeks)."""
    # Unit level, 3 classes: π_θ=[2,0,0], π_ref=[0,2,0]. Per-class logit gaps are
    # [2,-2,0], so KL = (2e^2 - 2) / (e^2 + 2).
    e2 = math.exp(2.0)
    theta = torch.tensor([[2.0, 0.0, 0.0]])
    ref = torch.tensor([[0.0, 2.0, 0.0]])
    expected_sym = (2.0 * e2 - 2.0) / (e2 + 2.0)
    assert torch.allclose(spacer_forward_kl(theta, ref), torch.tensor(expected_sym), atol=1e-6)
    assert torch.allclose(spacer_forward_kl(ref, theta), torch.tensor(expected_sym), atol=1e-6)

    # Asymmetric fixture pins the direction: π_θ=[2,0,0] vs π_ref=[1,1,1].
    # KL(P‖Q) = Σ p_θ·(θ−ref) − (lse(θ) − lse(ref)); with lse gap
    # L = ln((e²+2)/(3e)): forward = (e²−2)/(e²+2) − L, reverse = 1/3 + L.
    uniform = torch.tensor([[1.0, 1.0, 1.0]])
    lse_gap = math.log((e2 + 2.0) / (3.0 * math.e))
    expected_fwd = (e2 - 2.0) / (e2 + 2.0) - lse_gap
    expected_rev = 1.0 / 3.0 + lse_gap
    fwd = spacer_forward_kl(theta, uniform)
    rev = spacer_forward_kl(uniform, theta)
    assert torch.allclose(fwd, torch.tensor(expected_fwd), atol=1e-6)
    assert torch.allclose(rev, torch.tensor(expected_rev), atol=1e-6)
    assert not torch.allclose(fwd, rev, atol=1e-3)

    # Integration: zero-weight stubs make logits == action_bias, so the real
    # _ppo_loss sees exactly the toy fixture above (4 classes).
    num_actions = 4
    theta_bias = [2.0, 0.0, 0.0, 0.0]
    ref_bias = [0.0, 2.0, 0.0, 0.0]
    ref_path = _save_ref(tmp_path, "kl_ref", num_actions, ref_bias)
    _seed_torch(7)
    vecenv = StubVecenv()
    policy = StubPolicy(num_actions=num_actions, action_bias=theta_bias)
    trainer = PuffeRL(_train_config(spacer_kl_coef=0.03, selfplay_ref_model_path=ref_path), vecenv, policy)
    try:
        assert trainer.anchor_ref is not None
        n = 5
        mb_obs = torch.randn(n, OBS_DIM)
        mb_actions = torch.tensor([0, 1, 2, 3, 0])
        mb_logprobs = torch.zeros(n)
        mb_values = torch.zeros(n)
        mb_returns = torch.ones(n)
        mb_adv = torch.ones(n)

        trainer.config["spacer_kl_coef"] = 0.0
        loss_off, _, _, stats_off = trainer._ppo_loss(mb_obs, mb_actions, mb_logprobs, mb_values, mb_returns, mb_adv)
        assert "kl_to_ref" not in stats_off  # off path: no KL stats, no metric keys

        trainer.config["spacer_kl_coef"] = 0.03
        loss_on, _, _, stats_on = trainer._ppo_loss(mb_obs, mb_actions, mb_logprobs, mb_values, mb_returns, mb_adv)
        expected = (2.0 * e2 - 2.0) / (e2 + 3.0)  # 4 classes: denom e^2 + 3
        assert torch.allclose(stats_on["kl_to_ref"], torch.tensor(expected), atol=1e-6)
        assert stats_on["anchor_kl_coef"] == pytest.approx(0.03)
        assert torch.allclose(loss_on - loss_off, torch.tensor(0.03 * expected), atol=1e-7)
    finally:
        trainer.utilization.stop()


# -------------------------------------------------------- D3 anchored rewards --


def test_anchor_reward_floored(tmp_path):
    """α·clamp(logπ_ref, -floor, 0): a logπ_ref = -8 nats action is floored at
    -3.0 (review F1), a mid-range action keeps its exact value, and masked
    (partner) rows are excluded (D3)."""
    # 3 classes with bias = log-probs: logp(0) = -8 exactly, logp(2) = -1 exactly.
    bias = [-8.0, math.log(1.0 - math.exp(-8.0) - math.exp(-1.0)), -1.0]
    ref_path = _save_ref(tmp_path, "floor_ref", 3, bias)
    alpha, floor = 0.1, 3.0
    _seed_torch(11)
    vecenv = StubVecenv()
    policy = StubPolicy(num_actions=3, action_bias=[0.0, 0.0, 0.0])
    trainer = PuffeRL(
        _train_config(spacer_reward_coef=alpha, spacer_logprob_floor=floor, selfplay_ref_model_path=ref_path),
        vecenv,
        policy,
    )
    try:
        assert trainer.anchor_ref is not None
        trainer.rewards[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        trainer.actions[0] = torch.tensor([0, 2, 1, 0])  # logp: -8 (floored), -1, ~-0.46, -8 (floored)
        trainer.masks[0] = torch.tensor([True, True, False, True])  # row 2 = partner row

        losses = {}
        trainer._apply_anchor_reward(losses)

        expected = torch.tensor([1.0 - alpha * 3.0, 2.0 - alpha * 1.0, 3.0, 4.0 - alpha * 3.0])
        assert torch.allclose(trainer.rewards[0], expected, atol=1e-6)
        # Everything outside the crafted row stays untouched.
        assert (trainer.rewards[1:] == 0).all()
        # Sentinel stats over active rows only (rows 0, 1, 3).
        assert losses["anchor_reward_min"] == pytest.approx(-alpha * floor, abs=1e-6)
        assert losses["anchor_reward_mean"] == pytest.approx((-alpha * 3.0 - alpha * 1.0 - alpha * 3.0) / 3.0, abs=1e-6)
        assert losses["anchor_reward_min"] >= -alpha * floor - 1e-6
    finally:
        trainer.utilization.stop()


# --------------------------------------------------- review F2 hard rejection --


def test_normalize_rejected(tmp_path):
    """Spacer on + normalize_rewards=true is rejected at construction
    (ValueError) and at schema validation (review F2 / must 3)."""
    ref_path = _save_ref(tmp_path, "norm_ref", 4, [0.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="normalize_rewards"):
        PuffeRL(
            _train_config(spacer_kl_coef=0.01, normalize_rewards=True, selfplay_ref_model_path=ref_path),
            StubVecenv(),
            StubPolicy(num_actions=4),
        )

    with pytest.raises(ValueError, match="normalize_rewards"):
        PuffeRL(
            _train_config(spacer_reward_coef=0.001, normalize_rewards=True, selfplay_ref_model_path=ref_path),
            StubVecenv(),
            StubPolicy(num_actions=4),
        )

    # Schema level (the real pipeline gate): same rule, APIUsageError flavor.
    cfg = _load_default_config()
    cfg["train"].update(
        {"spacer_kl_coef": 0.01, "normalize_rewards": True, "selfplay_ref_model_path": "ref.pt"}
    )
    with pytest.raises(pufferlib.APIUsageError, match="normalize_rewards"):
        validate_puffer_drive_config(cfg, "training")


# -------------------------------------------------------------- D1 ref sharing --


def test_ref_shared_single_load(tmp_path):
    """D1: self-play + anchor share one frozen ref object; the anchor also works
    with self-play fully off (p_learner=1.0, no mixer)."""
    ref_path = _save_ref(tmp_path, "shared_ref", 4, [1.0, 0.0, 0.0, 0.0])
    _seed_torch(21)
    trainer = PuffeRL(
        _train_config(
            selfplay_p_learner=0.5,
            selfplay_p_ref=0.5,
            selfplay_ref_model_path=ref_path,
            spacer_kl_coef=0.01,
            spacer_reward_coef=0.001,
        ),
        StubVecenv(),
        StubPolicy(num_actions=4),
    )
    try:
        assert trainer.selfplay is not None
        assert trainer.anchor_ref is not None
        assert trainer.selfplay.ref_policy is trainer.anchor_ref  # one copy, not two
    finally:
        trainer.utilization.stop()

    _seed_torch(22)
    solo = PuffeRL(
        _train_config(spacer_kl_coef=0.01, selfplay_ref_model_path=ref_path),
        StubVecenv(),
        StubPolicy(num_actions=4),
    )
    try:
        assert solo.selfplay is None  # anchor independent of the mixer
        assert solo.anchor_ref is not None
        assert isinstance(solo.anchor_ref, type(solo.uncompiled_policy))
        assert not any(p.requires_grad for p in solo.anchor_ref.parameters())
        assert solo.anchor_ref.training is False
    finally:
        solo.utilization.stop()


def test_build_frozen_ref_policy_matches_state(tmp_path):
    """The extracted helper is the old 5-line mixer ref load: cleaned, frozen, strict."""
    bias = [3.0, 0.0, 0.0, 0.0]
    ref_path = _save_ref(tmp_path, "helper_ref", 4, bias)
    policy = StubPolicy(num_actions=4, action_bias=[0.0, 0.0, 0.0, 0.0])
    ref = build_frozen_ref_policy(policy, ref_path)
    assert torch.equal(ref.action_bias.detach(), torch.tensor(bias))
    assert ref.training is False
    assert not any(p.requires_grad for p in ref.parameters())
    # Shape/key mismatch fails fast (strict load), mirroring the mixer contract.
    bad_path = _save_ref(tmp_path, "helper_bad", 3, [0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError):
        build_frozen_ref_policy(policy, bad_path)


# ------------------------------------------------------------- schema level --


def _load_default_config():
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv
    return args


def test_spacer_config_validation():
    base = _load_default_config()
    # Defaults are pure no-ops and validate clean.
    assert base["train"]["spacer_kl_coef"] == 0.0
    assert base["train"]["spacer_reward_coef"] == 0.0
    assert base["train"]["spacer_logprob_floor"] == 3.0  # review must 3: 2-4 nats midpoint
    validate_puffer_drive_config(base, "training")

    missing_path = _load_default_config()
    missing_path["train"]["spacer_kl_coef"] = 0.03
    with pytest.raises(pufferlib.APIUsageError, match="selfplay_ref_model_path"):
        validate_puffer_drive_config(missing_path, "training")

    missing_path_alpha = _load_default_config()
    missing_path_alpha["train"]["spacer_reward_coef"] = 0.005
    with pytest.raises(pufferlib.APIUsageError, match="selfplay_ref_model_path"):
        validate_puffer_drive_config(missing_path_alpha, "training")

    rnn_conflict = _load_default_config()
    rnn_conflict["train"].update({"spacer_kl_coef": 0.03, "selfplay_ref_model_path": "ref.pt"})
    rnn_conflict["train"]["use_rnn"] = True
    with pytest.raises(pufferlib.APIUsageError, match="use_rnn"):
        validate_puffer_drive_config(rnn_conflict, "training")

    bad_floor = _load_default_config()
    bad_floor["train"].update({"spacer_kl_coef": 0.03, "selfplay_ref_model_path": "ref.pt"})
    bad_floor["train"]["spacer_logprob_floor"] = 0.0
    with pytest.raises(pufferlib.APIUsageError, match="spacer_logprob_floor"):
        validate_puffer_drive_config(bad_floor, "training")

    ok = _load_default_config()
    ok["train"].update(
        {
            "spacer_kl_coef": 0.03,
            "spacer_reward_coef": 0.005,
            "selfplay_ref_model_path": "ref.pt",
        }
    )
    validate_puffer_drive_config(ok, "training")
