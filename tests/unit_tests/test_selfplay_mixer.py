#!/usr/bin/env python3
"""Unit tests for the v4 SPACeR self-play mixer (proposal v4-selfplay + review musts).

Covers:
- test_partner_rows_follow_assigned_policy (proposal red/green test): partner
  rows' sampled actions come from their assigned policy, learner rows from the
  learner, and ego_agent_mask excludes partner slots.
- test_partner_rows_carry_source_value_and_zero_logprob (D2/F5)
- test_snapshot_pool_ring_write_changes_partner_actions (G6)
- test_selfplay_config_validation (D4 schema rules)
- test_selfplay_off_short_circuit (review must 3: p_learner=1.0 skips role
  assignment, leaves masks untouched, and keeps the original single forward)
- test_selfplay_off_rng_bit_equivalence (review must 2: off keys leave the
  torch RNG stream and sampled actions bit-identical to a v3-style config)
- test_selfplay_on_masks_match_roles (G2/D3 integration through PuffeRL.evaluate)
- test_selfplay_continuity_guard_fails_fast (review must 4)
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pufferlib
from pufferlib.config_schema import validate_puffer_drive_config
from pufferlib.pufferl import PuffeRL, SelfplayMixer, load_config

OBS_DIM = 6
NUM_ACTIONS = 10
HORIZON = 4
NUM_AGENTS = 8


class StubPolicy(torch.nn.Module):
    """Tiny actor-critic with a fixed per-action bias on the actor logits.

    A large bias on one action makes multinomial sampling deterministic in
    practice, so tests can assert exact per-row actions per source policy.
    """

    def __init__(self, preferred_action=None, bias=50.0):
        super().__init__()
        self.backbone = torch.nn.Linear(OBS_DIM, 8)
        self.actor_head = torch.nn.Linear(8, NUM_ACTIONS)
        self.critic_head = torch.nn.Linear(8, 1)
        self.is_continuous = False
        self.action_bias = torch.nn.Parameter(torch.zeros(NUM_ACTIONS))
        if preferred_action is not None:
            with torch.no_grad():
                self.action_bias[preferred_action] = bias

    def forward(self, observations, state=None):
        hidden = torch.tanh(self.backbone(observations))
        return self.actor_head(hidden) + self.action_bias, self.critic_head(hidden)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


class StubVecenv:
    """Deterministic vecenv stand-in for PuffeRL.evaluate().

    Every recv returns one contiguous block of all agent slots with fixed
    observations and a fixed mask pattern (every 4th row masked). ``env_ids``
    overrides the returned agent-id block to simulate non-contiguous arrivals.
    """

    def __init__(self, num_agents=NUM_AGENTS, horizon=HORIZON, env_ids=None):
        self.num_agents = num_agents
        self.horizon = horizon
        self.env_ids = env_ids
        rng = np.random.RandomState(7)
        self.observations = rng.randn(num_agents, OBS_DIM).astype(np.float32)
        self.single_observation_space = SimpleNamespace(shape=(OBS_DIM,), dtype=np.dtype("float32"))
        self.single_action_space = SimpleNamespace(shape=(), dtype=np.dtype("int64"))
        self.action_space = SimpleNamespace(low=-np.ones(1), high=np.ones(1))
        self.driver_env = SimpleNamespace(obs_stats_feature_mask=np.ones(OBS_DIM, dtype=bool))

    def async_reset(self, seed=None):
        pass

    def recv(self):
        rows = self.num_agents if self.env_ids is None else len(self.env_ids)
        env_id = np.arange(self.num_agents) if self.env_ids is None else np.asarray(self.env_ids)
        reward = np.zeros(rows, dtype=np.float32)
        done = np.zeros(rows, dtype=np.float32)
        truncated = np.zeros(rows, dtype=np.float32)
        info = [{} for _ in range(rows)]
        mask = np.arange(rows) % 4 != 3
        return self.observations[:rows].copy(), reward, done, truncated, info, env_id, mask

    def send(self, action):
        pass

    def close(self):
        pass


def _env_mask_pattern(rows):
    return torch.as_tensor(np.arange(rows) % 4 != 3)


def _train_config(**selfplay):
    config = {
        "env": "stub",
        "run_name": "selfplay_unit_test",
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
        "use_value_bootstrapping": False,
        "gamma": 0.999,
        "checkpoint_interval": 10_000_000,
        # v4 self-play keys (defaults = pure v3: mixer never constructed)
        "selfplay_p_learner": 1.0,
        "selfplay_p_ref": 0.0,
        "selfplay_p_snapshot": 0.0,
        "selfplay_ref_model_path": None,
        "selfplay_snapshot_pool_size": 4,
        "selfplay_snapshot_interval": 50,
    }
    config.update(selfplay)
    return config


def _build_trainer(vecenv, policy, **selfplay):
    return PuffeRL(_train_config(**selfplay), vecenv, policy)


def _seed_torch(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _save_ref(tmp_path, preferred_action):
    ref = StubPolicy(preferred_action=preferred_action)
    ref_path = tmp_path / f"ref_{preferred_action}.pt"
    torch.save(ref.state_dict(), ref_path)
    return str(ref_path)


# --------------------------------------------------------------- mixer level --


def test_partner_rows_follow_assigned_policy(tmp_path):
    """Proposal red/green test: rows follow their assigned policy, not the learner."""
    learner = StubPolicy(preferred_action=7)
    ref_path = _save_ref(tmp_path, preferred_action=3)
    vecenv = StubVecenv()
    config = _train_config(
        selfplay_p_learner=0.5,
        selfplay_p_ref=0.5,
        selfplay_ref_model_path=ref_path,
    )
    mixer = SelfplayMixer(learner, vecenv.num_agents, config, seed=42, device="cpu")

    # Inject the spec's role layout directly: odd slots reference, even slots learner.
    roles = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int64)
    mixer.roles = roles
    mixer.ego_agent_mask = roles == 0

    obs = torch.as_tensor(vecenv.observations)
    action, _logprob, _value, cont_action = mixer.forward_batch(
        obs, roles, state={}, learner_policy=learner, env_continuous=False
    )

    assert action.tolist() == [7, 3, 7, 3, 7, 3, 7, 3]
    assert cont_action is None
    # G2: partner slots are never ego rows.
    assert not mixer.ego_agent_mask[roles == 1].any()
    assert mixer.ego_agent_mask[roles == 0].all()


def test_partner_rows_carry_source_value_and_zero_logprob(tmp_path):
    """D2: partner value comes from the source policy; partner logprob is exactly 0."""
    # Moderate bias: a near-unity chosen probability would round the learner's
    # own logprob to exactly 0.0, making it indistinguishable from the zeroed
    # partner logprobs this test must distinguish.
    learner = StubPolicy(preferred_action=7, bias=2.0)
    ref_path = _save_ref(tmp_path, preferred_action=3)
    vecenv = StubVecenv()
    config = _train_config(
        selfplay_p_learner=0.5,
        selfplay_p_ref=0.5,
        selfplay_ref_model_path=ref_path,
    )
    mixer = SelfplayMixer(learner, vecenv.num_agents, config, seed=42, device="cpu")
    roles = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int64)

    obs = torch.as_tensor(vecenv.observations)
    _action, logprob, value, _cont_action = mixer.forward_batch(
        obs, roles, state={}, learner_policy=learner, env_continuous=False
    )

    partner_rows = torch.tensor([1, 3, 5, 7])
    learner_rows = torch.tensor([0, 2, 4, 6])
    # Partner values are their source policy's critic outputs.
    with torch.no_grad():
        _, ref_value = mixer.ref_policy.forward_eval(obs)
    assert torch.allclose(value[partner_rows], ref_value.flatten().float()[partner_rows])
    with torch.no_grad():
        _, learner_value = learner.forward_eval(obs)
    assert torch.allclose(value[learner_rows], learner_value.flatten().float()[learner_rows])
    # Partner logprobs are zeroed (they never enter the PPO ratio; see review F5).
    assert (logprob[partner_rows] == 0).all()
    assert (logprob[learner_rows] != 0).all()


def test_snapshot_pool_ring_write_changes_partner_actions(tmp_path):
    """G6: interval-gated ring insertions; partner actions follow pool contents."""
    learner = StubPolicy(preferred_action=7)
    vecenv = StubVecenv(num_agents=4)
    config = _train_config(
        selfplay_p_learner=0.0,
        selfplay_p_ref=0.0,
        selfplay_p_snapshot=1.0,
        selfplay_snapshot_pool_size=2,
        selfplay_snapshot_interval=1,
    )
    mixer = SelfplayMixer(learner, vecenv.num_agents, config, seed=42, device="cpu")
    obs = torch.as_tensor(vecenv.observations)

    slot0 = torch.full((4,), 2, dtype=torch.int64)  # everyone reads snapshot slot 0
    slot1 = torch.full((4,), 3, dtype=torch.int64)  # snapshot slot 1

    # Epoch-0 pool = learner initial weights.
    before, _, _, _ = mixer.forward_batch(obs, slot0, state={}, learner_policy=learner, env_continuous=False)
    assert before.tolist() == [7, 7, 7, 7]

    def _prefer(action):
        with torch.no_grad():
            learner.action_bias[:] = 0.0
            learner.action_bias[action] = 50.0

    def _forward(roles):
        out, _, _, _ = mixer.forward_batch(obs, roles, state={}, learner_policy=learner, env_continuous=False)
        return out.tolist()

    _prefer(4)
    assert mixer.snapshot_update(epoch=0) is True
    assert mixer.snapshot_writes == 1
    assert _forward(slot0) == [4, 4, 4, 4]

    _prefer(5)
    assert mixer.snapshot_update(epoch=1) is True
    assert mixer.snapshot_writes == 2
    assert _forward(slot1) == [5, 5, 5, 5]
    assert _forward(slot0) == [4, 4, 4, 4]  # slot 0 untouched by the slot-1 write

    _prefer(6)
    assert mixer.snapshot_update(epoch=2) is True  # ring wraps back to slot 0
    assert mixer.snapshot_writes == 3
    assert _forward(slot0) == [6, 6, 6, 6]
    assert _forward(slot1) == [5, 5, 5, 5]  # slot 1 survived the wrap

    # Interval gating: only every `interval`-th completed epoch inserts.
    gated = SelfplayMixer(
        StubPolicy(),
        vecenv.num_agents,
        _train_config(
            selfplay_p_learner=0.0,
            selfplay_p_ref=0.0,
            selfplay_p_snapshot=1.0,
            selfplay_snapshot_pool_size=2,
            selfplay_snapshot_interval=3,
        ),
        seed=42,
        device="cpu",
    )
    assert gated.snapshot_update(epoch=0) is False
    assert gated.snapshot_update(epoch=1) is False
    assert gated.snapshot_writes == 0
    assert gated.snapshot_update(epoch=2) is True
    assert gated.snapshot_writes == 1


# ------------------------------------------------------- config schema level --


def _load_default_config():
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv
    return args


def test_selfplay_config_validation():
    # Defaults are pure v3: the learner drives every agent.
    base = _load_default_config()
    assert base["train"]["selfplay_p_learner"] == 1.0
    assert base["train"]["selfplay_ref_model_path"] is None
    validate_puffer_drive_config(base, "training")

    bad_sum = _load_default_config()
    bad_sum["train"]["selfplay_p_learner"] = 0.5
    with pytest.raises(pufferlib.APIUsageError, match="selfplay_p_learner"):
        validate_puffer_drive_config(bad_sum, "training")

    missing_ref = _load_default_config()
    missing_ref["train"].update({"selfplay_p_learner": 0.5, "selfplay_p_ref": 0.5})
    with pytest.raises(pufferlib.APIUsageError, match="selfplay_ref_model_path"):
        validate_puffer_drive_config(missing_ref, "training")

    rnn_conflict = _load_default_config()
    rnn_conflict["train"].update(
        {"selfplay_p_learner": 0.5, "selfplay_p_ref": 0.5, "selfplay_ref_model_path": "ref.pt"}
    )
    rnn_conflict["train"]["use_rnn"] = True
    with pytest.raises(pufferlib.APIUsageError, match="use_rnn"):
        validate_puffer_drive_config(rnn_conflict, "training")

    ok = _load_default_config()
    ok["train"].update(
        {
            "selfplay_p_learner": 0.5,
            "selfplay_p_ref": 0.3,
            "selfplay_p_snapshot": 0.2,
            "selfplay_ref_model_path": "ref.pt",
            "selfplay_snapshot_pool_size": 1,
            "selfplay_snapshot_interval": 10,
        }
    )
    validate_puffer_drive_config(ok, "training")


# ------------------------------------------------------- PuffeRL integration --


def test_selfplay_off_short_circuit():
    """Review must 3: with p_learner=1.0 the three off-path short-circuits hold."""
    _seed_torch(123)
    vecenv = StubVecenv()
    policy = StubPolicy(preferred_action=5)
    rng_before_init = torch.random.get_rng_state()

    trainer = _build_trainer(vecenv, policy)
    try:
        # (1) init skips role assignment entirely: no mixer object, no RNG draw.
        assert trainer.selfplay is None
        assert torch.equal(torch.random.get_rng_state(), rng_before_init)

        # (3) rollout keeps the original single full-block forward path.
        calls = []
        original_forward_eval = trainer.policy.forward_eval

        def spy(obs, state=None):
            calls.append(tuple(obs.shape))
            return original_forward_eval(obs, state)

        trainer.policy.forward_eval = spy
        trainer.evaluate()
        assert len(calls) == HORIZON  # exactly one forward per recv block
        assert set(calls) == {(NUM_AGENTS, OBS_DIM)}  # always the full block, never a role subset

        # (2) stored masks are the env masks verbatim (no ego-mask AND happened).
        expected = torch.zeros(NUM_AGENTS, HORIZON, dtype=torch.bool)
        for step in range(HORIZON):
            expected[:, step] = _env_mask_pattern(NUM_AGENTS)
        assert torch.equal(trainer.masks, expected)
        assert not trainer.masks.all()  # the env pattern has False rows: identity is observable
    finally:
        trainer.utilization.stop()


def _run_off_rollout(with_selfplay_keys):
    _seed_torch(1234)
    vecenv = StubVecenv()
    policy = StubPolicy(preferred_action=5)
    config = _train_config()
    if not with_selfplay_keys:
        for key in [k for k in config if k.startswith("selfplay_")]:
            del config[key]
    rng_after_policy_init = torch.random.get_rng_state()

    trainer = PuffeRL(config, vecenv, policy)
    try:
        # __init__ consumes no torch RNG when self-play is off.
        assert torch.equal(torch.random.get_rng_state(), rng_after_policy_init)
        trainer.evaluate()
        actions = trainer.actions.clone()
        rng_after_evaluate = torch.random.get_rng_state()
    finally:
        trainer.utilization.stop()
    return actions, rng_after_evaluate


def test_selfplay_off_rng_bit_equivalence():
    """Review must 2: explicit off keys keep the v3 action stream and RNG stream
    bit-identical to a config that never carried the selfplay keys."""
    actions_v3, rng_v3 = _run_off_rollout(with_selfplay_keys=False)
    actions_off, rng_off = _run_off_rollout(with_selfplay_keys=True)

    assert torch.equal(actions_v3, actions_off)
    assert torch.equal(rng_v3, rng_off)
    assert ((actions_v3 >= 0) & (actions_v3 < NUM_ACTIONS)).all()


def test_selfplay_on_masks_match_roles(tmp_path):
    """G2/D3 through the real evaluate(): buffer masks = env mask AND ego mask."""
    ref_path = _save_ref(tmp_path, preferred_action=3)
    _seed_torch(99)
    vecenv = StubVecenv()
    policy = StubPolicy(preferred_action=5)
    trainer = _build_trainer(
        vecenv,
        policy,
        selfplay_p_learner=0.5,
        selfplay_p_ref=0.5,
        selfplay_ref_model_path=ref_path,
    )
    try:
        mixer = trainer.selfplay
        assert mixer is not None
        ego = mixer.ego_agent_mask.cpu()
        assert 0 < int(ego.sum()) < NUM_AGENTS  # both roles present in this draw

        trainer.evaluate()

        expected = torch.zeros(NUM_AGENTS, HORIZON, dtype=torch.bool)
        for step in range(HORIZON):
            expected[:, step] = _env_mask_pattern(NUM_AGENTS) & ego
        assert torch.equal(trainer.masks, expected)

        partner_rows = (~ego).nonzero().flatten()
        learner_rows = ego.nonzero().flatten()
        assert (trainer.actions[partner_rows] == 3).all()
        assert (trainer.actions[learner_rows] == 5).all()
        # Partner logprobs are exactly zero (masked out of the PPO ratio; see
        # review F5). The learner's own logprobs are checked (with a suitable
        # bias) in test_partner_rows_carry_source_value_and_zero_logprob.
        assert (trainer.logprobs[partner_rows] == 0).all()

        obs = torch.as_tensor(vecenv.observations)
        with torch.no_grad():
            _, ref_value = mixer.ref_policy.forward_eval(obs)
        for step in range(HORIZON):
            assert torch.allclose(trainer.values[partner_rows, step], ref_value.flatten().float()[partner_rows])
    finally:
        trainer.utilization.stop()


def test_selfplay_continuity_guard_fails_fast(tmp_path):
    """Review must 4: a non-contiguous agent block fails fast instead of
    silently scattering self-play masks across wrong agent slots."""
    ref_path = _save_ref(tmp_path, preferred_action=3)
    _seed_torch(99)
    vecenv = StubVecenv(env_ids=[0, 1, 3])  # span 4 != 3 recv rows
    policy = StubPolicy(preferred_action=5)
    trainer = _build_trainer(
        vecenv,
        policy,
        selfplay_p_learner=0.5,
        selfplay_p_ref=0.5,
        selfplay_ref_model_path=ref_path,
    )
    try:
        with pytest.raises(AssertionError, match="non-contiguous"):
            trainer.evaluate()
    finally:
        trainer.utilization.stop()
