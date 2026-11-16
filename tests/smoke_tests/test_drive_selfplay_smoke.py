#!/usr/bin/env python3
"""Self-play (v4 SPACeR) CPU smoke test: G3 mask flow, G4 loop health, G6 pool writes.

Runs the real pipeline (load_config -> load_env -> load_policy -> PuffeRL) with the
self-play mixer enabled (p_learner=0.5, p_ref=0.25, p_snapshot=0.25, interval=1) for
2 epochs on CPU with the Serial backend, mirroring tests/smoke_tests/test_drive_train.py.
The reference checkpoint is the learner's own initial weights written to a temp file
(architecture-identical by construction), so the run stays self-contained.

Writes artifacts/v4-selfplay-smoke/smoke.json with the v4 proposal §3 fields and
asserts:
  G3  |masked_fraction - expected masked fraction| < 0.03, where the expectation
      accounts for the env's own blind-agent masking (measured in-run by wrapping
      vecenv.recv): expected = 1 - env_keep_fraction * ego_fraction (role and
      blindness draws are independent). The naive "expected = partner fraction"
      reading from the proposal assumes an unmasked env; the drive env masks
      ~24% of rows even with selfplay off (see golden masked_fraction).
  G4  all training losses finite (no NaN/Inf) within the watchdog
  G6  snapshot pool writes == epochs (interval=1, 2 epochs)
Note (review A2): the v3 golden gate is a 2% METRIC regression, not bit-level; the
bit-level claim for selfplay=off is covered by tests/unit_tests/test_selfplay_mixer.py.
"""

import json
import os
import random
import signal
import sys

# Stabilize CPU math for cross-machine reproducibility (must be set before
# torch/numpy load their BLAS); mirrors test_drive_train.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")
os.environ.setdefault("DNNL_MAX_CPU_ISA", "AVX2")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.pufferl import PuffeRL, load_config, load_env, load_policy
from pufferlib.utils import finalize_environment_metrics

SEED = 42
EPOCHS = 2
BPTT_HORIZON = 64  # == scenario_length so episodes complete each epoch
WATCHDOG_SECONDS = 1800  # local budget: 30 min; shrink num_envs if this trips

NUM_ENVS = int(os.environ.get("SELFPLAY_SMOKE_NUM_ENVS", "2"))
NUM_AGENTS = 128
P_LEARNER, P_REF, P_SNAPSHOT = 0.5, 0.25, 0.25
MASK_EXCESS_TOL = 0.03

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_PATH = os.path.join(REPO_ROOT, "artifacts", "v4-selfplay-smoke", "smoke.json")


class _DummyLogger:
    run_id = "selfplay_smoke"

    def log(self, *args, **kwargs):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def _set_existing(section, updates):
    for k, v in updates.items():
        if k in section:
            section[k] = v


def _build_config():
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv

    _set_existing(
        args["vec"],
        {"backend": "Serial", "num_envs": NUM_ENVS, "seed": SEED},
    )
    _set_existing(
        args["env"],
        {
            "num_agents": NUM_AGENTS,
            "min_agents_per_env": NUM_AGENTS,
            "max_agents_per_env": NUM_AGENTS,
            "action_type": "discrete",
            "num_maps": 2,
            "use_map_cache": True,
            "map_dir": "pufferlib/resources/drive/binaries/carla",
            "scenario_length": BPTT_HORIZON,
            "stagger_first_episode": False,
            "seed": SEED,
        },
    )
    _set_existing(
        args["policy"],
        {
            "ego_input_size": 32,
            "partner_input_size": 128,
            "lane_input_size": 64,
            "boundary_input_size": 64,
            "traffic_control_input_size": 16,
            "context_input_size": 8,
            "backbone_hidden_size": 256,
            "actor_hidden_size": 128,
            "critic_hidden_size": 64,
            "fused_slot_encoder": False,
        },
    )
    _set_existing(args["rnn"], {"input_size": 256, "hidden_size": 256})

    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = None
    return args


def _finalize_train_config(args, total_agents, ref_model_path):
    minibatch_size = total_agents * BPTT_HORIZON // 8
    _set_existing(
        args["train"],
        {
            "device": "cpu",
            "compile": False,
            "seed": SEED,
            "torch_deterministic": True,
            "anneal_lr": False,
            "learning_rate": 0.001,
            "update_epochs": 1,
            "bptt_horizon": BPTT_HORIZON,
            "minibatch_size": minibatch_size,
            "max_minibatch_size": minibatch_size,
            "total_timesteps": 10_000_000,
            "checkpoint_interval": 10_000_000,
            "selfplay_p_learner": P_LEARNER,
            "selfplay_p_ref": P_REF,
            "selfplay_p_snapshot": P_SNAPSHOT,
            "selfplay_ref_model_path": ref_model_path,
            "selfplay_snapshot_pool_size": 4,
            "selfplay_snapshot_interval": 1,
        },
    )
    return dict(**args["train"], env="puffer_drive", eval=args.get("eval", {}))


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _pairwise_disagreement_ref_learner(pufferl, max_rows=512):
    """Sampled action disagreement between ref (initial weights) and the trained
    learner on the same rollout observations (F2 homogenization heuristic)."""
    import pufferlib.pytorch

    obs_shape = pufferl.vecenv.single_observation_space.shape
    obs = pufferl.observations.reshape(-1, *obs_shape)[:max_rows].float()
    with torch.no_grad():
        learner_logits, _ = pufferl.uncompiled_policy.forward_eval(obs, {})
        ref_logits, _ = pufferl.selfplay.ref_policy.forward_eval(obs, {})
        learner_action, _, _, _ = pufferlib.pytorch.sample_logits(learner_logits.float())
        ref_action, _, _, _ = pufferlib.pytorch.sample_logits(ref_logits.float())
    return float((learner_action != ref_action).float().mean().item())


class _Watchdog:
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._fire)
        signal.alarm(WATCHDOG_SECONDS)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        return False

    @staticmethod
    def _fire(signum, frame):
        raise TimeoutError(f"selfplay smoke test exceeded {WATCHDOG_SECONDS}s watchdog")


def test_drive_selfplay_smoke():
    _seed_everything()
    args = _build_config()

    vecenv = None
    pufferl = None
    with _Watchdog():
        try:
            vecenv = load_env("puffer_drive", args)
            total_agents = vecenv.num_agents
            # Finalize train.device=cpu BEFORE load_policy (golden ordering) so
            # the policy is materialized on cpu; the ref checkpoint is then the
            # learner's initial weights (architecture matches by construction —
            # the real arms load the R champion ckpt).
            ref_dir = os.path.join(REPO_ROOT, "artifacts", "v4-selfplay-smoke")
            os.makedirs(ref_dir, exist_ok=True)
            ref_model_path = os.path.join(ref_dir, "ref_init.pt")
            train_config = _finalize_train_config(args, total_agents, ref_model_path)

            _seed_everything()  # re-seed right before weight init
            policy = load_policy(args, vecenv, "puffer_drive")
            torch.save(policy.state_dict(), ref_model_path)

            pufferl = PuffeRL(train_config, vecenv, policy, logger=_DummyLogger())

            assert pufferl.selfplay is not None, "mixer must be constructed with p_learner=0.5"
            # Measure the env's own blind-agent mask rate in-run (the drive env
            # masks rows regardless of self-play) so G3 compares against the
            # env-aware expectation instead of the nominal partner mix.
            env_mask_ones, env_mask_total = [0], [0]
            original_recv = vecenv.recv

            def recv_accumulating_masks():
                o, r, d, t, info, env_id, mask = original_recv()
                env_mask_ones[0] += int(np.asarray(mask).sum())
                env_mask_total[0] += int(np.asarray(mask).size)
                return o, r, d, t, info, env_id, mask

            vecenv.recv = recv_accumulating_masks
            env_acc = {}
            for _ in range(EPOCHS):
                pufferl.evaluate()
                for k, (value_sum, weight_sum) in pufferl.collect_environment_metric_sums().items():
                    entry = env_acc.setdefault(k, [0.0, 0.0])
                    entry[0] += value_sum
                    entry[1] += weight_sum
                pufferl.last_log_time = 0.0  # force logging so self.losses is deterministic
                pufferl.train()

            losses = {k: float(v) for k, v in pufferl.losses.items() if _is_number(v)}
            env_means = finalize_environment_metrics(env_acc)
        finally:
            if pufferl is not None and hasattr(pufferl, "utilization"):
                try:
                    pufferl.utilization.stop()
                except Exception:
                    pass
            if vecenv is not None:
                try:
                    vecenv.close()
                except Exception:
                    pass

    mixer = pufferl.selfplay
    ego_fraction = float(mixer.ego_agent_mask.float().mean().item())
    partner_fraction = 1.0 - ego_fraction
    env_keep_fraction = env_mask_ones[0] / max(env_mask_total[0], 1)
    # Roles (per-slot categorical) and env blindness (per-step) are independent
    # draws, so P(row kept) = P(env keeps) * P(ego).
    expected_masked_fraction = 1.0 - env_keep_fraction * ego_fraction
    masked_fraction = float(losses["masked_fraction"])
    mask_excess = abs(masked_fraction - expected_masked_fraction)
    loss_finite = all(np.isfinite(v) for v in losses.values())
    disagreement = _pairwise_disagreement_ref_learner(pufferl)
    elapsed_s = pufferl.uptime

    record = {
        "ok": bool(loss_finite and mask_excess < MASK_EXCESS_TOL and mixer.snapshot_writes == EPOCHS),
        "epochs": EPOCHS,
        "loss_finite": bool(loss_finite),
        "ego_row_fraction": ego_fraction,
        "partner_row_fraction": partner_fraction,
        "expected_partner_fraction": P_REF + P_SNAPSHOT,
        "mask_excess": mask_excess,
        "snapshot_pool_writes": int(mixer.snapshot_writes),
        "pairwise_disagreement_ref_learner": disagreement,
        "elapsed_s": elapsed_s,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)

    print("\n[selfplay-smoke] total_agents:", total_agents, "num_envs:", NUM_ENVS)
    print("[selfplay-smoke] losses:", json.dumps(losses, indent=2, sort_keys=True))
    print("[selfplay-smoke] env_means:", json.dumps(env_means, indent=2, sort_keys=True))
    print("[selfplay-smoke] record:", json.dumps(record, indent=2, sort_keys=True))

    # G3: partner rows reached the loss gate at the realized mix ratio.
    assert mask_excess < MASK_EXCESS_TOL, f"masked_fraction {masked_fraction} vs partner fraction {partner_fraction}"
    # G4: finite losses (no NaN/Inf anywhere).
    assert loss_finite, f"non-finite losses: {losses}"
    # G6: interval=1 over 2 epochs -> exactly 2 ring writes.
    assert mixer.snapshot_writes == EPOCHS, f"pool writes {mixer.snapshot_writes} != {EPOCHS}"
    # G1 sanity at integration level: partner slots took non-ego masks.
    assert partner_fraction > 0.0


if __name__ == "__main__":
    test_drive_selfplay_smoke()
    print("Self-play smoke test passed!")
