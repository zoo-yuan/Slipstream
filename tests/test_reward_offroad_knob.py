"""T6: reward_offroad_lo/hi kernel interval knob (reward_randomization branch).

The randomization branch of generate_reward_coefs samples REWARD_COEF_OFFROAD from
the env-level [reward_offroad_lo, reward_offroad_hi] interval (T1 progress special-
case style) instead of the static REWARD_BOUNDS U(0, 3) entry. Contracts proven here:

a. RNG stream equivalence: the default [0, 3] interval reproduces the pre-knob
   kernel's per-agent offroad coefficients bit-for-bit (same draw count, same
   order). Two subprocesses dump N seeds x M agents of coefficients — one with
   PYTHONPATH pinned to the main tree .so (pre-knob oracle, read-only), one to
   this worktree's .so — and their hashes must match exactly.
b. Pinning: lo=hi=3.0 pins every agent's offroad coef to exactly 3.0 while every
   other coef column stays bit-identical to the main-tree default output under
   the same seed (proves only the offroad interval mapping moved).
c. Validation: lo>hi and NaN fail fast at init with ValueError (supervisor ruling
   A: mirrors the reward_progress_lo/hi precedent — Python guard in Drive.__init__
   and the defense-in-depth C binding guard in my_init, each asserted separately).
d. Hot-path discipline: the kernel diff adds allocation-free lines only (struct
   decls + my_init unpack/validate + generate_reward_coefs special case); the
   interval is read from state set at init, so c_step/obs/collision see zero new
   work and zero new mallocs.

Run (repo kernel must shadow the venv copy; main tree .so is the test-a oracle):

    PYTHONPATH=$PWD /home/zoo/labs-y/30-simulation-and-generation/Self-Play/Slipstream/.venv/bin/python \
        -m pytest tests/test_reward_offroad_knob.py -v
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Isolated worktrees live under <repo-parent>/worktrees/Slipstream/<worktree>; the
# main tree (pre-knob oracle for test a/b) is the sibling Slipstream checkout.
MAIN_TREE = REPO_ROOT.parents[2] / "Slipstream"
MAP_BIN_REL = Path("pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin")
OFFROAD_COEF_IDX = 3  # #define REWARD_COEF_OFFROAD 3 (pufferlib/ocean/drive/constants.h:340)
SEEDS = [11, 22, 33]
NUM_AGENTS = 16
NUM_COEFS = 19  # binding.NUM_REWARD_COEFS


# --- guards -------------------------------------------------------------------

def _repo_kernel_probe():
    try:
        import pufferlib
        import pufferlib.ocean.drive.binding as binding
    except Exception as exc:  # noqa: BLE001 - any import failure is a skip reason
        return False, f"pufferlib import failed ({exc})"
    ok = (Path(pufferlib.__file__).resolve().is_relative_to(REPO_ROOT)
          and Path(binding.__file__).resolve().is_relative_to(REPO_ROOT))
    reason = ("repo kernel not on the import path — run with "
              "PYTHONPATH=$PWD <venv python> -m pytest so the worktree kernel "
              "shadows the venv copy")
    return ok, reason


_KERNEL_OK, _KERNEL_SKIP_REASON = _repo_kernel_probe()
requires_kernel = pytest.mark.skipif(not _KERNEL_OK, reason=_KERNEL_SKIP_REASON)


def _main_kernel_so():
    matches = sorted((MAIN_TREE / "pufferlib/ocean/drive").glob("binding.cpython-*.so"))
    return matches[0] if matches else None


_MAIN_MISSING = (
    f"main-tree kernel .so not found under {MAIN_TREE} — test a/b need the "
    "pre-knob .so as the read-only RNG oracle"
)


@pytest.fixture(scope="module")
def map_bin():
    path = REPO_ROOT / MAP_BIN_REL
    if not path.exists():
        pytest.skip(f"map bin missing: {path}")
    return path


@pytest.fixture(scope="module")
def main_map_bin():
    path = MAIN_TREE / MAP_BIN_REL
    if not path.exists():
        pytest.skip(f"main-tree map bin missing: {path}")
    return path


# --- subprocess coefficient dump (the test-a/b oracle protocol) ----------------

_DUMP_PROGRAM = r"""
import hashlib, json, sys
import numpy as np

payload = json.loads(sys.argv[1])

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

off_idx = payload["off_idx"]


def capture_coefs(env, cap):
    n = env.num_envs
    traffic_cap = 256
    agent_f32 = np.zeros((n, cap, binding.AGENT_F32_FIELDS), np.float32)
    agent_i32 = np.zeros((n, cap, binding.AGENT_I32_FIELDS), np.int32)
    metrics_f32 = np.zeros((n, cap, binding.METRICS_F32_FIELDS), np.float32)
    puffer_f32 = np.zeros((n, cap, binding.SCORE_F32_FIELDS), np.float32)
    traffic_i16 = np.zeros((n, traffic_cap, binding.TRAFFIC_I16_FIELDS), np.int16)
    goals_f32 = np.zeros((n, cap, env.num_goals * binding.GOAL_XY_FIELDS), np.float32)
    rewards_f32 = np.zeros((n, cap, binding.REWARD_F32_FIELDS), np.float32)
    coefs_f32 = np.zeros((n, cap, binding.NUM_REWARD_COEFS), np.float32)
    env.get_obs_html_frame(agent_f32, agent_i32, metrics_f32, puffer_f32,
                           traffic_i16, goals_f32, rewards_f32, coefs_f32)
    # rows of live agents only (sim_valid & active_agent; hazard_demo.env_active_mask)
    active = (agent_i32[0][:, 2] == 1) & (agent_i32[0][:, 3] == 1)
    return coefs_f32[0][active]


out = {}
for seed in payload["seeds"]:
    env = Drive(
        map_dir=payload["map_bin"], num_maps=1, num_agents=payload["num_agents"],
        min_agents_per_env=payload["num_agents"], max_agents_per_env=payload["num_agents"],
        scenario_length=100, simulation_mode="gigaflow",
        reward_randomization=True, reward_conditioning=True,
        seed=seed, **payload.get("extra") or {})
    env.reset(seed=seed)
    coefs = capture_coefs(env, payload["num_agents"])
    out[str(seed)] = {
        "n_agents": int(coefs.shape[0]),
        "offroad_raw": coefs[:, off_idx].astype(np.float64).tolist(),
        "offroad_sha": hashlib.sha256(np.ascontiguousarray(coefs[:, off_idx], np.float32).tobytes()).hexdigest(),
        "column_shas": [
            hashlib.sha256(np.ascontiguousarray(coefs[:, c], np.float32).tobytes()).hexdigest()
            for c in range(coefs.shape[1])
        ],
        "all_sha": hashlib.sha256(np.ascontiguousarray(coefs, np.float32).tobytes()).hexdigest(),
    }
print(json.dumps(out))
"""


def _run_kernel_dump(tree_root, map_bin, seeds=SEEDS, num_agents=NUM_AGENTS, extra=None):
    """Dump per-agent reward coefficients from the kernel .so of `tree_root` in a
    subprocess (PYTHONPATH + cwd pin the imported kernel; PYTHONDONTWRITEBYTECODE
    keeps the read-only oracle tree untouched)."""
    payload = {
        "map_bin": str(map_bin),
        "seeds": list(seeds),
        "num_agents": num_agents,
        "off_idx": OFFROAD_COEF_IDX,
        "extra": extra or {},
    }
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _DUMP_PROGRAM, json.dumps(payload)],
        cwd=str(tree_root), env=env, capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"kernel dump subprocess failed under {tree_root}:\n{proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def default_dumps(main_map_bin, map_bin):
    """Main-tree default dump (oracle) + two independent worktree default dumps
    (second one doubles as the deterministic double-run contract)."""
    if _main_kernel_so() is None:
        pytest.skip(_MAIN_MISSING)
    main = _run_kernel_dump(MAIN_TREE, main_map_bin)
    work = _run_kernel_dump(REPO_ROOT, map_bin)
    work_repeat = _run_kernel_dump(REPO_ROOT, map_bin)
    return main, work, work_repeat


@pytest.fixture(scope="module")
def pinned_dump(map_bin):
    return _run_kernel_dump(REPO_ROOT, map_bin, extra={"reward_offroad_lo": 3.0, "reward_offroad_hi": 3.0})


# --- a. RNG stream equivalence -------------------------------------------------

@requires_kernel
class TestRngStreamEquivalence:
    def test_worktree_deterministic_double_run(self, default_dumps):
        _, work, work_repeat = default_dumps
        assert work == work_repeat, "same-tree double run diverged — kernel nondeterminism"

    def test_default_interval_matches_pre_knob_kernel_bit_for_bit(self, default_dumps):
        main, work, _ = default_dumps
        for seed in SEEDS:
            m, w = main[str(seed)], work[str(seed)]
            assert m["n_agents"] == w["n_agents"] == NUM_AGENTS
            assert w["offroad_sha"] == m["offroad_sha"], (
                f"seed {seed}: offroad column diverged "
                f"(main={m['offroad_sha'][:12]} work={w['offroad_sha'][:12]})"
            )
            assert w["all_sha"] == m["all_sha"], f"seed {seed}: full coef matrix diverged"
            # pytest-evidence line for the task md
            print(f"EVIDENCE a: seed={seed} offroad_sha={w['offroad_sha'][:16]} == main (bit-identical)")

    def test_worktree_sixteen_agents_coefficients_in_default_band(self, default_dumps):
        _, work, _ = default_dumps
        for seed in SEEDS:
            vals = np.asarray(work[str(seed)]["offroad_raw"], dtype=np.float64)
            assert vals.shape == (NUM_AGENTS,)
            assert np.all(vals >= 0.0) and np.all(vals <= 3.0)


# --- b. pinning ----------------------------------------------------------------

@requires_kernel
class TestIntervalPinning:
    def test_lo_hi_3_pins_offroad_and_preserves_other_columns(self, default_dumps, pinned_dump):
        main, _, _ = default_dumps
        for seed in SEEDS:
            m, p = main[str(seed)], pinned_dump[str(seed)]
            assert p["n_agents"] == NUM_AGENTS
            raw = np.asarray(p["offroad_raw"], dtype=np.float64)
            assert raw.shape == (NUM_AGENTS,) and np.all(raw == 3.0), (
                f"seed {seed}: pinned coefs not exactly 3.0 (min={raw.min()}, max={raw.max()})"
            )
            assert p["offroad_sha"] != m["offroad_sha"], f"seed {seed}: pin did not move the offroad mapping"
            for c in range(NUM_COEFS):
                if c == OFFROAD_COEF_IDX:
                    continue
                assert p["column_shas"][c] == m["column_shas"][c], (
                    f"seed {seed}: coef column {c} diverged under pin — "
                    "offroad draw must consume the same RNG slot"
                )
            print(f"EVIDENCE b: seed={seed} offroad all==3.0; other 18 column shas == main")

    def test_pinned_matrix_differs_only_in_offroad_column(self, default_dumps, pinned_dump):
        main, _, _ = default_dumps
        for seed in SEEDS:
            assert pinned_dump[str(seed)]["all_sha"] != main[str(seed)]["all_sha"]


# --- c. fail-fast validation (ValueError, per supervisor ruling A) -------------

@requires_kernel
class TestIntervalValidation:
    def test_python_guard_rejects_lo_gt_hi_and_nan(self):
        from pufferlib.ocean.drive.drive import Drive

        with pytest.raises(ValueError, match="reward_offroad_lo/hi"):
            Drive(config_only=True, reward_offroad_lo=1.0, reward_offroad_hi=0.5)
        with pytest.raises(ValueError, match="reward_offroad_lo/hi"):
            Drive(config_only=True, reward_offroad_lo=float("nan"), reward_offroad_hi=1.0)
        print("EVIDENCE c: config_only guard ValueError on lo>hi and NaN")

    def test_live_init_surfaces_valueerror(self, map_bin):
        from pufferlib.ocean.drive.drive import Drive

        base = dict(
            map_dir=str(map_bin), num_maps=1, num_agents=NUM_AGENTS,
            min_agents_per_env=NUM_AGENTS, max_agents_per_env=NUM_AGENTS,
            scenario_length=100, simulation_mode="gigaflow",
        )
        with pytest.raises(ValueError, match="reward_offroad_lo/hi"):
            Drive(**base, reward_offroad_lo=1.0, reward_offroad_hi=0.5)
        print("EVIDENCE c: live Drive init ValueError before any C work")

    def test_c_binding_guard_reaches_my_init(self, map_bin):
        # Defense-in-depth: bypass the Python guard and hand bad kwargs straight
        # to binding.env_init; the C my_init must fail fast with ValueError.
        import pufferlib.ocean.drive.binding as binding
        from pufferlib.ocean.drive.drive import Drive

        env = Drive(
            map_dir=str(map_bin), num_maps=1, num_agents=NUM_AGENTS,
            min_agents_per_env=NUM_AGENTS, max_agents_per_env=NUM_AGENTS,
            scenario_length=100, simulation_mode="gigaflow",
        )
        kwargs = env._env_init_kwargs(map_file=env.map_files[env.map_ids[0]], max_agents=NUM_AGENTS)
        n = NUM_AGENTS  # single env covering the first NUM_AGENTS buffer slots
        buffers = (
            env.observations[:n], env.actions[:n], env.rewards[:n],
            env.terminals[:n], env.truncations[:n], env.masks[:n], 7,
        )
        bad_cases = (
            {"reward_offroad_lo": 1.0, "reward_offroad_hi": 0.5},
            {"reward_offroad_lo": float("nan"), "reward_offroad_hi": 1.0},
            {"reward_offroad_lo": float("inf"), "reward_offroad_hi": float("inf")},
        )
        for bad in bad_cases:
            case_kwargs = dict(kwargs)
            case_kwargs.update(bad)
            with pytest.raises(ValueError, match="reward_offroad_lo/hi"):
                # NOTE: a failed env_init leaks its (tiny) Env allocation by design
                # of env_binding.h env_new; one leak per bad case in a test process.
                binding.env_init(*buffers, **case_kwargs)
        print("EVIDENCE c: C-side my_init ValueError on lo>hi / NaN / inf")


# --- d. hot-path discipline ----------------------------------------------------

@requires_kernel
class TestHotPathDiscipline:
    EXPECTED_ADDED_LINES = {
        # drive.h — Drive struct interval fields (init-time state, never read in hot path)
        "float reward_offroad_lo; // T6 offroad coef randomization sampling interval; default [0, 3] = legacy static U(0, 3) stream",
        "float reward_offroad_hi;",
        # drive.h — generate_reward_coefs special case (spawn/reset-time, allocation-free)
        "if (c == REWARD_COEF_OFFROAD) {",
        "// T6 offroad interval knob: sampled from the env-level [lo, hi] interval instead of",
        "// the static REWARD_BOUNDS entry. The draw stays in the loop slot and unconditional,",
        "// so draw count and order are unchanged and the default [0, 3] interval reproduces",
        "// the legacy U(0, 3) stream bit-for-bit.",
        "agent->reward_coefs[c] = sample_uniform(&env->rng_state, env->reward_offroad_lo, env->reward_offroad_hi);",
        "continue;",
        "}",
        # binding.c — my_init unpack + trust-boundary validation (init only)
        'env->reward_offroad_lo = (float) unpack(kwargs, "reward_offroad_lo");',
        'env->reward_offroad_hi = (float) unpack(kwargs, "reward_offroad_hi");',
        "if (!isfinite(env->reward_offroad_lo) || !isfinite(env->reward_offroad_hi)",
        "|| env->reward_offroad_lo > env->reward_offroad_hi) {",
        'PyErr_SetString(PyExc_ValueError, "reward_offroad_lo/hi must be finite with lo <= hi (default [0, 3] = legacy U(0, 3))");',
        "return -1;",
        "}",
    }

    def test_kernel_diff_is_exactly_the_declared_alloc_free_lines(self):
        proc = subprocess.run(
            ["git", "diff", "HEAD", "--",
             "pufferlib/ocean/drive/drive.h", "pufferlib/ocean/drive/binding.c"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            pytest.skip(f"git diff unavailable: {proc.stderr.strip()[:120]}")
        added = [
            line[1:].strip()
            for line in proc.stdout.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        unexpected = [line for line in added if line not in self.EXPECTED_ADDED_LINES]
        assert not unexpected, f"kernel diff contains undeclared added lines: {unexpected}"
        missing = self.EXPECTED_ADDED_LINES - set(added)
        assert not missing, f"declared kernel lines absent from diff: {missing}"
        alloc_tokens = ("malloc", "calloc", "realloc", "pymem_", "new ")
        offenders = [line for line in added if any(tok in line.lower() for tok in alloc_tokens)]
        assert not offenders, f"allocation tokens added to kernel: {offenders}"
        print(f"EVIDENCE d: kernel diff adds exactly {len(added)} alloc-free lines (struct + my_init + generate_reward_coefs)")

    def test_interval_state_read_only_in_struct_and_generate_reward_coefs(self):
        drive_h = (REPO_ROOT / "pufferlib/ocean/drive/drive.h").read_text(encoding="utf-8")
        # struct decl + single read site in generate_reward_coefs; c_step/obs/collision untouched
        assert drive_h.count("reward_offroad_lo") == 2
        gen_start = drive_h.index("static void generate_reward_coefs")
        gen_end = drive_h.index("\nstatic ", gen_start + 1)
        special_case = drive_h.index("sample_uniform(&env->rng_state, env->reward_offroad_lo", gen_start)
        assert special_case < gen_end, "offroad interval read escaped generate_reward_coefs"


# --- plumb surface -------------------------------------------------------------

@requires_kernel
class TestPlumbSurface:
    def test_defaults_reach_binding_kwargs(self):
        from pufferlib.ocean.drive.drive import Drive

        env = Drive(config_only=True)
        assert env.reward_offroad_lo == 0.0
        assert env.reward_offroad_hi == 3.0
        kwargs = env._env_init_kwargs(map_file="x.bin", max_agents=NUM_AGENTS)
        assert kwargs["reward_offroad_lo"] == 0.0
        assert kwargs["reward_offroad_hi"] == 3.0

    def test_yaml_documents_kernel_interval_knob(self):
        yaml_text = (REPO_ROOT / "pufferlib/config/puffer_drive.yaml").read_text(encoding="utf-8")
        assert "reward_offroad_lo: 0.0" in yaml_text
        assert "reward_offroad_hi: 3.0" in yaml_text
        assert "kernel interval knob, default = legacy U(0,3)" in yaml_text

    def test_config_schema_declares_interval_fields(self):
        from pufferlib.config_schema import DriveEnvConfig

        fields = DriveEnvConfig.__dataclass_fields__
        assert "reward_offroad_lo" in fields
        assert "reward_offroad_hi" in fields

    def test_explicit_interval_passes_through_kwargs(self):
        from pufferlib.ocean.drive.drive import Drive

        env = Drive(config_only=True, reward_offroad_lo=1.0, reward_offroad_hi=2.0)
        kwargs = env._env_init_kwargs(map_file="x.bin", max_agents=NUM_AGENTS)
        assert kwargs["reward_offroad_lo"] == 1.0
        assert kwargs["reward_offroad_hi"] == 2.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
