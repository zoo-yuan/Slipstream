#!/usr/bin/env python3
"""Export a PufferDrive Drive policy checkpoint to ONNX.

Faithful reconstruction of the eval-time policy graph (same construction path as
pufferl.load_policy): the checkpoint's own config.yaml drives env + policy
instantiation, so obs/action dims and slot layout always match the weights.

Notes on fidelity:
  - fused_slot_encoder is disabled before export (Triton kernels cannot be
    traced); the dense masked-fill path in DriveBackbone._encode_and_pool is
    mathematically identical (encode all slots -> masked max-pool).
  - The exported graph takes the RAW observation vector (Box[-1,1], C-kernel
    output). PufferLib applies no input normalization at eval (obs_stats are
    monitoring-only), so no preprocessing layer is needed on either side.
  - Outputs: action logits [B, num_classes] (jerk discrete) and value [B, 1].
    Decoding: k = argmax(logits); long_jerk = JERK_LONG[k // 3], lat_jerk = JERK_LAT[k % 3].

Usage:
  PYTHONPATH=$PWD .venv/bin/python scripts/export_onnx.py \
      --run-dir artifacts/t3-R4x/ckpt/R_ent5_off23 \
      --ckpt artifacts/t3-R4x/ckpt/R_ent5_off23/models/model_puffer_drive_007424.pt \
      --out artifacts/onnx/t3-R4x-champ.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CARLA_MAPS = REPO_ROOT / "pufferlib/resources/drive/binaries/carla"


class PolicyOnnx(nn.Module):
    """Flat inference wrapper: obs -> (action_logits, value)."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observations: torch.Tensor):
        actions, value = self.policy(observations)
        if hasattr(actions, "logits"):  # torch.distributions.Categorical
            actions = actions.logits
        return actions, value


def ensure_flat_state_dict(ckpt_path: Path):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj, "flat_state_dict_as_is"
    for key in ("policy_state_dict", "state_dict", "policy", "model"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            inner = obj[key]
            if inner and all(torch.is_tensor(v) for v in inner.values()):
                return inner, f"extracted_from_key={key}"
    raise RuntimeError(f"{ckpt_path}: not a flat state_dict and no known wrapper key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="directory holding config.yaml")
    ap.add_argument("--ckpt", required=True, help="policy .pt (flat state_dict or wrapper)")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--map-dir", default=str(CARLA_MAPS), help="override env map_dir (binaries)")
    args = ap.parse_args()

    from pufferlib.ocean.drive.drive import Drive as DriveEnv
    from pufferlib.ocean.torch import Drive as DrivePolicy

    cfg = yaml.safe_load((Path(args.run_dir) / "config.yaml").read_text())
    env_kwargs = dict(cfg["env"])
    env_kwargs["map_dir"] = args.map_dir  # checkpoint config points at the training host's path

    env = DriveEnv(**env_kwargs)
    policy = DrivePolicy(env, **cfg["policy"])

    state_dict, handling = ensure_flat_state_dict(Path(args.ckpt))
    missing, unexpected = policy.load_state_dict(state_dict, strict=True), None
    policy.eval()

    # Triton fused encoder cannot be traced; dense masked path is math-identical.
    for backbone in (policy.actor_backbone, policy.critic_backbone):
        backbone.fused_slot_encoder = False

    model = PolicyOnnx(policy).eval()
    # Real observations from the env itself: traffic-control type/state columns are
    # integer class indices (not normalized), so random tensors are invalid inputs.
    env.reset(seed=42)
    real_obs = torch.as_tensor(np.array(env.observations), dtype=torch.float32)
    dummy = real_obs[:4].clone()
    probe = real_obs[4:12].clone() if real_obs.shape[0] >= 12 else real_obs.clone()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obs_dim = env.num_obs
    num_classes = int(policy.action_dim)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["observations"],
        output_names=["action_logits", "value"],
        dynamic_axes={"observations": {0: "batch"}, "action_logits": {0: "batch"}, "value": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # ---- validation: checker + onnxruntime numerical parity ----
    import onnx
    onnx.checker.check_model(onnx.load(str(out_path)))

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref_logits, ref_value = model(probe)
    ort_logits, ort_value = sess.run(None, {"observations": probe.numpy()})

    max_diff = max(
        (ref_logits - torch.from_numpy(ort_logits)).abs().max().item(),
        (ref_value - torch.from_numpy(ort_value)).abs().max().item(),
    )
    n_params = sum(p.numel() for p in policy.parameters())
    n_encoders = sum(1 for m in policy.modules() if isinstance(m, nn.Sequential) and hasattr(m, "__len__") and len(m) >= 2 and "encoder" in str(getattr(m, "_name", "")))
    summary = {
        "ckpt": str(args.ckpt),
        "state_dict_handling": handling,
        "config": str(Path(args.run_dir) / "config.yaml"),
        "obs_dim": obs_dim,
        "action_classes": num_classes,
        "params_total": n_params,
        "opset": args.opset,
        "onnx_size_bytes": out_path.stat().st_size,
        "ort_max_abs_diff": max_diff,
        "parity_pass": max_diff < 1e-4,
        "input_name": "observations",
        "output_names": ["action_logits", "value"],
    }
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["parity_pass"]:
        raise SystemExit(f"ONNX/torch parity failed: max abs diff {max_diff}")


if __name__ == "__main__":
    main()
