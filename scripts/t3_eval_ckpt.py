#!/usr/bin/env python3
"""T3 checkpoint 对比评测 runner：同一 face 同一场景数，串行跑多个 checkpoint 的
pufferl eval，收集各自 episode_metrics.csv，汇总逐指标均值对照表（summary.json +
index.html）。

Owner 批准的首跑（T3 早停判定用）：e168 best vs e1682 final，carla_fast 干净面
400 场景；--mix-face 则换 hazard mix 面（hazard 四开 + cosim_partner_slots 提到
hazard_slot_capacity）。真跑由 supervisor 发起，本脚本自测通过即可交付。

与 pufferl eval 的对接事实（读自 pufferlib/pufferl.py eval() +
pufferlib/ocean/evaluation_utils/evaluation_utils.py，2026-11-10）：
  - 命令形态：.venv/bin/python -m pufferlib.pufferl eval puffer_drive <benchmark> key=value…
    main() pop 掉 mode/env/benchmark 后，剩余 argv 全部成为 Hydra override，并在
    _finalize_benchmark_args 里以 OmegaConf dotlist **最后合并** —— 即 CLI `env.*`
    落在 benchmark env 之上（layering：Hydra base → checkpoint config.yaml env →
    benchmark.yaml 顶层 env → benchmark.env → eval.* 官方 override → CLI key=value）。
    本脚本的面控制机制 = CLI Hydra override（不改 benchmark.yaml）。
  - load_model_path 必须是顶层 policy state_dict（load_policy 直接 torch.load 后
    load_state_dict）。best_models/best_trainer_state_000168.pt 实测虽名带
    trainer_state、实为 flat state_dict，可直接喂；脚本仍带防御性抽取分支。
  - checkpoint 架构/obs 布局来自 <run_dir>/config.yaml（resolve_run_dir：.pt 同目录
    无 config.yaml 则取上一级）。注意：C_recipe_v1 训练 config env 带 hazard_enable=true
    + cosim_partner_slots=44，benchmark.yaml env 不重置这些键 —— 干净面必须显式
    CLI env.hazard_enable=false env.cosim_partner_slots=0 压回，否则「干净面」名不副实。
  - 场景数覆盖 = eval.num_scenarios=N（官方 override）。
  - 输出目录 = os.path.join(run_dir, eval.output_dir_name)：给绝对路径即整体重定向；
    结构 <output_dir_name>/<benchmark>/<output_subdir>/{episode_metrics.csv,
    evaluation_summary.json, resolved_benchmark.yaml}（os.makedirs 无 exist_ok，
    本脚本先清空 <out>/<label> 保证可重跑）。

用法（repo 根）：
  PYTHONPATH=$PWD .venv/bin/python scripts/t3_eval_ckpt.py \
      --models artifacts/t2-haztrain/ckpt/C_recipe_v1/best_models/best_trainer_state_000168.pt \
               artifacts/t2-haztrain/ckpt/C_recipe_v1/models/model_puffer_drive_001682.pt \
      --benchmark carla_fast --num-scenarios 400 --out artifacts/t3-eval
  # mix 面：加 --mix-face；自定义名：--label e168_best e1682_final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def hardware_threads() -> int:
    """与 pufferlib/vector.py MultiprocessingVC 的 num_workers 上限同源：psutil 逻辑核数。
    eval 的 worker 数 = min(vec.num_envs, num_scenarios)，场景数大时会超线程数直接
    APIUsageError；本脚本统一压到线程数（各模型同值，不影响 face/场景集一致性）。"""
    try:
        import psutil

        return psutil.cpu_count(logical=True) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "t3-eval"
PUFFER_DRIVE_YAML = REPO_ROOT / "pufferlib" / "config" / "puffer_drive.yaml"

# hazard_slot_capacity 依赖的全部旋钮（缺哪个都从 puffer_drive.yaml 默认补齐）
HAZARD_KNOB_KEYS = (
    "hazard_enable",
    "hazard_parked_enable",
    "hazard_parked_n_hi",
    "hazard_clusters_enable",
    "hazard_clusters_n_hi",
    "hazard_cones_enable",
    "hazard_cones_zones_hi",
    "hazard_cones_taper_cones",
    "hazard_cones_block_rows",
    "hazard_cones_block_cols",
    "hazard_walkers_enable",
    "hazard_walkers_n_hi",
)
MIX_HAZARD_FLAGS = {
    "hazard_enable": True,
    "hazard_parked_enable": True,
    "hazard_clusters_enable": True,
    "hazard_cones_enable": True,
    "hazard_walkers_enable": True,
}

# 汇总表头部优先展示的判读指标（列名以 eval 实际输出为准，见摸底 episode_metrics.csv）
HEADLINE_METRICS = (
    "episode_return",
    "collision_rate",
    "at_fault_collision_rate",
    "offroad_rate",
    "red_light_violation_rate",
    "stop_sign_violation_rate",
    "num_goals_reached",
    "progress_ratio",
    "making_progress_rate",
    "comfort_score",
    "dnf_rate",
    "score",
    "puffer_score",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def verify_in_repo_kernel() -> dict:
    """内核钉死纪律（同 t2_sweep 用法：从 hazard_demo import，provenance 进 summary）。"""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import hazard_demo as hd  # repo scripts dir; provides verify/provenance helpers

    hd.verify_in_repo_kernel()
    prov = hd.kernel_provenance()
    c_so = REPO_ROOT / "pufferlib" / "_C.cpython-311-x86_64-linux-gnu.so"
    if not c_so.exists():
        raise RuntimeError(f"in-repo advantage kernel required: {c_so} missing")
    prov["advantage_so"] = str(c_so)
    prov["advantage_sha256"] = hashlib.sha256(c_so.read_bytes()).hexdigest()
    import torch

    prov["torch"] = torch.__version__
    prov["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return prov


def default_label(model_path: Path) -> str:
    """best_trainer_state_000168.pt → e168_best；model_puffer_drive_001682.pt → e1682_final；
    其余退化为文件 stem。"""
    stem = model_path.stem
    m = re.fullmatch(r"(best_trainer_state|model_puffer_drive)_(\d+)", stem)
    if m:
        kind = "best" if m.group(1) == "best_trainer_state" else "final"
        return f"e{int(m.group(2))}_{kind}"
    return stem


def resolve_run_dir(model_path: Path) -> Path:
    """同 evaluation_utils.resolve_run_dir：.pt 同目录无 config.yaml 则上一级。"""
    ckpt_dir = model_path.parent
    if (ckpt_dir / "config.yaml").is_file():
        return ckpt_dir
    return ckpt_dir.parent


def ensure_policy_state_dict(model_path: Path, extract_dir: Path, label: str) -> tuple[Path, str]:
    """eval 只吃顶层 state_dict。flat tensor dict 原样喂；否则从已知包装键抽取落盘
    （抽到 <out>/_extracted/<label>.pt），并返回（喂给 eval 的路径, 格式处理说明）。"""
    import torch

    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return model_path, "flat_state_dict_as_is"
    for key in ("policy_state_dict", "state_dict", "policy", "model"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            inner = obj[key]
            if inner and all(torch.is_tensor(v) for v in inner.values()):
                extract_dir.mkdir(parents=True, exist_ok=True)
                out_path = extract_dir / f"{label}.pt"
                torch.save(inner, out_path)
                return out_path, f"extracted_from_trainer_state_key={key}"
    raise RuntimeError(
        f"{model_path} 不是顶层 state_dict，也未含已知包装键"
        "（policy_state_dict/state_dict/policy/model），无法喂 pufferl eval load_model_path"
    )


def hazard_slot_capacity(env_cfg: dict) -> int:
    """Mirror of Drive.hazard_slot_capacity（原样拷自 t2_haztrain.py，勿跨脚本 import）:
    planned-count highs of the hazard face, computed from the *final* env config so the
    slot budget always fits the knobs."""
    if not env_cfg["hazard_enable"]:
        return 0
    cap = 0
    if env_cfg["hazard_parked_enable"]:
        cap += env_cfg["hazard_parked_n_hi"]
    if env_cfg["hazard_clusters_enable"]:
        cap += 2 * env_cfg["hazard_clusters_n_hi"]
    if env_cfg["hazard_cones_enable"]:
        cap += env_cfg["hazard_cones_zones_hi"] * max(
            env_cfg["hazard_cones_taper_cones"],
            env_cfg["hazard_cones_block_rows"] * env_cfg["hazard_cones_block_cols"],
        )
    if env_cfg["hazard_walkers_enable"]:
        cap += env_cfg["hazard_walkers_n_hi"]
    return cap


def mix_face_slots(model_path: Path) -> tuple[int, dict]:
    """--mix-face 的 cosim_partner_slots = hazard_slot_capacity（hazard 四开，旋钮取
    checkpoint config.yaml env，缺项补 puffer_drive.yaml 默认——与 eval 子进程实际生效
    的 knob 来源一致）。返回（slots, 记录块）。"""
    ckpt_env = yaml.safe_load((resolve_run_dir(model_path) / "config.yaml").read_text())["env"]
    defaults_env = yaml.safe_load(PUFFER_DRIVE_YAML.read_text())["env"]
    env_cfg = {k: defaults_env[k] for k in HAZARD_KNOB_KEYS}
    env_cfg.update({k: ckpt_env[k] for k in HAZARD_KNOB_KEYS if k in ckpt_env})
    env_cfg.update(MIX_HAZARD_FLAGS)
    slots = hazard_slot_capacity(env_cfg)
    knobs = {k: env_cfg[k] for k in HAZARD_KNOB_KEYS}
    record = {
        "cosim_partner_slots": slots,
        "knob_source": str(resolve_run_dir(model_path) / "config.yaml"),
        "defaults_fallback": PUFFER_DRIVE_YAML.name,
        "knobs": knobs,
    }
    return slots, record


def build_eval_overrides(model_path: Path, face: str, num_scenarios: int, out_label_dir: Path,
                         output_subdir: str) -> tuple[list[str], dict]:
    """单模型一次 eval 子进程的 key=value 覆盖（机制：CLI Hydra override，
    _finalize_benchmark_args 最后合并 → env.* 压在 benchmark env 之上）。"""
    overrides = [
        f"load_model_path={model_path}",
        f"eval.num_scenarios={num_scenarios}",
        f"eval.output_dir_name={out_label_dir}",
        f"eval.output_subdir={output_subdir}",
        # worker 数 = min(vec.num_envs, num_scenarios)，上限硬件线程数（vector.py fail-fast）
        f"vec.num_envs={min(hardware_threads(), num_scenarios)}",
    ]
    face_record: dict
    if face == "mix":
        slots, face_record = mix_face_slots(model_path)
        overrides += [
            "env.hazard_enable=true",
            "env.hazard_parked_enable=true",
            "env.hazard_clusters_enable=true",
            "env.hazard_cones_enable=true",
            "env.hazard_walkers_enable=true",
            f"env.cosim_partner_slots={slots}",
        ]
        face_record["overrides"] = "hazard 四开 + cosim_partner_slots=hazard_slot_capacity"
    else:
        # 训练 config env 带 hazard_enable=true（benchmark.yaml env 不重置），干净面须显式压回
        overrides += ["env.hazard_enable=false", "env.cosim_partner_slots=0"]
        face_record = {
            "cosim_partner_slots": 0,
            "overrides": "env.hazard_enable=false + env.cosim_partner_slots=0 "
                         "（压回 checkpoint config env 训练时带入的 hazard 面）",
        }
    return overrides, face_record


def run_eval_subprocess(benchmark: str, overrides: list[str], log_path: Path) -> int:
    """串行 eval 子进程；stdout/stderr 流式落 log；返回 rc。GPU 子进程带
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [str(PYTHON_BIN), "-m", "pufferlib.pufferl", "eval", "puffer_drive", benchmark] + overrides
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"eval cmd: PYTHONPATH={REPO_ROOT} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        + " ".join(cmd))
    with log_path.open("w") as lf:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
        rc = proc.wait()
    return rc


def load_model_means(result_dir: Path) -> tuple[dict, int, int]:
    """读 <dir>/evaluation_summary.json 的 metrics_mean（上游 reduce_environment_metrics
    语义，覆盖全部数值列 + 派生列），缺 JSON 时退回 episode_metrics.csv 朴素均值。"""
    summary_path = result_dir / "evaluation_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        return dict(summary["metrics_mean"]), int(summary["num_scenarios"]), int(summary["num_episodes"])
    import pandas as pd

    df = pd.read_csv(result_dir / "episode_metrics.csv")
    numeric = df.drop(columns=["seed"], errors="ignore").select_dtypes(include=["number"])
    return {k: float(v) for k, v in numeric.mean().items()}, len(df), len(df)


def fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_html(out_path: Path, ctx: dict) -> None:
    """简洁表格页，样式参照 artifacts/t2-hazsmoke/index.html。"""
    labels = ctx["labels"]
    comparison = ctx["comparison"]  # metric -> {label: mean}
    ordered = list(comparison.keys())
    headline = [m for m in HEADLINE_METRICS if m in comparison]
    rest = [m for m in ordered if m not in headline]
    two = len(labels) == 2

    def table(metrics: list[str], title: str) -> str:
        rows = []
        delta_col = "<th>Δ ({} − {})</th>".format(labels[1], labels[0]) if two else ""
        for metric in metrics:
            vals = comparison[metric]
            tds = "".join(f"<td>{fmt_val(vals.get(l, '—'))}</td>" for l in labels)
            delta = ""
            if two and all(isinstance(vals.get(l), (int, float)) for l in labels):
                delta = f"<td>{vals[labels[1]] - vals[labels[0]]:+.4f}</td>"
            rows.append(f"<tr><td>{metric}</td>{tds}{delta}</tr>")
        return (f"<h2>{title}</h2>\n<table><tr><th>metric</th>"
                + "".join(f"<th>{l}</th>" for l in labels) + delta_col + "</tr>\n"
                + "\n".join(rows) + "\n</table>")

    model_rows = "".join(
        f"<tr><td>{m['label']}</td><td><code>{m['eval_model_path']}</code></td>"
        f"<td>{m['format_handling']}</td><td>{m['num_episodes']}</td>"
        f"<td>{m['rc']}</td><td><code>{m['log_path']}</code></td></tr>"
        for m in ctx["models"]
    )
    repro = "\n\n".join(m["repro_cmd"] for m in ctx["models"])
    prov = ctx["kernel"]
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>T3 checkpoint 对比 — {ctx['benchmark']} × {ctx['num_scenarios']} scenarios × {ctx['face']} face</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2rem; color: #222; max-width: 1080px; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 1.6rem; }}
table {{ border-collapse: collapse; margin: .6rem 0; font-size: .92rem; }}
td, th {{ border: 1px solid #ccc; padding: .3rem .6rem; }}
code {{ background: #f4f4f4; padding: .5rem; display: block; white-space: pre-wrap; word-break: break-all; }}
ul {{ font-size: .92rem; }}
</style></head><body>
<h1>T3 checkpoint 对比评测 — {ctx['benchmark']} × {ctx['num_scenarios']} scenarios × {ctx['face']} face</h1>
<p>同 face 同场景数串行评测多个 checkpoint，唯一差异 = load_model_path；seed=42（benchmark 固定）。
首跑目的（owner 批准）：e168 best vs e1682 final 定早停。汇总判读以 headline 表为先，
全列均值备查（语义 = 上游 evaluation_summary.json 的 metrics_mean）。</p>
{table(headline, "headline 指标对照")}
{table(rest, "全列均值（其余数值列）")}
<h2>模型配置</h2>
<table><tr><th>label</th><th>eval 喂入权重</th><th>格式处理</th><th>episodes</th><th>rc</th><th>eval log</th></tr>
{model_rows}
</table>
<h2>face 控制（CLI Hydra override，压在 benchmark env 之上）</h2>
<p>{ctx['face_note']}</p>
<h2>kernel provenance</h2>
<ul>
<li>binding_sha256: <code>{prov['binding_sha256']}</code></li>
<li>git_sha: {prov['git_sha']} · torch: {prov['torch']} · gpu: {prov['gpu']}</li>
<li>generated_at: {ctx['generated_at']}</li>
</ul>
<h2>repro（每模型一条）</h2>
<code>{repro}</code>
</body></html>
"""
    out_path.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser(description="T3 checkpoint 对比评测 runner")
    ap.add_argument("--models", nargs="+", required=True, help="checkpoint .pt 路径（顶层 state_dict 或可抽取）")
    ap.add_argument("--label", nargs="+", default=None, help="与 --models 等长的别名")
    ap.add_argument("--benchmark", default="carla_fast")
    ap.add_argument("--num-scenarios", type=int, default=400)
    ap.add_argument("--mix-face", action="store_true",
                    help="hazard 四开 + cosim_partner_slots=hazard_slot_capacity（否则干净面：显式 hazard off）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = out_dir / "_extracted"
    logs_dir = out_dir / "logs"
    face = "mix" if args.mix_face else "clean"
    output_subdir = f"{face}_n{args.num_scenarios}"

    if args.label is not None and len(args.label) != len(args.models):
        ap.error("--label 数量须与 --models 一致")

    kernel = verify_in_repo_kernel()
    log(f"kernel pinned: binding={kernel['binding_sha256'][:12]}… gpu={kernel['gpu']}")

    models: list[dict] = []
    for i, raw in enumerate(args.models):
        model_path = Path(raw).resolve()
        if not model_path.is_file() or model_path.suffix != ".pt":
            raise RuntimeError(f"checkpoint 不存在或非 .pt: {model_path}")
        label = args.label[i] if args.label else default_label(model_path)
        run_dir = resolve_run_dir(model_path)
        if not (run_dir / "config.yaml").is_file():
            raise RuntimeError(f"run dir 无 config.yaml（eval 需要它恢复架构）: {run_dir}")

        eval_model_path, format_handling = ensure_policy_state_dict(model_path, extract_dir, label)
        out_label_dir = out_dir / label
        if out_label_dir.exists():
            shutil.rmtree(out_label_dir)  # os.makedirs(output_dir) 无 exist_ok，重跑须先清
        overrides, face_record = build_eval_overrides(
            model_path, face, args.num_scenarios, out_label_dir, output_subdir
        )
        log_path = logs_dir / f"{label}_{output_subdir}.log"
        log(f"[{label}] 开始 eval：benchmark={args.benchmark} n={args.num_scenarios} face={face}")
        rc = run_eval_subprocess(args.benchmark, overrides, log_path)
        if rc != 0:
            raise RuntimeError(f"[{label}] eval 子进程 rc={rc}≠0，日志: {log_path}")
        result_dir = out_label_dir / args.benchmark / output_subdir
        csv_path = result_dir / "episode_metrics.csv"
        if not csv_path.is_file():
            raise RuntimeError(f"[{label}] 未生成 episode_metrics.csv: {csv_path}")
        means, num_scenarios, num_episodes = load_model_means(result_dir)
        env_cmd = (f"PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                   f".venv/bin/python -m pufferlib.pufferl eval puffer_drive {args.benchmark} "
                   + " ".join(overrides))
        models.append({
            "label": label,
            "model_path": str(model_path),
            "eval_model_path": str(eval_model_path),
            "format_handling": format_handling,
            "run_dir": str(run_dir),
            "rc": rc,
            "log_path": str(log_path.relative_to(REPO_ROOT)) if log_path.is_relative_to(REPO_ROOT) else str(log_path),
            "csv_path": str(csv_path),
            "num_scenarios": num_scenarios,
            "num_episodes": num_episodes,
            "overrides": overrides,
            "face": face_record,
            "repro_cmd": env_cmd,
            "means": means,
        })
        log(f"[{label}] 完成：episodes={num_episodes} episode_return={means.get('episode_return')!r} "
            f"puffer_score={means.get('puffer_score')!r}")

    # 汇总对比表：headline 先行，其余按首个模型 CSV/summary 列序
    comparison: dict[str, dict] = {}
    ordered_metrics: list[str] = []
    for m in models:
        for k in m["means"]:
            if k not in ordered_metrics:
                ordered_metrics.append(k)
    for metric in HEADLINE_METRICS:
        if metric in ordered_metrics:
            comparison[metric] = {m["label"]: m["means"].get(metric) for m in models}
    for metric in ordered_metrics:
        if metric not in comparison:
            comparison[metric] = {m["label"]: m["means"].get(metric) for m in models}

    face_note = models[0]["face"]["overrides"] + (
        "；mix 面 slots 按 checkpoint config.yaml 旋钮逐模型计算" if face == "mix" else ""
    )
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kernel": kernel,
        "benchmark": args.benchmark,
        "num_scenarios": args.num_scenarios,
        "face": face,
        "face_note": face_note,
        "models": [{k: v for k, v in m.items() if k != "means"} for m in models],
        "comparison": comparison,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    render_html(out_dir / "index.html", {
        "labels": [m["label"] for m in models],
        "comparison": comparison,
        "models": models,
        "kernel": kernel,
        "benchmark": args.benchmark,
        "num_scenarios": args.num_scenarios,
        "face": face,
        "face_note": face_note,
        "generated_at": summary["generated_at"],
    })
    log(f"summary → {summary_path}")
    log(f"index   → {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
