#!/usr/bin/env python3
"""Unified launcher for the open-source Unicycle T2T benchmark."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent


def build_command(args: argparse.Namespace) -> List[str]:
    common = [
        "--input",
        args.input,
        "--output",
        args.output,
        "--out_image_dir",
        args.out_image_dir,
    ]

    if args.backend == "bagel":
        script = ROOT / "t2t_bagel_inference.py"
        cmd = [
            sys.executable,
            str(script),
            *common,
            "--model_dir",
            args.model_dir,
            "--flush_every",
            str(args.flush_every),
        ]
    elif args.backend == "janus":
        script = ROOT / "t2t_janus_inference.py"
        cmd = [
            sys.executable,
            str(script),
            *common,
            "--model_path",
            args.model_path,
            "--img_size",
            str(args.img_size),
            "--cfg_weight",
            str(args.cfg_weight),
            "--gen_temperature",
            str(args.gen_temperature),
            "--seed",
            str(args.seed),
            "--flush_every",
            str(args.flush_every),
        ]
        if args.gen_do_sample:
            cmd.append("--gen_do_sample")
    elif args.backend == "showo":
        script = ROOT / "t2t_showo_inference.py"
        cmd = [
            sys.executable,
            str(script),
            "--config",
            args.config,
            *common,
            "--backend",
            args.dist_backend,
        ]
        if args.batch_size is not None:
            cmd.extend(["--batch_size", str(args.batch_size)])
        if args.guidance_scale is not None:
            cmd.extend(["--guidance_scale", str(args.guidance_scale)])
        if args.num_inference_steps is not None:
            cmd.extend(["--num_inference_steps", str(args.num_inference_steps)])
        if args.mmu_max_new_tokens is not None:
            cmd.extend(["--mmu_max_new_tokens", str(args.mmu_max_new_tokens)])
        if args.mmu_top_k is not None:
            cmd.extend(["--mmu_top_k", str(args.mmu_top_k)])
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    for item in args.extra_args:
        cmd.extend(shlex.split(item))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified T2T launcher for the open-source benchmark package.")
    parser.add_argument("--backend", choices=["bagel", "janus", "showo"], required=True)
    parser.add_argument("--input", required=True, help="Benchmark jsonl path.")
    parser.add_argument("--output", required=True, help="Output jsonl path.")
    parser.add_argument("--out_image_dir", required=True, help="Directory to save generated images.")
    parser.add_argument("--extra-args", action="append", default=[], help="Extra backend-specific arguments.")

    parser.add_argument("--model_path", default="deepseek-ai/Janus-1.3B")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--cfg_weight", type=float, default=5.0)
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_do_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--model_dir", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--flush_every", type=int, default=1)

    parser.add_argument("--dist_backend", default="nccl")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--mmu_max_new_tokens", type=int, default=None)
    parser.add_argument("--mmu_top_k", type=int, default=None)
    args = parser.parse_args()

    if args.backend == "bagel" and not args.model_dir:
        parser.error("--model_dir is required when --backend bagel")
    if args.backend == "showo" and not args.config:
        parser.error("--config is required when --backend showo")

    cmd = build_command(args)
    print("Launching:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
