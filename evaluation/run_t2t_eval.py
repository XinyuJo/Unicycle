#!/usr/bin/env python3
"""End-to-end benchmark runner: inference -> LLM judge -> metrics."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent
INFER_ROOT = ROOT.parent / "inference"


def build_inference_command(args: argparse.Namespace, prediction_path: Path, image_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(INFER_ROOT / "run_t2t_inference.py"),
        "--backend",
        args.backend,
        "--input",
        args.bench,
        "--output",
        str(prediction_path),
        "--out_image_dir",
        str(image_dir),
    ]

    if args.backend == "bagel":
        cmd.extend(["--model_dir", args.model_dir])
    elif args.backend == "janus":
        cmd.extend(
            [
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
        )
        if args.gen_do_sample:
            cmd.append("--gen_do_sample")
    elif args.backend == "showo":
        cmd.extend(["--config", args.config, "--dist_backend", args.dist_backend])
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

    for item in args.inference_extra_args:
        cmd.extend(shlex.split(item))
    return cmd


def build_judge_command(args: argparse.Namespace, prediction_path: Path, judged_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "T2T_llm_judge.py"),
        "--input",
        str(prediction_path),
        "--output",
        str(judged_path),
        "--api_key",
        args.api_key,
        "--model",
        args.judge_model,
        "--temperature",
        str(args.judge_temperature),
        "--top_p",
        str(args.judge_top_p),
        "--timeout_sec",
        str(args.timeout_sec),
        "--max_retries",
        str(args.max_retries),
        "--max_qps",
        str(args.max_qps),
    ]
    if args.base_url:
        cmd.extend(["--base_url", args.base_url])
    for item in args.judge_extra_args:
        cmd.extend(shlex.split(item))
    return cmd


def build_metric_command(judged_path: Path, question_out: Path, prompt_out: Path) -> List[str]:
    return [
        sys.executable,
        str(ROOT / "metrics_t2t.py"),
        "--input",
        str(judged_path),
        "--question_out",
        str(question_out),
        "--prompt_out",
        str(prompt_out),
    ]


def run_and_print(cmd: List[str], cwd: Path) -> None:
    print("Launching:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end T2T benchmark evaluation.")
    parser.add_argument("--backend", choices=["bagel", "janus", "showo"], required=True)
    parser.add_argument("--bench", required=True, help="Benchmark jsonl path.")
    parser.add_argument("--output_dir", required=True, help="Directory for predictions, judged rows, and metrics.")
    parser.add_argument("--run_name", default="", help="Optional output prefix. Defaults to <backend>_<bench-stem>.")

    parser.add_argument("--api_key", required=True, help="OpenAI API key for the judge.")
    parser.add_argument("--base_url", default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--judge_model", default="gpt-4o-mini")
    parser.add_argument("--judge_temperature", type=float, default=0.001)
    parser.add_argument("--judge_top_p", type=float, default=0.95)
    parser.add_argument("--timeout_sec", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--max_qps", type=float, default=1.0)

    parser.add_argument("--model_dir", default="")
    parser.add_argument("--model_path", default="deepseek-ai/Janus-1.3B")
    parser.add_argument("--config", default="")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--cfg_weight", type=float, default=5.0)
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_do_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--flush_every", type=int, default=1)

    parser.add_argument("--dist_backend", default="nccl")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--mmu_max_new_tokens", type=int, default=None)
    parser.add_argument("--mmu_top_k", type=int, default=None)

    parser.add_argument("--inference-extra-args", action="append", default=[], help="Extra args appended to inference.")
    parser.add_argument("--judge-extra-args", action="append", default=[], help="Extra args appended to judge.")
    args = parser.parse_args()

    if args.backend == "bagel" and not args.model_dir:
        parser.error("--model_dir is required when --backend bagel")
    if args.backend == "showo" and not args.config:
        parser.error("--config is required when --backend showo")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_path = Path(args.bench)
    run_name = args.run_name or f"{args.backend}_{bench_path.stem}"

    prediction_path = output_dir / f"{run_name}_predictions.jsonl"
    judged_path = output_dir / f"{run_name}_judged.jsonl"
    question_out = output_dir / f"{run_name}_question_scores.jsonl"
    prompt_out = output_dir / f"{run_name}_prompt_scores.jsonl"
    image_dir = output_dir / f"{run_name}_images"

    run_and_print(build_inference_command(args, prediction_path, image_dir), INFER_ROOT)
    run_and_print(build_judge_command(args, prediction_path, judged_path), ROOT)
    run_and_print(build_metric_command(judged_path, question_out, prompt_out), ROOT)

    print("[DONE]")
    print(f"Predictions : {prediction_path}")
    print(f"Judged rows : {judged_path}")
    print(f"Question out: {question_out}")
    print(f"Prompt out  : {prompt_out}")
    print(f"Images      : {image_dir}")


if __name__ == "__main__":
    main()
