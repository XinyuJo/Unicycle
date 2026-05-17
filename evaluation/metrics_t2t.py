#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


NO_EVAL_VALUE_DEFAULT = "no_eval_value"


def get_prompt_key(data: Dict[str, Any]) -> str:
    prompt_type = data.get("type", "unknown")
    short_desc = (data.get("short_description", "") or "").strip()
    return f"{prompt_type}|{short_desc}"


def split_prompt_key(k: str) -> Tuple[str, str]:
    if "|" in k:
        t, sd = k.split("|", 1)
        return t, sd
    return k, ""


def calculate_avg(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return round(sum(xs) / len(xs), 4)


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def has_llm_error(data: Dict[str, Any]) -> bool:
    err = data.get("llm_error", None)
    return err is not None and str(err).strip() != ""


def is_no_eval(data: Dict[str, Any], no_eval_value: str) -> bool:
    judge = data.get("llm_judge", {}) or {}
    ev = judge.get("evaluation", None)
    return str(ev).strip() == str(no_eval_value).strip()


def get_item_soft(data: Dict[str, Any]) -> float:
    """
    yes -> 1
    no  -> 0
    text -> correct_words / word_count
    """
    t = str(data.get("type", "")).strip()
    judge = data.get("llm_judge", {}) or {}
    ev = judge.get("evaluation", None)

    if t != "text":
        return 1.0 if str(ev).strip().lower() == "yes" else 0.0

    correct = _to_float(ev)
    wc = _to_float(data.get("word_count"))
    if correct is None or wc is None or wc <= 0:
        return 0.0
    return _clamp01(correct / wc)


def first_pass_collect(infile: str, no_eval_value: str):
    prompt_items = defaultdict(list)
    no_eval_cnt = 0
    llm_error_cnt = 0
    bad_json = 0
    no_eval_keys = set()
    llm_error_keys = set()

    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            if has_llm_error(data):
                llm_error_cnt += 1
                llm_error_keys.add(get_prompt_key(data))
                continue

            if is_no_eval(data, no_eval_value):
                no_eval_cnt += 1
                no_eval_keys.add(get_prompt_key(data))
                continue

            prompt_items[get_prompt_key(data)].append(get_item_soft(data))

    print(
        f"[INFO] First pass: prompts={len(prompt_items)}, "
        f"no_eval_items={no_eval_cnt}, llm_error_items={llm_error_cnt}, bad_json={bad_json}"
    )
    return prompt_items, no_eval_cnt, llm_error_cnt, no_eval_keys, llm_error_keys


def compute_prompt_scores(prompt_items: Dict[str, List[float]], no_eval_keys: set[str], llm_error_keys: set[str]) -> Dict[str, Dict[str, float]]:
    prompt_scores: Dict[str, Dict[str, float]] = {}
    for k, soft_list in prompt_items.items():
        soft = calculate_avg(soft_list) if soft_list else 0.0
        hard = 1.0 if soft == 1.0 else 0.0
        has_no_eval = k in no_eval_keys
        has_llm_error = k in llm_error_keys
        strict_soft = 0.0 if (has_no_eval or has_llm_error) else soft
        strict_hard = 1.0 if strict_soft == 1.0 else 0.0
        prompt_scores[k] = {
            "soft_score": soft,
            "hard_score": hard,
            "strict_soft_score": strict_soft,
            "strict_hard_score": strict_hard,
            "has_no_eval": has_no_eval,
            "has_llm_error": has_llm_error,
        }
    return prompt_scores


def write_outputs(
    infile: str,
    question_out: str,
    prompt_out: str,
    prompt_scores: Dict[str, Dict[str, float]],
):
    with open(infile, "r", encoding="utf-8") as fin, open(question_out, "w", encoding="utf-8") as fq:
        for line in fin:
            data = json.loads(line)
            key = get_prompt_key(data)
            ps = prompt_scores.get(key, {})
            data["soft_score"] = ps.get("soft_score", 0.0)
            data["hard_score"] = ps.get("hard_score", 0.0)
            data["strict_soft_score"] = ps.get("strict_soft_score", 0.0)
            data["strict_hard_score"] = ps.get("strict_hard_score", 0.0)
            data["has_no_eval"] = ps.get("has_no_eval", False)
            data["has_llm_error"] = ps.get("has_llm_error", False)
            fq.write(json.dumps(data, ensure_ascii=False) + "\n")

    with open(prompt_out, "w", encoding="utf-8") as fp:
        for k, s in prompt_scores.items():
            t, sd = split_prompt_key(k)
            row = {
                "type": t,
                "short_description": sd,
                **s,
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[INFO] Question-level output: {question_out}")
    print(f"[INFO] Prompt-level output  : {prompt_out}")


def print_table(prompt_scores: Dict[str, Dict[str, float]]):
    by_type = defaultdict(list)

    for k, s in prompt_scores.items():
        t, _ = split_prompt_key(k)
        by_type[t].append(s)

    print("\n" + "=" * 110)
    print("Prompt-level Evaluation Table")
    print("-" * 110)
    print(f"{'Type':<25} {'#Prompt':<10} {'Soft':<10} {'Hard':<10} {'Strict-Soft':<15} {'Strict-Hard':<15}")
    print("-" * 110)

    for t in sorted(by_type.keys()):
        rows = by_type[t]
        print(
            f"{t:<25} "
            f"{len(rows):<10} "
            f"{calculate_avg([r['soft_score'] for r in rows]):<10} "
            f"{calculate_avg([r['hard_score'] for r in rows]):<10} "
            f"{calculate_avg([r['strict_soft_score'] for r in rows]):<15} "
            f"{calculate_avg([r['strict_hard_score'] for r in rows]):<15}"
        )

    print("-" * 110)

    all_rows = list(prompt_scores.values())
    print(
        f"{'OVERALL':<25} "
        f"{len(all_rows):<10} "
        f"{calculate_avg([r['soft_score'] for r in all_rows]):<10} "
        f"{calculate_avg([r['hard_score'] for r in all_rows]):<10} "
        f"{calculate_avg([r['strict_soft_score'] for r in all_rows]):<15} "
        f"{calculate_avg([r['strict_hard_score'] for r in all_rows]):<15}"
    )
    print("=" * 110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="llm_judge output jsonl")
    ap.add_argument("--question_out", required=True, help="question-level output jsonl")
    ap.add_argument("--prompt_out", required=True, help="prompt-level output jsonl")
    ap.add_argument("--no_eval_value", default=NO_EVAL_VALUE_DEFAULT)
    args = ap.parse_args()

    prompt_items, no_eval_cnt, llm_error_cnt, no_eval_keys, llm_error_keys = first_pass_collect(args.input, args.no_eval_value)
    prompt_scores = compute_prompt_scores(prompt_items, no_eval_keys, llm_error_keys)

    write_outputs(
        infile=args.input,
        question_out=args.question_out,
        prompt_out=args.prompt_out,
        prompt_scores=prompt_scores,
    )

    print_table(prompt_scores)

    print("\n[Filter statistics]")
    print(f"  no_eval items : {no_eval_cnt}")
    print(f"  llm_error items: {llm_error_cnt}")


if __name__ == "__main__":
    main()
