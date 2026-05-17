#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
import re
import argparse
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# OpenAI SDK (new)
from openai import OpenAI


TIMEOUT_SEC_DEFAULT = 120
MAX_RETRIES_DEFAULT = 6

# QPS throttle
_next_allowed_ts = 0.0


def throttle(max_qps: float):
    global _next_allowed_ts
    if max_qps <= 0:
        return
    now = time.time()
    if now < _next_allowed_ts:
        time.sleep(_next_allowed_ts - now)
    _next_allowed_ts = max(now, _next_allowed_ts) + 1.0 / max_qps


EVAL_PROMPT_SINGLE = r"""
You are a visual QA evaluation assistant.

You will be given:
1) TASK_TYPE: the evaluation dimension to focus on.
2) An IMAGE GENERATION PROMPT describing what the image should contain.
3) ONE QA pair (Question, Answer).

Your task:
Judge whether the Answer is consistent with what the IMAGE_PROMPT implies, focusing ONLY on TASK_TYPE.
Ignore other unrelated details.

Global strict rules:
- Do NOT use external knowledge; rely only on IMAGE_PROMPT text.
- Output "yes" if the answer is consistent with the prompt.
- Output "no" if the answer contradicts the prompt.
- If IMAGE_PROMPT is insufficient to verify the Answer, output "no".
- If the Answer is a refusal (e.g., "can't image from"), output "no".
- Be strict: if the prompt requires a specific detail, vague/generic answers are "no".

Normalization rules (treat as equivalent):
- Case-insensitive.
- Ignore punctuation and extra spaces.
- Ignore articles a/an/the ONLY for non-count tasks (e.g., "adult" == "the adult").
  For count/number tasks, do NOT ignore articles, because "a/an" may imply quantity=1.
- Minor spelling variants (e.g., "vein"=="veining", "veins"=="veining", "grey"=="gray").

Output JSON only with exactly these keys:
{{
  "question": "<question>",
  "answer": "<answer>",
  "evaluation": "yes" or "no"
}}

Now perform the evaluation.

[TASK_TYPE]
{task_type}

[IMAGE_PROMPT]
{image_prompt}

[QA]
Question: {question}
Answer: {answer}
""".strip()


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield line_no, json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSONL parse error at line {line_no}: {e}\n{s[:200]}") from e


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def to_str(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, str):
        return x
    return str(x)


def extract_questions_and_refs(obj: Dict[str, Any]) -> List[Tuple[str, str]]:
    qa_pairs = safe_list(obj.get("qa_pairs"))
    extracted: List[Tuple[str, str]] = []
    for qa in qa_pairs:
        if not isinstance(qa, dict):
            continue
        q = to_str(qa.get("question")).strip()
        ref = to_str(qa.get("answer")).strip()
        if not q:
            continue
        extracted.append((q, ref))
    return extracted


def normalize_answer(ans: str, task_type: str = "") -> str:
    """
    保守归一化：
    - strip 引号
    - lower
    - 仅去掉 'the'
    """
    if ans is None:
        return ""
    s = str(ans).strip()
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1].strip()
    s = s.lower()
    toks = [t for t in s.split() if t not in ("the",)]
    return " ".join(toks).strip()


# -------- JSON 容错解析 --------
def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()

    # remove ``` blocks
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1].strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()

    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        t = t[l:r+1].strip()

    try:
        return json.loads(t)
    except Exception:
        return None


def validate_single(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_dict"
    if set(obj.keys()) != {"question", "answer", "evaluation"}:
        return False, "bad_keys"
    if not isinstance(obj["question"], str) or not isinstance(obj["answer"], str):
        return False, "qa_not_str"
    if obj["evaluation"] not in ("yes", "no"):
        return False, "bad_eval_value"
    return True, "ok"


def normalize_eval(obj: Dict[str, Any], question: str, answer: str) -> Dict[str, Any]:
    ev = (obj.get("evaluation") or "").strip().lower()
    if ev not in ("yes", "no"):
        ev = "no"
    return {"question": question, "answer": answer, "evaluation": ev}


def call_llm_openai(
    client: OpenAI,
    prompt: str,
    model: str,
    max_retries: int,
    timeout_sec: int,
    max_qps: float,
    temperature: float,
    top_p: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    返回: (parsed_json_obj, raw_text, err_str)
    """
    last_raw = None
    last_err = None

    for attempt in range(max_retries):
        try:
            throttle(max_qps)

            # OpenAI SDK: per-request timeout 用 http_client 才能更细控
            # 这里用整体环境超时不做强绑定
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
            )

            raw = (resp.choices[0].message.content or "").strip()
            last_raw = raw

            obj = safe_json_loads(raw)
            ok, reason = validate_single(obj) if obj is not None else (False, "json_parse_fail")
            if ok:
                return obj, raw, None

            sleep_s = min(60.0, 1.0 * (2 ** attempt)) + random.uniform(0, 0.5)
            last_err = f"bad_output_{reason}_sleep_{sleep_s:.2f}"
            time.sleep(sleep_s)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            # 兼容各种网络/限流/服务端错误：指数退避
            sleep_s = min(60.0, 1.0 * (2 ** attempt)) + random.uniform(0, 0.5)
            last_err = f"request_error_{type(e).__name__}_sleep_{sleep_s:.2f}"
            time.sleep(sleep_s)

    return None, last_raw, (last_err or "llm_fail")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True, help="input judge_input.jsonl")
    ap.add_argument("--output", required=True, help="output judged.jsonl")

    # OpenAI
    ap.add_argument("--api_key", required=True, help="OpenAI API key")
    ap.add_argument("--base_url", default=None, help="OpenAI base_url, e.g. https://api.openai.com/v1 or your proxy")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")

    # generation params
    ap.add_argument("--temperature", type=float, default=0.001)
    ap.add_argument("--top_p", type=float, default=0.95)

    # retry / qps
    ap.add_argument("--timeout_sec", type=int, default=TIMEOUT_SEC_DEFAULT)
    ap.add_argument("--max_retries", type=int, default=MAX_RETRIES_DEFAULT)
    ap.add_argument("--max_qps", type=float, default=1.0)

    args = ap.parse_args()

    ensure_dir(args.output)

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,  # None 就默认官方
    )

    total_items = 0
    total_questions = 0
    written = 0
    api_errors = 0

    pbar = tqdm(iter_jsonl(args.input), desc="Items", dynamic_ncols=True)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for _, obj in pbar:
            total_items += 1

            image_prompt = (obj.get("short_description") or "").strip()
            task_type = to_str(obj.get("type") or "").strip()

            qa_pairs = extract_questions_and_refs(obj)
            a_list = safe_list(obj.get("ans_list"))

            n = min(len(qa_pairs), len(a_list))
            for i in range(n):
                total_questions += 1
                q, ref_answer = qa_pairs[i]
                a = to_str(a_list[i]).strip()

                # normalize answer before judging
                a_norm = normalize_answer(a, task_type=task_type)

                prompt = EVAL_PROMPT_SINGLE.format(
                    task_type=task_type,
                    image_prompt=image_prompt,
                    question=q,
                    answer=a_norm,
                )

                out_row = {
                    "type": task_type,
                    "short_description": image_prompt,
                    "q_index": i,
                    "question": q,
                    "reference_answer": ref_answer,
                    "answer": a,
                    "answer_norm": a_norm,
                }

                j, raw, err = call_llm_openai(
                    client=client,
                    prompt=prompt,
                    model=args.model,
                    max_retries=args.max_retries,
                    timeout_sec=args.timeout_sec,
                    max_qps=args.max_qps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                if j is None:
                    api_errors += 1
                    out_row["llm_judge"] = {"question": q, "answer": a_norm, "evaluation": "no"}
                    out_row["llm_error"] = err
                    out_row["llm_raw"] = raw
                else:
                    out_row["llm_judge"] = normalize_eval(j, q, a_norm)
                    out_row["llm_judge_raw"] = j
                    out_row["llm_raw"] = raw

                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1

    print("[DONE]")
    print(f"Items processed          : {total_items}")
    print(f"Questions processed      : {total_questions}")
    print(f"Rows written             : {written}")
    print(f"LLM API errors           : {api_errors}")
    print(f"Output written to        : {args.output}")


if __name__ == "__main__":
    main()
