import argparse
import json
import os
import re
import time
from typing import Any, Dict, Iterator, List, Tuple, Optional

import numpy as np
import PIL.Image
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


# ---------------------------
# torchrun / distributed utils
# ---------------------------
def _dist_is_launched() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def dist_init(backend: str = "nccl") -> Tuple[int, int, int, bool]:
    """
    Returns: (rank, local_rank, world_size, enabled)
    """
    if not _dist_is_launched():
        return 0, 0, 1, False

    import torch.distributed as dist
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()
    return rank, local_rank, world_size, True


def shard_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    base = total // world_size
    rem = total % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def merge_rank_outputs(final_path: str, world_size: int) -> None:
    tmp_paths = [f"{final_path}.rank{r}.jsonl" for r in range(world_size)]
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    with open(final_path, "w", encoding="utf-8") as w:
        for p in tmp_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as r:
                for line in r:
                    w.write(line)


# ---------------------------
# jsonl utils
# ---------------------------
def iter_jsonl(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield line_no, json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSONL parse error at line {line_no}: {e}\nHead: {s[:200]}") from e


def count_nonempty_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def sanitize_filename(name: str, max_len: int = 96) -> str:
    name = re.sub(r"\s+", "_", (name or "").strip())
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    return (name[:max_len] if len(name) > max_len else name) or "image"


# ---------------------------
# question templates (optional but helps)
# ---------------------------
RE_MCQ = re.compile(r"(\b[A-Z]\s*\.|\(\s*[A-Z]\s*\))", flags=re.IGNORECASE)


def is_multi_choice(question: str) -> bool:
    if not isinstance(question, str):
        return False
    return bool(RE_MCQ.search(question))


VQA_PROMPT_TEMPLATE = """You are a visual understanding assistant.

You are given:
- An image (provided as [input_image])
- One question about the image (provided as [QUESTION])

Your task:
Answer the question using only the visual information present in the image.

Guidelines:
- Base your answers strictly on what can be observed in the image.
- Do not rely on external knowledge, assumptions, or common sense beyond the image.
- Keep answers as short as possible. Prefer a single word or a short phrase whenever it fully answers the question.

Output format:
- Do not include any extra text, numbering, or explanations.

[QUESTION]
{question}"""

VQA_PROMPT_TEMPLATE_MULTI_CHOICE="""You are a visual understanding assistant.

You are given:
- An image (provided as [input_image])
- One question about the image (provided as [QUESTION]).The answer options (A, B, C, D) are included in the question text.

Your task:
Answer the question using only the visual information present in the image.
You MUST answer with ONLY the option letter (A, B, C, or D).
 

Guidelines:
- Base your answers strictly on what can be observed in the image.
- Do not rely on external knowledge, assumptions, or common sense beyond the image.
- Do NOT output the option text.
- Do NOT output anything except the single letter.

Output format:
- Do not include any extra text, numbering, or explanations.

[QUESTION]
{question}"""

VQA_PROMPT_TEMPLATE_NEGATION = """You are a visual understanding assistant.

You are given:
- An image (provided as [input_image])
- One yes-or-no question about the image (provided as [QUESTION])

Your task:
Answer each question using only the visual information present in the image.
You MUST answer with exactly one word: "yes" or "no".

Guidelines:
- Base your answers strictly on what can be observed in the image.
- Do not rely on external knowledge, assumptions, or common sense beyond the image.
- Output ONLY "yes" or "no".

Output format:
- Do not include any extra text, numbering, or explanations.

[QUESTION]
{question}"""



def pick_vqa_prompt(question: str, task_type: str = "", key_ans: str = "") -> str:
    q = (question or "").strip()
    t = (task_type or "").strip().lower()
    ka = (key_ans or "").strip().lower()

    if is_multi_choice(q):
        return VQA_PROMPT_TEMPLATE_MULTI_CHOICE.format(question=q)

    if t == "negation" or ka in ("yes", "no"):
        return VQA_PROMPT_TEMPLATE_NEGATION.format(question=q)

    return VQA_PROMPT_TEMPLATE.format(question=q)


# ---------------------------
# Janus: image generation (CFG on token space)
# ---------------------------
@torch.inference_mode()
def janus_generate_image(
    mmgpt: MultiModalityCausalLM,
    processor: VLChatProcessor,
    prompt_text: str,
    *,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    num_images: int = 1,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    do_sample: bool = True,
    seed: Optional[int] = None,
) -> List[PIL.Image.Image]:

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # encode prompt
    input_ids = processor.tokenizer.encode(prompt_text)
    input_ids = torch.LongTensor(input_ids).cuda()

    # CFG batching: [cond, uncond, cond, uncond, ...] => 2*num_images
    bs2 = num_images * 2
    tokens = torch.zeros((bs2, len(input_ids)), dtype=torch.long, device="cuda")
    for i in range(bs2):
        tokens[i, :] = input_ids
        if i % 2 == 1:
            # unconditional branch: mask out inner tokens (keep special boundaries)
            if len(input_ids) > 2:
                tokens[i, 1:-1] = processor.pad_id

    # initial embeds from LM token embedding table
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((num_images, image_token_num_per_image), dtype=torch.long, device="cuda")

    past_key_values = None
    for i in range(image_token_num_per_image):
        out = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = out.past_key_values
        hidden_states = out.last_hidden_state  # [2B, 1, hidden] after first step; generally last token state at -1

        logits_all = mmgpt.gen_head(hidden_states[:, -1, :])  # [2B, vocab_img]
        logit_cond = logits_all[0::2, :]
        logit_uncond = logits_all[1::2, :]

        # classifier-free guidance
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / max(1e-6, float(temperature)), dim=-1)

        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [B, 1]

        generated_tokens[:, i] = next_token.squeeze(-1)

        # duplicate for cond/uncond streams: [t0,t0,t1,t1,...] -> [2B]
        next_token_2b = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)

        # map token -> img embedding for next step
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_2b)  # [2B, hidden]
        inputs_embeds = img_embeds.unsqueeze(1)  # [2B, 1, hidden]

    # decode code tokens to images
    # shape expected: [B, 8, H/patch, W/patch] for Janus-1.3B default
    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[num_images, 8, img_size // patch_size, img_size // patch_size],
    )
    # dec: [B, C, H, W] in [-1,1]
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)  # [B,H,W,C]
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    imgs: List[PIL.Image.Image] = []
    for i in range(num_images):
        imgs.append(PIL.Image.fromarray(dec[i]))
    return imgs


# ---------------------------
# Janus: VQA (image + question -> text)

def _safe_get(obj, key, default=None):
    """兼容 dict / mapping-like / attribute-like 三种返回类型。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    # BatchedVLChatProcessorOutput 往往支持 __getitem__
    try:
        return obj[key]
    except Exception:
        pass
    return getattr(obj, key, default)


@torch.inference_mode()
def janus_vqa_one(
    vl_gpt,
    vl_chat_processor,
    tokenizer,
    image_path: str,
    question_text: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.0,
):
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>\n" + question_text.strip(),
            "images": [image_path],
        },
        {"role": "Assistant", "content": ""},
    ]

    # load images and prepare inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
    ).to(vl_gpt.device)

    # run image encoder to get embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # generate
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=_safe_get(prepare_inputs, "attention_mask", None),
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        use_cache=True,
    )

    full = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    print(f"############full:{full}")

    # ---- FIX HERE: no .get() on BatchedVLChatProcessorOutput ----
    sft_format = _safe_get(prepare_inputs, "sft_format", None)
    if isinstance(sft_format, (list, tuple)) and len(sft_format) > 0:
        sft_prefix = sft_format[0] or ""
    elif isinstance(sft_format, str):
        sft_prefix = sft_format
    else:
        sft_prefix = ""

    # strip prefix if present
    if sft_prefix and full.startswith(sft_prefix):
        ans = full[len(sft_prefix):].strip()
    else:
        ans = full.strip()

    return ans


# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input JSONL path")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--out_image_dir", required=True, help="dir to save generated images")
    ap.add_argument("--model_path", default="deepseek-ai/Janus-1.3B")
    ap.add_argument("--backend", default="nccl")

    # image generation params
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--image_token_num", type=int, default=576)
    ap.add_argument("--cfg_weight", type=float, default=5.0)
    ap.add_argument("--gen_temperature", type=float, default=1.0)
    ap.add_argument("--num_images", type=int, default=1, help="generate how many candidates; we will pick the first by default")
    ap.add_argument("--gen_do_sample", action="store_true", help="sample image tokens instead of using greedy decoding")
    ap.add_argument("--seed", type=int, default=1234)

    # vqa params
    ap.add_argument("--max_vqa_tokens", type=int, default=512)
    ap.add_argument("--vqa_temperature", type=float, default=0)

    # io params
    ap.add_argument("--flush_every", type=int, default=1)
    ap.add_argument("--keep_rank_files", action="store_true")
    args = ap.parse_args()

    rank, local_rank, world_size, dist_enabled = dist_init(args.backend)
    device = f"cuda:{local_rank}" if dist_enabled else "cuda:0"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(args.out_image_dir, exist_ok=True)

    out_rank = f"{args.output}.rank{rank}.jsonl" if dist_enabled else args.output

    # load Janus
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    mmgpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    ).to(torch.bfloat16).to(device).eval()

    # total + shard
    total = count_nonempty_lines(args.input)
    start, end = shard_range(total, rank, world_size)

    # global index among non-empty jsonl lines
    gidx = -1
    written = 0
    t0_all = time.perf_counter()

    with open(out_rank, "w", encoding="utf-8") as out_f:
        for _, row in iter_jsonl(args.input):
            gidx += 1
            if gidx < start:
                continue
            if gidx >= end:
                break

            t0 = time.perf_counter()

            # --------- 1) read prompt ----------
            prompt = row.get("short_description")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            prompt = prompt.strip()

            # build Janus gen prompt (multi-turn SFT + image_start_tag)
            conversation = [
                {"role": "User", "content": prompt},
                {"role": "Assistant", "content": ""},
            ]
            sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=processor.sft_format,
                system_prompt="",
            )
            gen_prompt = sft_format + processor.image_start_tag

            # --------- 2) generate images ----------
            imgs = janus_generate_image(
                mmgpt,
                processor,
                gen_prompt,
                temperature=args.gen_temperature,
                cfg_weight=args.cfg_weight,
                num_images=args.num_images,
                image_token_num_per_image=args.image_token_num,
                img_size=args.img_size,
                patch_size=args.patch_size,
                do_sample=args.gen_do_sample,
                seed=args.seed + gidx,  # per-sample deterministic but different
            )

            if not imgs:
                continue
            img0 = imgs[0]

            # save
            fname = f"{gidx:06d}_{sanitize_filename(prompt)}.png"
            img_path = os.path.join(args.out_image_dir, fname)
            img0.save(img_path)

            # --------- 3) VQA one-by-one ----------
            # tolerant key variants
            qa_list = row.get("qa_pairs") or []
            if not isinstance(qa_list, list):
                qa_list = []

            task_type = str(row.get("type", "") or "")
            janus_ans_list: List[str] = []

            for qa in qa_list:
                q = (qa or {}).get("question", "")
                if not isinstance(q, str) or not q.strip():
                    janus_ans_list.append("")
                    continue

                key_ans = (qa or {}).get("answer", "")
                prompt_text = pick_vqa_prompt(q.strip(), task_type=task_type, key_ans=str(key_ans))
                print(f"######prompt_text:{prompt_text}")

                ans = janus_vqa_one(
                    mmgpt,
                    processor,
                    tokenizer,
                    image_path=img_path,
                    question_text=prompt_text,
                    max_new_tokens=args.max_vqa_tokens,
                    do_sample=False,
                    temperature=args.vqa_temperature,
                )
                janus_ans_list.append(ans)
                print(f"#########ans:{ans}")

            # --------- 4) write output ----------
            new_row = dict(row)
            new_row["image_path"] = img_path
            new_row["ans_list"] = janus_ans_list
            out_f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            written += 1

            if args.flush_every > 0 and (written % args.flush_every == 0):
                out_f.flush()

            dt = time.perf_counter() - t0
            elapsed = time.perf_counter() - t0_all
            avg = elapsed / max(1, (gidx - start + 1))
            if rank == 0 and ((gidx - start) % 10 == 0):
                print(f"[r{rank}] idx={gidx} last={dt:.2f}s avg={avg:.2f}s written={written}")

    # merge
    if dist_enabled:
        import torch.distributed as dist
        dist.barrier()

    if dist_enabled and rank == 0:
        merge_rank_outputs(args.output, world_size)
        if not args.keep_rank_files:
            for r in range(world_size):
                p = f"{args.output}.rank{r}.jsonl"
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        print(f"[DONE] merged_output={args.output}")

    if dist_enabled:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
