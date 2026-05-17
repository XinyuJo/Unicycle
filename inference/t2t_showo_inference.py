import argparse
import os
import json
import re
import sys
from typing import Any, Dict, List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
from accelerate.logging import get_logger

from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import (
    get_config,
    denorm,
    get_hyper_params,
    path_to_llm_name,
    load_state_dict,
)

from datasets.utils import image_transform
from transport import Sampler, create_transport

logger = get_logger(__name__, log_level="INFO")
rank = int(os.environ.get("RANK", "0"))


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--out_image_dir", required=True)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--mmu_max_new_tokens", type=int, default=None)
    parser.add_argument("--mmu_top_k", type=int, default=None)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--config", args.config, *remaining]
    return args


def _dist_is_launched() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def dist_init(backend: str = "nccl") -> Tuple[int, int, int, bool]:
    if not _dist_is_launched():
        return 0, 0, 1, False

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
    tmp_paths = [final_path.replace(".jsonl", f".rank{r:02d}.jsonl") for r in range(world_size)]
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    with open(final_path, "w", encoding="utf-8") as w:
        for p in tmp_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as r:
                for line in r:
                    w.write(line)

# -------------------------
# 3 prompt templates 
# -------------------------
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


_mc_pat = re.compile(r"(\bA[\).\:]|\bB[\).\:]|\bC[\).\:]|\bD[\).\:])", re.IGNORECASE)


def is_multi_choice(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    return _mc_pat.search(q) is not None


def pick_vqa_prompt(question: str, task_type: str = "", key_ans: str = "") -> str:
    q = (question or "").strip()
    t = (task_type or "").strip().lower()
    ka = (key_ans or "").strip().lower()

    if is_multi_choice(q):
        return VQA_PROMPT_TEMPLATE_MULTI_CHOICE.format(question=q)
    if t == "negation" or ka in ("yes", "no"):
        return VQA_PROMPT_TEMPLATE_NEGATION.format(question=q)
    return VQA_PROMPT_TEMPLATE.format(question=q)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _get_prompt_from_row(row: Dict[str, Any]) -> str:
    p = row.get("short_description")
    if p is None:
        p = row.get("short_desription")  # tolerate typo
    if not p:
        p = row.get("prompt", row.get("prompts", ""))
        if isinstance(p, list):
            p = p[0] if p else ""
    return str(p).strip()


def sanitize_filename(name: str, max_len: int = 64) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    return (name[:max_len] if len(name) > max_len else name) or "image"


def _get_qa_list_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    qa = row.get("qa_pairs") or []
    return qa if isinstance(qa, list) else []


# -------------------------
# MMU single-image answering
# -------------------------
@torch.no_grad()
def mmu_answer_one(
    model: Showo2Qwen2_5,
    vae_model,
    text_tokenizer,
    showo_token_ids: Dict[str, int],
    image_pil: Image.Image,
    question_prompt: str,
    device: torch.device,
    weight_type: torch.dtype,
    resolution: int,
    num_mmu_image_tokens: int,
    max_new_tokens: int,
    top_k: int,
) -> str:
    # image -> tensor -> vae latent
    image = image_transform(image_pil.convert("RGB"), resolution=resolution).to(device)
    image = image.unsqueeze(0)  # (1,C,H,W)
    image_latents = vae_model.sample(image.unsqueeze(2)).squeeze(2).to(weight_type)

    # image latents -> image embeds
    image_embeds_und = model.image_embedder_und(image_latents)
    image_embeds_gen = model.image_embedder_gen(image_latents)
    image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
    image_embeds_und = model.und_trans(image_embeds_und)["last_hidden_state"]
    image_embeds = model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

    # text tokens
    sys_prompt_ids = text_tokenizer(
        "system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False
    )["input_ids"]
    role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    role_b = text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

    input_ids = text_tokenizer(question_prompt, add_special_tokens=False).input_ids

    text_tokens_a = torch.tensor([showo_token_ids["bos_id"]] + sys_prompt_ids + role_a, device=device)[None, :]
    text_tokens_b = torch.tensor(
        [showo_token_ids["boi_id"], showo_token_ids["eoi_id"]] + input_ids + role_b,
        device=device
    )[None, :]

    text_embeds_a = model.showo.model.embed_tokens(text_tokens_a)
    text_embeds_b = model.showo.model.embed_tokens(text_tokens_b)

    # concat embeds (time embeds optional)
    if getattr(model, "add_time_embeds", False) or getattr(model.config, "add_time_embeds", False):
        time_embeds = model.time_embed(torch.Tensor([[1.0]]).to(device), text_embeds_a.dtype)
        if hasattr(model, "time_embed_proj"):
            time_embeds = model.time_embed_proj(time_embeds)
        input_embeds = torch.cat(
            [text_embeds_a, text_embeds_b[:, :1], time_embeds, image_embeds, text_embeds_b[:, 1:]],
            dim=1
        ).to(weight_type)
        modality_positions = torch.tensor(
            [text_tokens_a.shape[1] + 2, num_mmu_image_tokens],
            device=device
        )[None, None, :]
    else:
        input_embeds = torch.cat(
            [text_embeds_a, text_embeds_b[:, :1], image_embeds, text_embeds_b[:, 1:]],
            dim=1
        ).to(weight_type)
        modality_positions = torch.tensor(
            [text_tokens_a.shape[1] + 1, num_mmu_image_tokens],
            device=device
        )[None, None, :]

    attn = omni_attn_mask_naive(
        B=input_embeds.size(0),
        LEN=input_embeds.size(1),
        modalities=modality_positions,
        device=device,
        inverted=True,
    ).to(input_embeds.dtype)

    out_tokens = model.mmu_generate(
        input_embeds=input_embeds,
        attention_mask=attn,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        eos_token=text_tokenizer.eos_token_id,
    )
    out_tokens = torch.stack(out_tokens).squeeze()[None]
    text = text_tokenizer.batch_decode(out_tokens, skip_special_tokens=True)
    return (text[0] or "").strip()


def main():
    cli_args = parse_cli_args()
    config = get_config()
    backend = cli_args.backend or getattr(config, "backend", "nccl")
    rank, local_rank, world_size, dist_enabled = dist_init(backend)

    # ----- device (torchrun friendly) -----
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # ----- weight dtype -----
    if config.model.weight_type == "bfloat16":
        weight_type = torch.bfloat16
    elif config.model.weight_type == "float32":
        weight_type = torch.float32
    else:
        raise NotImplementedError

    # ----- VAE -----
    if config.model.vae_model.type == "wan21":
        from models import WanVAE
        vae_model = WanVAE(
            vae_pth=config.model.vae_model.pretrained_model_path,
            dtype=weight_type,
            device=device
        )
    else:
        raise NotImplementedError

    # ----- tokenizer (Show-o2) -----
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    # ----- model -----
    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path,
            use_safetensors=False
        ).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model_path)
        model.load_state_dict(state_dict)

    model.to(weight_type)
    model.eval()

    # ----- hyper params -----
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, showo_token_ids)

    # ----- transport/sampler (T2I) -----
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=num_t2i_image_tokens,
    )
    sampler = Sampler(transport)

    # ----- IO config -----
    input_jsonl = cli_args.input
    output_jsonl = cli_args.output
    out_image_dir = cli_args.out_image_dir
    out_path_rank = output_jsonl.replace(".jsonl", f".rank{rank:02d}.jsonl") if dist_enabled else output_jsonl
    os.makedirs(os.path.dirname(out_path_rank), exist_ok=True)


    if not input_jsonl or not output_jsonl:
        raise ValueError("Need input_jsonl and output_jsonl in YAML (top-level or dataset.params.*).")

    if out_image_dir:
        os.makedirs(out_image_dir, exist_ok=True)

    rows = read_jsonl(input_jsonl)
    start, end = shard_range(len(rows), rank, world_size)
    shard_rows = rows[start:end]

    # generation params
    batch_size = int(cli_args.batch_size if cli_args.batch_size is not None else getattr(config, "batch_size", 1))
    guidance_scale = float(cli_args.guidance_scale if cli_args.guidance_scale is not None else getattr(config, "guidance_scale", config.transport.guidance_scale))
    num_inference_steps = int(cli_args.num_inference_steps if cli_args.num_inference_steps is not None else getattr(config, "num_inference_steps", config.transport.num_inference_steps))
    mmu_max_new_tokens = int(cli_args.mmu_max_new_tokens if cli_args.mmu_max_new_tokens is not None else getattr(config, "mmu_max_new_tokens", 128))
    mmu_top_k = int(cli_args.mmu_top_k if cli_args.mmu_top_k is not None else getattr(config, "mmu_top_k", 1))
    resolution = int(config.dataset.preprocessing.resolution)

    # override transport steps
    config.transport.num_inference_steps = num_inference_steps

    out_rows: List[Dict[str, Any]] = []

    # ----- main loop -----
    fout = open(out_path_rank, "w", encoding="utf-8")
    for st in tqdm(range(0, len(shard_rows), batch_size), desc=f"t2i->mmu[r{rank}/{world_size}]"):
        batch = shard_rows[st: st + batch_size]

        prompts: List[str] = []
        metas: List[Dict[str, Any]] = []
        task_types: List[str] = []
        qa_lists: List[List[Dict[str, Any]]] = []

        for r in batch:
            p = _get_prompt_from_row(r)
            qa = _get_qa_list_from_row(r)
            if not p or not qa:
                rr = dict(r)
                rr["skipped"] = True
                rr["reason"] = "missing prompt or qa_pairs"
                out_rows.append(rr)
                continue
            prompts.append(p)
            metas.append(r)
            task_types.append(str(r.get("type", "") or ""))
            qa_lists.append(qa)

        if not prompts:
            continue

        # ----- T2I generate -----
        batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
            prepare_gen_input(
                prompts,
                text_tokenizer,
                num_t2i_image_tokens,
                bos_id, eos_id, boi_id, eoi_id,
                pad_id, img_pad_id,
                max_text_len,
                device,
            )

        z = torch.randn(
            (len(prompts), image_latent_dim, latent_height * patch_size, latent_width * patch_size),
            device=device,
            dtype=torch.bfloat16 if weight_type == torch.bfloat16 else torch.float32,
        )

        if guidance_scale > 0:
            z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
            block_mask = omni_attn_mask_naive(
                text_tokens.size(0), max_seq_len, modality_positions, device
            ).to(weight_type)
        else:
            text_tokens = batch_text_tokens
            modality_positions = batch_modality_positions
            block_mask = omni_attn_mask_naive(
                text_tokens.size(0), max_seq_len, modality_positions, device
            ).to(weight_type)

        model_kwargs = dict(
            text_tokens=text_tokens,
            attention_mask=block_mask,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=max_seq_len,
            guidance_scale=guidance_scale,
        )

        sample_fn = sampler.sample_ode(
            sampling_method=config.transport.sampling_method,
            num_steps=config.transport.num_inference_steps,
            atol=config.transport.atol,
            rtol=config.transport.rtol,
            reverse=config.transport.reverse,
            time_shifting_factor=config.transport.time_shifting_factor,
        )

        samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
        if guidance_scale > 0:
            samples = torch.chunk(samples, 2)[0]

        # decode via WanVAE
        samples = samples.unsqueeze(2)
        images = vae_model.batch_decode(samples).squeeze(2)

        images = denorm(images)
        pil_images = [Image.fromarray(im) for im in images]

        # optional: save images
        img_paths: List[str] = []
        if out_image_dir:
            for i, im in enumerate(pil_images):
                gidx = start + st + i
                filename_safe = sanitize_filename(prompts[i])  # 确保这里的 prompts 与 pil_images 对齐
                img_filename = f"{gidx:05d}_{filename_safe}.png"
                p = os.path.join(out_image_dir, img_filename)
                im.save(p)
                img_paths.append(p)


        # ----- MMU QA -----
        for i, (meta, prompt, qa_list, ttype, gen_img) in enumerate(zip(metas, prompts, qa_lists, task_types, pil_images)):
            ans_list = []
            for qa in qa_list:
                q = (qa or {}).get("question", "")
                if not isinstance(q, str) or not q.strip():
                    ans_list.append("")
                    continue
                key_ans = str((qa or {}).get("answer", "") or "")
                q_prompt = pick_vqa_prompt(q.strip(), task_type=ttype, key_ans=key_ans)

                ans = mmu_answer_one(
                    model=model,
                    vae_model=vae_model,
                    text_tokenizer=text_tokenizer,
                    showo_token_ids=showo_token_ids,
                    image_pil=gen_img,
                    question_prompt=q_prompt,
                    device=device,
                    weight_type=weight_type,
                    resolution=resolution,
                    num_mmu_image_tokens=num_mmu_image_tokens,
                    max_new_tokens=mmu_max_new_tokens,
                    top_k=mmu_top_k,
                )
                ans_list.append(ans)

            rr = dict(meta)
            rr["ans_list"] = ans_list
            if img_paths:
                rr["image_path"] = img_paths[i]
            out_rows.append(rr)
            fout.write(json.dumps(rr, ensure_ascii=False) + "\n")
            fout.flush()
    fout.close()

    if dist_enabled:
        dist.barrier()

    if dist_enabled and rank == 0:
        merge_rank_outputs(output_jsonl, world_size)
        logger.info(f"Done. Merged outputs to {output_jsonl}")
    elif not dist_enabled:
        write_jsonl(output_jsonl, out_rows)
        logger.info(f"Done. Wrote {len(out_rows)} rows to {output_jsonl}")

    if dist_enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
