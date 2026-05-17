import argparse
import json
import os
import re
import time
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from PIL import Image
from tqdm import tqdm

import torch

from data.data_utils import pil_img2rgb, add_special_tokens
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel,
    Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache

from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import torch.distributed as dist


VLM_THINK_SYSTEM_PROMPT = (
    "You should first think about the reasoning process in the mind and then provide the user with the answer. \n"
    "The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"
)

GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind and then generate the image. \n"
    "The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here"
)


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

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()
    return rank, local_rank, world_size, True


def shard_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    """
    Contiguous sharding that preserves original order after merging by rank order.
    """
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
RE_MCQ = re.compile(
    r"(\b[A-Z]\s*\.|\(\s*[A-Z]\s*\))",
    flags=re.IGNORECASE
)
def is_multi_choice(question):
    if not isinstance(question, str):
        return False
    return bool(RE_MCQ.search(question))



# ---------------------------
# core loading / inference
# ---------------------------
def build_bagel(model_dir: str, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
    """
    Load BAGEL from a local checkpoint directory.

    Required files in model_dir:
      - llm_config.json
      - vit_config.json
      - ae.safetensors
      - ema.safetensors
    """
    llm_config = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_dir, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)

    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    ckpt = os.path.join(model_dir, "ema.safetensors")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=ckpt,
        device_map={"": device},
        dtype=dtype,
        offload_buffers=False,
        force_hooks=True,
    ).eval()

    vae_model = vae_model.to(device=device, dtype=dtype).eval()
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def init_gen_context(self):
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        assert vae or vit
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        if vae:
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)

        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids["eos_token_id"],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:, 0])
        output = output.split("<|im_end|>")[0].split("<|im_start|>")[1]
        return output

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )

        cfg_text_past_key_values = cfg_text_precontext["past_key_values"]
        kv_lens_cfg = cfg_text_precontext["kv_lens"]
        ropes_cfg = cfg_text_precontext["ropes"]
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg, curr_rope=ropes_cfg, image_sizes=[image_shape]
        )

        cfg_img_past_key_values = cfg_img_precontext["past_key_values"]
        kv_lens_cfg = cfg_img_precontext["kv_lens"]
        ropes_cfg = cfg_img_precontext["ropes"]
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg, curr_rope=ropes_cfg, image_sizes=[image_shape]
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
            enable_taylorseer=enable_taylorseer,
        )
        return self.decode_image(unpacked_latent[0], image_shape)

    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(
            1, h, w,
            self.model.latent_patch_size,
            self.model.latent_patch_size,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1,
            self.model.latent_channel,
            h * self.model.latent_patch_size,
            w * self.model.latent_patch_size,
        )

        # FIX: match VAE dtype/device (bf16 vs fp32 mismatch)
        p = next(self.vae_model.parameters())
        latent = latent.to(device=p.device, dtype=p.dtype).contiguous()

        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        return Image.fromarray(image.to(torch.uint8).cpu().numpy())

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think: bool = False,
        understanding_output: bool = False,
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval=(0.4, 1.0),
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        image_shapes=(1024, 1024),
        enable_taylorseer: bool = False,
    ):
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                system_prompt = VLM_THINK_SYSTEM_PROMPT if understanding_output else GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                output_list.append(self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                ))
            else:
                if think:
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                    )
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                output_list.append(self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                ))

        return output_list

    def __call__(self, image: Optional[Image.Image] = None, text: Optional[str] = None, **kargs):
        output_dict = {"image": None, "text": None}
        if image is None and text is None:
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)
        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict["image"] = i
            elif isinstance(i, str):
                output_dict["text"] = i
        return output_dict


def sanitize_filename(name: str, max_len: int = 64) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    return (name[:max_len] if len(name) > max_len else name) or "image"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--out_image_dir", required=True)
    ap.add_argument("--model_dir", default="/home/users/wuweiqun/sunxinyu/BAGEL-7B-MoT")

    # single-GPU fallback
    ap.add_argument("--device", default="cuda:0", help="Only used when NOT torchrun.")

    # generation / vqa params (same as single gpu)
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--num_timesteps", type=int, default=50)
    ap.add_argument("--timestep_shift", type=float, default=3.0)
    ap.add_argument("--cfg_text_scale", type=float, default=3.0)
    ap.add_argument("--cfg_img_scale", type=float, default=1.5)
    ap.add_argument("--max_vqa_tokens", type=int, default=64)
    ap.add_argument("--vqa_temperature", type=float, default=0.001)
    ap.add_argument("--flush_every", type=int, default=1)

    # torchrun / ddp
    ap.add_argument("--backend", default="nccl")
    ap.add_argument("--think", action="store_true", help="Enable <think> prompts for gen & vqa")
    ap.add_argument("--keep_rank_files", action="store_true", help="Keep output.rank*.jsonl files")
    ap.add_argument("--disable_multi_tqdm", action="store_true", help="Only show tqdm on rank0")
    args = ap.parse_args()

    rank, local_rank, world_size, dist_enabled = dist_init(args.backend)
    device = f"cuda:{local_rank}" if dist_enabled else args.device

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(args.out_image_dir, exist_ok=True)

    # each rank saves images into its own subdir to avoid collisions
    img_dir_rank = args.out_image_dir
    os.makedirs(img_dir_rank, exist_ok=True)


    out_rank = f"{args.output}.rank{rank}.jsonl" if dist_enabled else args.output

    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = build_bagel(
        args.model_dir, device=device, dtype=torch.bfloat16
    )
    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)

    total = count_nonempty_lines(args.input)
    start, end = shard_range(total, rank, world_size)
    shard_total = end - start

    # per-rank progress bar (or only rank0)
    pbar = tqdm(
        total=shard_total,
        desc=f"samples[r{rank}/{world_size}]",
        unit="sample",
        dynamic_ncols=True,
        position=rank if not args.disable_multi_tqdm else 0,
        leave=True,
        disable=(args.disable_multi_tqdm and rank != 0),
    )

    written = 0
    t0_all = time.perf_counter()
    gidx = -1  # global index among non-empty JSONL lines

    with open(out_rank, "w", encoding="utf-8") as out_f:
        for _, row in iter_jsonl(args.input):
            gidx += 1
            if gidx < start:
                continue
            if gidx >= end:
                break

            t0 = time.perf_counter()

            # handle common key typo
            prompt = row.get("short_description")
            task_type = row.get("type").lower()
            # if task_type!="text":
            #     continue
            if prompt is None:
                prompt = row.get("short_desription")  # legacy typo support

            if not isinstance(prompt, str) or not prompt.strip():
                pbar.update(1)
                continue
            prompt = prompt.strip()

            # 1) generate image
            gen = inferencer(
                text=prompt,
                understanding_output=False,
                think=args.think,
                image_shapes=(args.image_size, args.image_size),
                num_timesteps=args.num_timesteps,
                timestep_shift=args.timestep_shift,
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
            )
            img = gen["image"]
            if not isinstance(img, Image.Image):
                pbar.update(1)
                continue

            filename_safe = sanitize_filename(prompt)
            img_filename = f"{gidx:05d}_{filename_safe}.png"
            img_path = os.path.join(img_dir_rank, img_filename)
            img.save(img_path)

            # 2) VQA one-by-one
            qa_pairs = row.get("qa_pairs") or []
            if not isinstance(qa_pairs, list):
                qa_pairs = []

            ans_list: List[str] = []
            for qa in qa_pairs:
                q = (qa or {}).get("question", "")
                if not isinstance(q, str) or not q.strip():
                    ans_list.append("")
                    continue

                question = q.strip()

                # print(f"#####quetsion:{question}")
                # print(f"#####task_type:{task_type}")

                key_ans = qa.get("answer","").strip().lower()
                if is_multi_choice(question):
                    # print("use muti choice tmplate")
                    prompt_text = VQA_PROMPT_TEMPLATE_MULTI_CHOICE.format(question=question)
                elif task_type == "negation" or key_ans in ["yes","no"]:
                    # print("use negation tmplate")
                    prompt_text = VQA_PROMPT_TEMPLATE_NEGATION.format(question=question)
                else:
                    # print("use general tmplate")
                    prompt_text = VQA_PROMPT_TEMPLATE.format(question=question)
                resp = inferencer(
                    image=img,
                    text=prompt_text,
                    understanding_output=True,
                    think=args.think,
                    do_sample=False,
                    text_temperature=args.vqa_temperature,
                    max_think_token_n=args.max_vqa_tokens,
                )
                ans = (resp.get("text") or "").strip()
                ans = ans.splitlines()[0].strip() if ans else ""
                ans_list.append(ans)

            # 3) write back
            new_row = dict(row)
            new_row["image_path"] = img_path
            new_row["ans_list"] = ans_list
            # if task_type == "text":
            #     new_row["word_count"] = row.get("word_count")
            out_f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            written += 1

            if args.flush_every > 0 and (written % args.flush_every == 0):
                out_f.flush()

            dt = time.perf_counter() - t0
            elapsed = time.perf_counter() - t0_all
            avg = elapsed / max(1, (gidx - start + 1))
            pbar.set_postfix_str(f"last={dt:.2f}s avg={avg:.2f}s written={written}")
            pbar.update(1)

    pbar.close()

    if dist_enabled:
        dist.barrier()

    # rank0 merge
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
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
