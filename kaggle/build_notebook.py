"""Builds kaggle/sslvlm_kaggle.ipynb from the cell sources below (edit here, then rerun this script).

    python kaggle/build_notebook.py
"""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


md("""
# SSLVLM on Kaggle: GRPO on self-supervised COCO puzzles

Trains a Qwen3-VL model (4-bit + LoRA via Unsloth) with GRPO on the SSL-R1 puzzles from
[erow/ServerlessRL](https://github.com/erow/ServerlessRL), including the hard-negative inpainting arms.
Unlike W&B Serverless RL, the LoRA can also cover the **vision encoder** (`FINETUNE_VISION=True`), so the
eval protocol's representation probes (dimension F) apply to these checkpoints.

Edit cell 2 (or pass `KEY=VALUE` to `run.sh`). Outputs go to `/kaggle/working/out/<ARM>/`.
""")

code("""
# Cell 1: install. If this breaks against Kaggle's preinstalled torch, copy the install cell of Unsloth's
# official Qwen3-VL Kaggle notebook here.
%pip install -q unsloth
%pip install -q "transformers>=4.57" pycocotools
""")

code("""
# Cell 2: configuration. run.sh replaces the CONFIG_OVERRIDES line below; keep it on one line.
CONFIG = dict(
    ARM="A-spatial",                 # one of ARMS below
    REPO_URL="https://github.com/erow/ServerlessRL.git",
    REPO_REF="hard-negative-inpainting",  # branch, tag or commit with the task generators
    MODEL_NAME="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    FINETUNE_VISION=True,            # LoRA on the vision encoder too (impossible on W&B Serverless)
    LORA_R=16, LORA_ALPHA=16, LORA_DROPOUT=0.0,
    STEPS=60,                        # optimizer steps; 60 = pipeline check
    PROMPTS_PER_STEP=4,              # puzzles per step
    GROUP_SIZE=8,                    # rollouts per puzzle (GRPO group)
    LR=1e-5, MAX_GRAD_NORM=1.0,
    MAX_NEW_TOKENS=768, TEMPERATURE=1.0, TOP_P=0.95,
    FORMAT_REWARD=0.2,
    VAL_EVERY=20, VAL_N=48,          # greedy held-out eval on val2017 puzzles (0 disables)
    SAVE_EVERY=20,
    VAL_AT_START=True,               # step-0 eval; false skips it (e.g. to reach the first update fast)
    TIME_BUDGET_H=11.0,              # stop and save before Kaggle's 12 h limit
    HF_REPO="",                      # e.g. "you/sslvlm-kaggle"; needs Kaggle secret HF_TOKEN; enables resume
    RESUME=True,                     # resume ARM from HF_REPO/<ARM>/latest if present
    WANDB_PROJECT="sslvlm-kaggle",   # used only if Kaggle secret WANDB_API_KEY exists
    SEED=0,
    GEN_BATCH=16,                    # sequences generated together on cuda:0 (caption game)
    GEN_BATCH_GPU1=24,               # same on the frozen copy on cuda:1 (no training state there)
    FROZEN_ON_GPU1=True,             # caption game: frozen captioner/judge on a second model copy on cuda:1
    # Masked caption game (ARM capgame-*): forger-only against a frozen captioner and judge (the base model,
    # i.e. LoRA disabled), as in runs v4-v7 on W&B Serverless.
    # CG_JUDGE_SAMPLES 2 (W&B runs used 4) and judge cap 512 (768) halve the judge cost on a T4.
    CG_MASK_RATIO=0.3, CG_IMAGES_PER_STEP=8, CG_CAPTION_SAMPLES=4, CG_JUDGE_SAMPLES=2,
    CG_CAPTION_MAX_TOKENS=320, CG_JUDGE_MAX_TOKENS=512, CG_VAL_IMAGES=32,
    CG_FOOL_WEIGHT=0.5, CG_CONTENT_WEIGHT=0.5, CG_CONTENT_METRIC="recall", CG_GATE_PENALTY=-0.5,
)
CONFIG_OVERRIDES = {}

# Arms of the hard-negative inpainting study (plan doc phase 1), the SSL-R1 multi-task baseline, and the masked
# caption game. "oracle_index" = "auto" builds the COCO label index for the split on first use. An arm's
# "config" entries apply unless CONFIG_OVERRIDES sets the same key.
SQ = {"mask": "square"}
ARMS = {
    "A-spatial": dict(game="puzzles", tasks="inpainting", task_params={"inpainting": {**SQ, "distractors": "spatial"}}),
    "B-oracle":  dict(game="puzzles", tasks="inpainting", task_params={"inpainting": {
        **SQ, "distractors": "oracle-cat,oracle-dist,oracle-dist", "oracle_index": "auto"}}),
    "C-mixed":   dict(game="puzzles", tasks="inpainting", task_params={"inpainting": {
        **SQ, "distractors": "rot90,oracle-cat,oracle-dist", "oracle_index": "auto"}}),
    "D-random":  dict(game="puzzles", tasks="inpainting", task_params={"inpainting": {**SQ, "distractors": "random"}}),
    "ssl-r1":    dict(game="puzzles", tasks="ssl-r1", task_params={}),
    # v7 on Kaggle: v6 settings (mask 0.3, forger-only, thinking off) with lr 3e-5, 100 steps.
    "capgame-v7": dict(game="caption", config=dict(REPO_REF="masked-caption-game", LR=3e-5, STEPS=100,
                                                   VAL_EVERY=20, SAVE_EVERY=10)),
}
ARM = ARMS[CONFIG_OVERRIDES.get("ARM", CONFIG["ARM"])]
for k, v in ARM.get("config", {}).items():
    CONFIG[k] = v
CONFIG.update(CONFIG_OVERRIDES)
GAME = ARM["game"]
CONFIG.setdefault("TASKS", ARM.get("tasks"))
CONFIG.setdefault("TASK_PARAMS", ARM.get("task_params", {}))
CONFIG
""")

code("""
# Cell 3: data and task code. COCO comes from the attached Kaggle dataset; the task generators from the repo.
import json, os, subprocess, sys, time
from pathlib import Path

def find(name, kind="dir", root=Path("/kaggle/input"), max_depth=4):
    for depth in range(max_depth + 1):
        for p in sorted(root.glob("/".join(["*"] * depth + [name]))):
            if (p.is_dir() if kind == "dir" else p.is_file()):
                return p
    raise FileNotFoundError(f"{name} not found under {root}; attach a COCO 2017 dataset")

TRAIN_DIR, VAL_DIR = find("train2017"), find("val2017")
print("COCO:", TRAIN_DIR, VAL_DIR)

SRC = Path("/tmp/src")
if not (SRC / "serverless").exists():
    subprocess.run(["git", "clone", "-q", CONFIG["REPO_URL"], str(SRC / "serverless")], check=True)
    subprocess.run(["git", "-C", str(SRC / "serverless"), "checkout", "-q", CONFIG["REPO_REF"]], check=True)
REPO_COMMIT = subprocess.run(["git", "-C", str(SRC / "serverless"), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
# The repo expects a parent package `sslvlm.config`; a two-line shim is enough.
(SRC / "sslvlm").mkdir(exist_ok=True)
(SRC / "sslvlm" / "__init__.py").write_text("")
(SRC / "sslvlm" / "config.py").write_text(
    f"from pathlib import Path\\nCOCO_DIR = Path({str(TRAIN_DIR)!r})\\nOUTPUT_ROOT = Path('/kaggle/working/out')\\n")
sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")
print("task code:", CONFIG["REPO_REF"], REPO_COMMIT)

OUT = Path("/kaggle/working/out") / CONFIG["ARM"]
OUT.mkdir(parents=True, exist_ok=True)
INDEX_DIR = Path("/kaggle/working/index")

def label_index(split):
    \"\"\"Oracle label index for train2017/val2017 (instances + COCO-Stuff when the dataset has it).\"\"\"
    out = INDEX_DIR / f"{split}_labels.npz"
    if not out.exists():
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "serverless.hardneg", "build-index",
               "--instances", str(find(f"instances_{split}.json", "file")), "--out", str(out)]
        try:
            cmd += ["--stuff", str(find(f"stuff_{split}.json", "file"))]
        except FileNotFoundError:
            print("no COCO-Stuff annotations; oracle-dist uses things only")
        t = time.time()
        subprocess.run(cmd, check=True)
        print(f"built {out} in {time.time() - t:.0f}s")
    return str(out)

def task_params(split):
    params = json.loads(json.dumps(CONFIG["TASK_PARAMS"]))
    for p in params.values():
        if p.get("oracle_index") == "auto":
            p["oracle_index"] = label_index(split)
    return params

from serverless.config import FORMAT_PROMPT
from serverless.images import ImageSource

train_src, val_src = ImageSource(TRAIN_DIR), ImageSource(VAL_DIR)
print(len(train_src), "train images,", len(val_src), "val images")
if GAME == "puzzles":
    from serverless.objective import Objective
    train_obj = Objective.from_spec(CONFIG["TASKS"], task_params=task_params("train2017"), seed=CONFIG["SEED"],
                                    format_reward=CONFIG["FORMAT_REWARD"])
    val_obj = Objective.from_spec(CONFIG["TASKS"], task_params=task_params("val2017"), seed=CONFIG["SEED"],
                                  format_reward=CONFIG["FORMAT_REWARD"])
    p = train_obj.batch(train_src, "train", 0, 1)[0]
    print(p.task, p.answer, p.text[:300])
else:
    # caption_game imports message_reasoning from serverless.train, which pulls in the W&B/ART client; a stub
    # is enough because thinking is off here.
    import types
    _stub = types.ModuleType("serverless.train")
    _stub.message_reasoning = lambda message: getattr(message, "reasoning", None)
    sys.modules["serverless.train"] = _stub
    import serverless.caption_game as CG
    CG_CFG = CG.GameConfig(
        caption_samples=CONFIG["CG_CAPTION_SAMPLES"], judge_samples=CONFIG["CG_JUDGE_SAMPLES"],
        temperature=CONFIG["TEMPERATURE"], format_reward=CONFIG["FORMAT_REWARD"],
        forger_fool_weight=CONFIG["CG_FOOL_WEIGHT"], forger_content_weight=CONFIG["CG_CONTENT_WEIGHT"],
        content_metric=CONFIG["CG_CONTENT_METRIC"], gate_penalty=CONFIG["CG_GATE_PENALTY"], thinking=False)
    name, img, rng = CG.step_images(train_src, CONFIG["SEED"], "train", 0, 1)[0]
    print("caption game:", name, img.size, CG.salient_box(rng, img, CONFIG["CG_MASK_RATIO"], 8, 28))
""")

code("""
# Cell 4: model. Unsloth must be imported before transformers.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")  # we need logits for the policy-gradient loss
from unsloth import FastVisionModel
import torch

model, processor = FastVisionModel.from_pretrained(
    CONFIG["MODEL_NAME"], load_in_4bit=True, use_gradient_checkpointing="unsloth")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=CONFIG["FINETUNE_VISION"],
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=CONFIG["LORA_R"], lora_alpha=CONFIG["LORA_ALPHA"], lora_dropout=CONFIG["LORA_DROPOUT"],
    bias="none", random_state=CONFIG["SEED"],
)
# LoRA weights in fp32 so the fp16 GradScaler can unscale them (T4 has no bf16).
trainable = [p for p in model.parameters() if p.requires_grad]
for p in trainable:
    p.data = p.data.float()
names = [n for n, p in model.named_parameters() if p.requires_grad]
n_vis = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "visual" in n)
print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.1f}M params, {n_vis / 1e6:.1f}M in the vision encoder")
TOK = processor.tokenizer if hasattr(processor, "tokenizer") else processor
STOP_IDS = {i for i in [TOK.eos_token_id, TOK.convert_tokens_to_ids("<|im_end|>"), TOK.pad_token_id] if i is not None}
PAD_ID = TOK.pad_token_id if TOK.pad_token_id is not None else TOK.eos_token_id
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
""")

code("""
# Cell 5: rollouts, GRPO loss, eval, checkpoints.
from PIL import Image

def encode(puzzle):
    \"\"\"Puzzle -> processor inputs: one user turn, text and images interleaved, format prompt appended.\"\"\"
    content, images = [], []
    for part in puzzle.parts:
        if isinstance(part, Image.Image):
            content.append({"type": "image"}); images.append(part)
        else:
            content.append({"type": "text", "text": part})
    content.append({"type": "text", "text": "\\n" + FORMAT_PROMPT})
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=images, return_tensors="pt").to("cuda")

def trim(seq):
    \"\"\"Completion tokens up to and including the first stop token.\"\"\"
    ids = seq.tolist()
    for i, t in enumerate(ids):
        if t in STOP_IDS:
            return seq[: i + 1]
    return seq

@torch.no_grad()
def generate(inputs, n, greedy=False):
    FastVisionModel.for_inference(model)
    kw = dict(max_new_tokens=CONFIG["MAX_NEW_TOKENS"], pad_token_id=PAD_ID, use_cache=True)
    kw.update(do_sample=False) if greedy else kw.update(
        do_sample=True, temperature=CONFIG["TEMPERATURE"], top_p=CONFIG["TOP_P"])
    plen = inputs["input_ids"].shape[1]
    try:
        out = model.generate(**inputs, num_return_sequences=n, **kw)
        seqs = [trim(s[plen:]) for s in out]
    except Exception as e:  # some VLM generate paths cannot expand pixel inputs; fall back to a loop
        if n == 1:
            raise
        print("batched generate failed, looping:", repr(e)[:200])
        seqs = [trim(model.generate(**inputs, **kw)[0, plen:]) for _ in range(n)]
    return seqs, [TOK.decode(s, skip_special_tokens=True) for s in seqs]

def completion_logprob(inputs, comp):
    \"\"\"Sum of log p(completion | prompt, images) under the current policy, with gradients.\"\"\"
    # generate() runs under torch.inference_mode, and autograd refuses to save inference tensors for backward
    # (gather's index, the vision inputs): clone them into normal tensors first.
    inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
    comp = comp.clone().to(inputs["input_ids"].device)
    plen = inputs["input_ids"].shape[1]
    ids = torch.cat([inputs["input_ids"][0], comp]).unsqueeze(0)
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    # Per-token processor outputs (transformers 5.x: mm_token_type_ids) cover only the prompt; extend them over
    # the completion as text tokens (0), or get_rope_index fails on mismatched lengths.
    for k in ("mm_token_type_ids", "token_type_ids"):
        v = extra.get(k)
        if v is not None and v.shape[-1] == plen:
            extra[k] = torch.cat([v, v.new_zeros(*v.shape[:-1], len(comp))], -1)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra).logits
    logits = logits[0, plen - 1 : plen - 1 + len(comp)].float()
    return torch.log_softmax(logits, -1).gather(-1, comp[:, None]).squeeze(-1).sum()

def evaluate(step):
    \"\"\"Greedy accuracy on fixed val2017 puzzles (same puzzles at every eval).\"\"\"
    puzzles = val_obj.batch(val_src, "val", 0, CONFIG["VAL_N"])
    rows = []
    for pz in puzzles:
        _, texts = generate(encode(pz), 1, greedy=True)
        s = val_obj.score(pz, texts[0])
        rows.append({"step": step, "id": pz.meta["puzzle_id"], "task": pz.task, "answer": pz.answer,
                     "accuracy": s.accuracy, "format": s.format, "parsed": s.parsed, "response": texts[0]})
    with open(OUT / f"val_step{step}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\\n" for r in rows)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["accuracy"])
    summary = {f"val/{t}/accuracy": sum(v) / len(v) for t, v in by_task.items()}
    summary["val/accuracy"] = sum(r["accuracy"] for r in rows) / len(rows)
    summary["val/parsed"] = sum(r["parsed"] for r in rows) / len(rows)
    return summary

def lora_delta_norms():
    \"\"\"||B A|| * scaling summed over LoRA modules, split into vision encoder vs language model.
    A quick check that training actually moved the encoder; run the eval protocol's F probes for the real answer.\"\"\"
    tot = {"vision": 0.0, "language": 0.0}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            a, b = mod.lora_A["default"].weight.float(), mod.lora_B["default"].weight.float()
            d = (b @ a).norm().item() * mod.scaling["default"]
            tot["vision" if "visual" in name else "language"] += d ** 2
    return {f"lora_delta/{k}": v ** 0.5 for k, v in tot.items()}

def policy_update(rollouts):
    \"\"\"One GRPO step from (inputs, completion ids, advantage); token-level normalization over the batch.\"\"\"
    FastVisionModel.for_training(model)
    optimizer.zero_grad(set_to_none=True)
    n_tok = max(1, sum(len(c) for _, c, _ in rollouts))
    for inputs, comp, adv in rollouts:
        loss = -adv * completion_logprob(inputs, comp) / n_tok
        scaler.scale(loss).backward()
    if not rollouts:
        return 0.0
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, CONFIG["MAX_GRAD_NORM"]).item()
    scaler.step(optimizer)
    scaler.update()
    return grad_norm

def group_advantages(rewards):
    \"\"\"GRPO advantages for one group, or None when all rewards are equal (no signal).\"\"\"
    mu, sd = statistics.mean(rewards), statistics.pstdev(rewards)
    return None if sd <= 1e-6 else [(r - mu) / (sd + 1e-4) for r in rewards]

# ---- batched generation for the caption game (chat-completions-style messages with data-URI images)
import asyncio, base64, contextlib, io, statistics
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

def _decode(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")

def chat_inputs(messages):
    content, images = [], []
    for c in messages[0]["content"]:
        if c["type"] == "image_url":
            content.append({"type": "image"}); images.append(_decode(c["image_url"]["url"]))
        else:
            content.append({"type": "text", "text": c["text"]})
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    return text, images

def encode_messages(messages):
    text, images = chat_inputs(messages)
    return processor(text=[text], images=images, return_tensors="pt").to("cuda")

# A second copy of the base model on the second GPU serves the frozen roles (captioner, judge) in parallel with
# the policy on cuda:0. Same 4-bit weights as the policy with its LoRA disabled. Falls back to the policy model
# with the adapter disabled when there is one GPU or the copy fails.
import threading
POLICY_LOCK = threading.Lock()
FROZEN = None
if GAME == "caption" and CONFIG["FROZEN_ON_GPU1"] and torch.cuda.device_count() > 1:
    try:
        from transformers import AutoModelForImageTextToText
        FROZEN = AutoModelForImageTextToText.from_pretrained(
            CONFIG["MODEL_NAME"], device_map={"": 1}, dtype=torch.float16).eval()
        print("frozen captioner/judge model on cuda:1")
    except Exception as e:
        print("no frozen copy on cuda:1, using the policy with LoRA disabled:", repr(e)[:300])
        FROZEN = None

def _sample_kw(n, max_new_tokens, temperature):
    greedy = temperature == 0
    k = 1 if greedy else n
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=PAD_ID, use_cache=True)
    kw.update(do_sample=False) if greedy else kw.update(do_sample=True, temperature=temperature, top_p=CONFIG["TOP_P"])
    return k, kw

def _generate_on(m, device, prompts, n, max_new_tokens, temperature, batch):
    \"\"\"n samples for each prompt (list of messages) on model m, batched with left padding.\"\"\"
    k, kw = _sample_kw(n, max_new_tokens, temperature)
    per = max(1, batch // k)
    results = []
    for i in range(0, len(prompts), per):
        chunk = [chat_inputs(p) for p in prompts[i : i + per]]
        try:
            inputs = processor(text=[t for t, _ in chunk], images=[im for _, ims in chunk for im in ims],
                               padding=True, return_tensors="pt").to(device)
            out = m.generate(**inputs, num_return_sequences=k, **kw)
            plen = inputs["input_ids"].shape[1]
            rows = [[trim(s[plen:].cpu()) for s in out[j * k : (j + 1) * k]] for j in range(len(chunk))]
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # fall back to one prompt and one sample at a time
            print("batched generate failed, looping:", repr(e)[:200])
            rows = []
            for text, ims in chunk:
                inp = processor(text=[text], images=ims, return_tensors="pt").to(device)
                pl = inp["input_ids"].shape[1]
                rows.append([trim(m.generate(**inp, **kw)[0, pl:].cpu()) for _ in range(k)])
        for seqs in rows:
            seqs = seqs * (n // k)
            results.append([(s, TOK.decode(s, skip_special_tokens=True)) for s in seqs])
    return results

@torch.no_grad()
def generate_many(prompts, n, max_new_tokens, temperature, adapter=True, gpu1=False):
    \"\"\"Returns [[(ids, text)] * n] per prompt. adapter=False samples the frozen base model; gpu1=True runs it on
    the cuda:1 copy when there is one.\"\"\"
    global FROZEN
    TOK.padding_side = "left"
    if gpu1 and not adapter and FROZEN is not None:
        try:
            return _generate_on(FROZEN, "cuda:1", prompts, n, max_new_tokens, temperature, CONFIG["GEN_BATCH_GPU1"])
        except Exception as e:
            print("cuda:1 generation failed; frozen roles move to the policy with LoRA disabled:", repr(e)[:300])
            FROZEN = None
    with POLICY_LOCK:
        FastVisionModel.for_inference(model)
        with (contextlib.nullcontext() if adapter else model.disable_adapter()):
            return _generate_on(model, "cuda", prompts, n, max_new_tokens, temperature, CONFIG["GEN_BATCH"])

class LocalBackend:
    \"\"\"`complete(messages, temperature, role)` for caption_game.play, run on the local model(s).

    Calls made while play() coroutines are gathered are collected and generated together; identical messages
    (the G samples of one prompt) become one prompt with num_return_sequences=G. Roles in `trained` use the
    LoRA policy (cuda:0), every other role the frozen base model. With a cuda:1 copy, the two GPUs work in
    parallel: frozen work goes to cuda:1, and when there is no policy work in a round it is split across both.\"\"\"

    def __init__(self, trained=("forger",)):
        self.trained, self.pending, self.flusher = set(trained), [], None

    async def complete(self, messages, temperature, role):
        fut = asyncio.get_running_loop().create_future()
        self.pending.append((messages, temperature, role, fut))
        if self.flusher is None or self.flusher.done():
            self.flusher = asyncio.ensure_future(self._flush())
        return await fut

    async def _flush(self):
        for _ in range(50):  # let every gathered coroutine reach its complete() call
            await asyncio.sleep(0)
        batch, self.pending = self.pending, []
        groups = {}
        for messages, temp, role, fut in batch:
            groups.setdefault((role, temp), {}).setdefault(id(messages), (messages, []))[1].append(fut)
        # jobs: (items, n, limit, temp, adapter, gpu1)
        jobs = []
        for (role, temp), prompts in groups.items():
            by_n = {}
            for messages, futs in prompts.values():
                by_n.setdefault(len(futs), []).append((messages, futs))
            limit = CONFIG["CG_JUDGE_MAX_TOKENS"] if role == "judge" else CONFIG["CG_CAPTION_MAX_TOKENS"]
            for n, items in by_n.items():
                jobs.append([items, n, limit, temp, role in self.trained, False])
        policy_busy = any(j[4] for j in jobs)
        split = []
        for j in jobs:
            if j[4] or FROZEN is None:
                split.append(j)
            elif policy_busy:
                j[5] = True; split.append(j)
            else:  # only frozen work this round: give cuda:1 about 60% (cuda:0 also does LoRA bookkeeping)
                cut = max(1, round(0.6 * len(j[0])))
                split.append([j[0][:cut], *j[1:5], True])
                if j[0][cut:]:
                    split.append([j[0][cut:], *j[1:5], False])
        lanes = {True: [j for j in split if j[5]], False: [j for j in split if not j[5]]}

        def run(lane):
            for items, n, limit, temp, adapter, gpu1 in lane:
                try:
                    outs = generate_many([m for m, _ in items], n, limit, temp, adapter=adapter, gpu1=gpu1)
                except Exception as e:
                    outs = e
                yield items, outs

        def drain(lane):
            return list(run(lane))

        with ThreadPoolExecutor(2) as ex:
            done = [f.result() for f in [ex.submit(drain, lanes[True]), ex.submit(drain, lanes[False])]]
        for lane in done:
            for items, outs in lane:
                if isinstance(outs, Exception):
                    for _, futs in items:
                        for f in futs:
                            f.set_exception(outs)
                    continue
                for (_, futs), samples in zip(items, outs):
                    for f, (ids, text) in zip(futs, samples):
                        msg = SimpleNamespace(content=text, reasoning=None, model_extra={})
                        f.set_result(SimpleNamespace(choices=[SimpleNamespace(message=msg, logprobs=None)], ids=ids))

def play_images(images, game, temperature=None):
    \"\"\"caption_game.play on every image, with batched local generation (in a worker thread: Jupyter's
    own event loop is already running).\"\"\"
    async def go():
        backend = LocalBackend()
        return await asyncio.gather(*[CG.play(game, backend.complete, img, name, CONFIG["CG_MASK_RATIO"], rng,
                                              temperature) for name, img, rng in images])
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, go()).result()

# ---- checkpoints (local + optional HF Hub for resume across 12 h sessions)
from kaggle_secrets import UserSecretsClient
def secret(name):
    try:
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None

HF_TOKEN = secret("HF_TOKEN") if CONFIG["HF_REPO"] else None
if CONFIG["HF_REPO"] and not HF_TOKEN:
    print("HF_REPO is set but the Kaggle secret HF_TOKEN is missing: checkpoints stay local only")

def save(step, optimizer, scaler):
    ck = OUT / "latest"
    ck.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ck)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, ck / "trainer.pt")
    (ck / "config.json.run").write_text(json.dumps({**CONFIG, "repo_commit": REPO_COMMIT, "step": step}, indent=1))
    if HF_TOKEN:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(CONFIG["HF_REPO"], private=True, exist_ok=True)
        api.upload_folder(folder_path=str(ck), repo_id=CONFIG["HF_REPO"], path_in_repo=f"{CONFIG['ARM']}/latest",
                          commit_message=f"{CONFIG['ARM']} step {step}")
    print(f"saved step {step}")

def try_resume(optimizer, scaler):
    if not (CONFIG["RESUME"] and HF_TOKEN):
        return 0
    from huggingface_hub import snapshot_download
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    try:
        d = Path(snapshot_download(CONFIG["HF_REPO"], token=HF_TOKEN, allow_patterns=[f"{CONFIG['ARM']}/latest/*"]))
    except Exception as e:
        print("no checkpoint to resume:", repr(e)[:200]); return 0
    ck = d / CONFIG["ARM"] / "latest"
    if not (ck / "trainer.pt").exists():
        return 0
    set_peft_model_state_dict(model, load_file(ck / "adapter_model.safetensors"))
    state = torch.load(ck / "trainer.pt", map_location="cuda", weights_only=False)
    optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
    print(f"resumed {CONFIG['ARM']} from step {state['step']}")
    return state["step"]
""")

code("""
# Cell 6: training. GRPO, on-policy with one update per batch (so the PPO ratio is 1), no KL (as in SSL-R1),
# group-normalized advantages, token-level loss normalization over the whole step.
import random, statistics

WANDB_KEY = secret("WANDB_API_KEY")
run = None
if WANDB_KEY:
    import wandb
    wandb.login(key=WANDB_KEY)
    run = wandb.init(project=CONFIG["WANDB_PROJECT"], name=f"kaggle-{CONFIG['ARM']}",
                     config={**CONFIG, "repo_commit": REPO_COMMIT}, resume="allow", id=f"kaggle-{CONFIG['ARM']}")

def log(row):
    with open(OUT / "train_metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\\n")
    if run:
        run.log(row, step=row["step"])
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})

def append(name, rows):
    with open(OUT / name, "a") as f:
        f.writelines(json.dumps(r, default=str) + "\\n" for r in rows)

optimizer = torch.optim.AdamW(trainable, lr=CONFIG["LR"], betas=(0.9, 0.99), weight_decay=0.0)
# init_scale 2**10 (default 2**16): fewer skipped (inf/nan) updates while the scale settles; each step is costly.
scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 10, enabled=AMP_DTYPE == torch.float16)
start = try_resume(optimizer, scaler)
import dataclasses
(OUT / "run_config.json").write_text(json.dumps(
    {**CONFIG, "repo_commit": REPO_COMMIT,
     "objective": train_obj.describe() if GAME == "puzzles" else dataclasses.asdict(CG_CFG)},
    indent=1, default=str))

def puzzle_step(step):
    puzzles = train_obj.batch(train_src, "train", step, CONFIG["PROMPTS_PER_STEP"])
    rollouts, rewards, accs, lens, samples = [], [], [], [], []
    for pz in puzzles:
        inputs = encode(pz)
        seqs, texts = generate(inputs, CONFIG["GROUP_SIZE"])
        scores = [train_obj.score(pz, t) for t in texts]
        r = [s.reward for s in scores]
        rewards += r; accs += [s.accuracy for s in scores]; lens += [len(s) for s in seqs]
        samples.append({"step": step, "id": pz.meta["puzzle_id"], "task": pz.task, "answer": pz.answer,
                        "rewards": r, "response": texts[0]})
        adv = group_advantages(r)
        if adv:
            rollouts += [(inputs, s, a) for s, a in zip(seqs, adv)]
    append("samples.jsonl", samples)
    return rollouts, {"train/reward": statistics.mean(rewards), "train/accuracy": statistics.mean(accs),
                      "train/completion_tokens": statistics.mean(lens),
                      "train/truncated": sum(l >= CONFIG["MAX_NEW_TOKENS"] for l in lens) / len(lens),
                      "train/groups_with_signal": len(rollouts) / CONFIG["GROUP_SIZE"] / len(puzzles)}

def caption_step(step):
    images = CG.step_images(train_src, CONFIG["SEED"], "train", step, CONFIG["CG_IMAGES_PER_STEP"])
    results = play_images(images, CG_CFG)
    rollouts, samples = [], []
    for res in results:
        fakes = res.forgeries
        adv = group_advantages([f.reward for f in fakes])
        if adv:
            inputs = encode_messages(fakes[0].messages)  # all forgeries of an image share one prompt
            rollouts += [(inputs, f.completion.ids, a) for f, a in zip(fakes, adv)]
        samples.append({"step": step, "image": res.image, "box": res.box,
                        "real": [c.text for c in res.captions], "forged": [f.text for f in fakes],
                        "forger_reward": [f.reward for f in fakes], "gate": [f.metadata.get("gate") for f in fakes]})
    append("samples.jsonl", samples)
    stats = {f"train/{k}": v for k, v in CG.mean_stats(results).items()}
    stats["train/groups_with_signal"] = len(rollouts) / CONFIG["CG_CAPTION_SAMPLES"] / len(results)
    return rollouts, stats

def caption_evaluate(step):
    \"\"\"Greedy game on fixed val2017 images: one caption, one forgery, one judgement each (as on W&B).\"\"\"
    images = CG.step_images(val_src, CONFIG["SEED"], "val", 0, CONFIG["CG_VAL_IMAGES"])
    results = play_images(images, dataclasses.replace(CG_CFG, caption_samples=1, judge_samples=1), temperature=0.0)
    append(f"val_games_step{step:05d}.jsonl", [
        {"image": r.image, "box": r.box, "real": r.captions[0].text, "forged": r.forgeries[0].text,
         "forger_reward": r.forgeries[0].reward, "stats": r.stats} for r in results])
    return {f"val/{k}": v for k, v in CG.mean_stats(results).items()}

train_step = puzzle_step if GAME == "puzzles" else caption_step
val_fn = evaluate if GAME == "puzzles" else caption_evaluate

t0 = time.time()
if start == 0 and CONFIG["VAL_EVERY"] and CONFIG["VAL_AT_START"]:
    log({"step": 0, **val_fn(0)})
step = start
for step in range(start + 1, CONFIG["STEPS"] + 1):
    ts = time.time()
    torch.manual_seed(CONFIG["SEED"] * 100003 + step)
    rollouts, row = train_step(step)
    gen_s = time.time() - ts
    row = {"step": step, **row, "train/grad_norm": policy_update(rollouts), "time/gen_s": gen_s,
           "time/step_s": time.time() - ts}
    del rollouts
    if CONFIG["VAL_EVERY"] and step % CONFIG["VAL_EVERY"] == 0:
        row.update(val_fn(step)); row.update(lora_delta_norms())
    log(row)
    out_of_time = (time.time() - t0) / 3600 > CONFIG["TIME_BUDGET_H"]
    if step % CONFIG["SAVE_EVERY"] == 0 or step == CONFIG["STEPS"] or out_of_time:
        save(step, optimizer, scaler)
    if out_of_time:
        print(f"time budget reached at step {step}; rerun the same ARM with HF_REPO set to continue")
        break
""")

code("""
# Cell 7: summary. The adapter (LoRA on vision encoder + LLM when FINETUNE_VISION) is in out/<ARM>/latest.
print(json.dumps(lora_delta_norms(), indent=1))
if run:
    run.finish()
for p in sorted(OUT.iterdir()):
    print(p)
""")

def build(path: Path) -> None:
    def lines(src: str) -> list[str]:
        parts = src.split("\n")
        return [l + "\n" for l in parts[:-1]] + [parts[-1]]

    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": lines(src)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    build(Path(__file__).with_name("sslvlm_kaggle.ipynb"))
