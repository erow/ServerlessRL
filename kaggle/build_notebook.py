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
    TIME_BUDGET_H=11.0,              # stop and save before Kaggle's 12 h limit
    HF_REPO="",                      # e.g. "you/sslvlm-kaggle"; needs Kaggle secret HF_TOKEN; enables resume
    RESUME=True,                     # resume ARM from HF_REPO/<ARM>/latest if present
    WANDB_PROJECT="sslvlm-kaggle",   # used only if Kaggle secret WANDB_API_KEY exists
    SEED=0,
)
CONFIG_OVERRIDES = {}
CONFIG.update(CONFIG_OVERRIDES)

# Arms of the hard-negative inpainting study (plan doc phase 1) plus the SSL-R1 multi-task baseline.
# "oracle_index" = "auto" builds the COCO label index for the split on first use.
SQ = {"mask": "square"}
ARMS = {
    "A-spatial": ("inpainting", {"inpainting": {**SQ, "distractors": "spatial"}}),
    "B-oracle":  ("inpainting", {"inpainting": {**SQ, "distractors": "oracle-cat,oracle-dist,oracle-dist",
                                                "oracle_index": "auto"}}),
    "C-mixed":   ("inpainting", {"inpainting": {**SQ, "distractors": "rot90,oracle-cat,oracle-dist",
                                                "oracle_index": "auto"}}),
    "D-random":  ("inpainting", {"inpainting": {**SQ, "distractors": "random"}}),
    "ssl-r1":    ("ssl-r1", {}),
}
TASKS, TASK_PARAMS = ARMS[CONFIG["ARM"]]
CONFIG.setdefault("TASKS", TASKS)
CONFIG.setdefault("TASK_PARAMS", TASK_PARAMS)
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
from serverless.objective import Objective

train_src, val_src = ImageSource(TRAIN_DIR), ImageSource(VAL_DIR)
train_obj = Objective.from_spec(CONFIG["TASKS"], task_params=task_params("train2017"), seed=CONFIG["SEED"],
                                format_reward=CONFIG["FORMAT_REWARD"])
val_obj = Objective.from_spec(CONFIG["TASKS"], task_params=task_params("val2017"), seed=CONFIG["SEED"],
                              format_reward=CONFIG["FORMAT_REWARD"])
print(len(train_src), "train images,", len(val_src), "val images")
p = train_obj.batch(train_src, "train", 0, 1)[0]
print(p.task, p.answer, p.text[:300])
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
    plen = inputs["input_ids"].shape[1]
    ids = torch.cat([inputs["input_ids"][0], comp]).unsqueeze(0)
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
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
# Cell 6: GRPO training loop. On-policy, one update per batch (so the PPO ratio is 1), no KL (as in SSL-R1),
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

optimizer = torch.optim.AdamW(trainable, lr=CONFIG["LR"], betas=(0.9, 0.99), weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)
start = try_resume(optimizer, scaler)
(OUT / "run_config.json").write_text(json.dumps({**CONFIG, "repo_commit": REPO_COMMIT,
                                                 "objective": train_obj.describe()}, indent=1, default=str))
t0 = time.time()
if start == 0 and CONFIG["VAL_EVERY"]:
    log({"step": 0, **evaluate(0)})

step = start
for step in range(start + 1, CONFIG["STEPS"] + 1):
    ts = time.time()
    torch.manual_seed(CONFIG["SEED"] * 100003 + step)
    puzzles = train_obj.batch(train_src, "train", step, CONFIG["PROMPTS_PER_STEP"])
    rollouts, rewards, accs, lens, samples = [], [], [], [], []
    for pz in puzzles:
        inputs = encode(pz)
        seqs, texts = generate(inputs, CONFIG["GROUP_SIZE"])
        scores = [train_obj.score(pz, t) for t in texts]
        r = [s.reward for s in scores]
        mu, sd = statistics.mean(r), statistics.pstdev(r)
        rewards += r; accs += [s.accuracy for s in scores]; lens += [len(s) for s in seqs]
        samples.append({"step": step, "id": pz.meta["puzzle_id"], "task": pz.task, "answer": pz.answer,
                        "rewards": r, "response": texts[0]})
        if sd > 1e-6:  # a group with equal rewards carries no GRPO signal
            rollouts += [(inputs, s, (ri - mu) / (sd + 1e-4)) for s, ri in zip(seqs, r)]
    gen_s = time.time() - ts

    FastVisionModel.for_training(model)
    optimizer.zero_grad(set_to_none=True)
    n_tok = max(1, sum(len(s) for _, s, _ in rollouts))
    for inputs, comp, adv in rollouts:
        loss = -adv * completion_logprob(inputs, comp) / n_tok
        scaler.scale(loss).backward()
    grad_norm = 0.0
    if rollouts:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, CONFIG["MAX_GRAD_NORM"]).item()
        scaler.step(optimizer)
        scaler.update()
    with open(OUT / "samples.jsonl", "a") as f:
        f.writelines(json.dumps(s) + "\\n" for s in samples)
    row = {"step": step, "train/reward": statistics.mean(rewards), "train/accuracy": statistics.mean(accs),
           "train/completion_tokens": statistics.mean(lens), "train/truncated": sum(l >= CONFIG["MAX_NEW_TOKENS"] for l in lens) / len(lens),
           "train/groups_with_signal": len(rollouts) / CONFIG["GROUP_SIZE"] / len(puzzles),
           "train/grad_norm": grad_norm, "time/gen_s": gen_s, "time/step_s": time.time() - ts}
    if CONFIG["VAL_EVERY"] and step % CONFIG["VAL_EVERY"] == 0:
        row.update(evaluate(step)); row.update(lora_delta_norms())
    log(row)
    out_of_time = (time.time() - t0) / 3600 > CONFIG["TIME_BUDGET_H"]
    if step % CONFIG["SAVE_EVERY"] == 0 or step == CONFIG["STEPS"] or out_of_time:
        save(step, optimizer, scaler)
    if out_of_time:
        print(f"time budget reached at step {step}; rerun with RESUME=True and HF_REPO set to continue")
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
