# Kaggle training (GPU, LoRA can include the vision encoder)

W&B Serverless RL is for quick idea checks; once an idea looks promising, train it here on a free Kaggle GPU.
The notebook reuses this repo's task generators and reward unchanged (it clones `REPO_REF`, default
`hard-negative-inpainting`), and runs GRPO locally on a 4-bit Qwen3-VL with Unsloth LoRA.

| file | what |
|---|---|
| `sslvlm_kaggle.ipynb` | the notebook (generated: edit `build_notebook.py`, then `python kaggle/build_notebook.py`) |
| `kernel-metadata.json` | Kaggle kernel template: private, GPU, internet on, COCO 2017 dataset `awsaf49/coco-2017-dataset` |
| `run.sh` | push one arm, poll every 2 min, download outputs to `kaggle/output/sslvlm-<arm>/` |

```bash
pip install -U kaggle            # token in ~/.kaggle/kaggle.json (never commit or share it)
./kaggle/run.sh <kaggle_user> A-spatial                    # 60-step pipeline check
./kaggle/run.sh <kaggle_user> B-oracle STEPS=300 HF_REPO=<hf_user>/sslvlm-kaggle
```

Arms (cell 2): `A-spatial` (SpatialSSRL distractors), `B-oracle`, `C-mixed`, `D-random` (hard-negative plan,
phase 1), `ssl-r1` (all five SSL-R1 tasks), and `capgame-v7` (masked caption game, below). Any `CONFIG` key can
be overridden as `KEY=VALUE`.
Oracle arms build the COCO label index from the dataset's `instances_*.json` on first use.

Training: on-policy GRPO, one update per batch, group-normalized advantages, no KL (SSL-R1), token-level loss
normalization, fp16 autocast + GradScaler on T4. Thinking is off and completions are capped at 768 tokens:
a T4 cannot afford the 16k-token thinking budget used on W&B, so results here are not directly comparable to
the Qwen3.8-27B thinking runs.

Vision encoder: `FINETUNE_VISION=True` (default) puts LoRA on the vision tower too. `lora_delta/vision` in the
metrics shows how far it moved; checkpoints trained this way should get the eval protocol's dimension F probes.

Long runs: a Kaggle session stops at ~12 h (the notebook saves at `TIME_BUDGET_H=11`). Set `HF_REPO` and add
the Kaggle secret `HF_TOKEN`; rerunning the same arm then resumes from `<HF_REPO>/<ARM>/latest`. Add the
secret `WANDB_API_KEY` to log to W&B project `sslvlm-kaggle`.

Caption game (`capgame-v7`): clones `masked-caption-game` and calls its `caption_game.play` unchanged, with a
local backend: the forger samples from the LoRA policy, the captioner and judge from the frozen base model (LoRA
disabled), i.e. the forger-only setup of runs v4-v7. Settings follow v6 (mask 0.3, 8 images x 4 captions x 4
judgements, recall content, gate penalty -0.5, thinking off) with lr 3e-5 and 100 steps. All pending calls of a
step are generated together (identical prompts as one prompt with `num_return_sequences`). With two GPUs
(Kaggle T4 x2) a second copy of the base model on cuda:1 serves the frozen roles in parallel with the policy;
judge samples per pair are 2 (W&B runs: 4) and the judge cap is 512 tokens, to fit more steps in 12 h. The base model is
Qwen3-VL-8B, not Gemma 4, so compare against this run's own step 0, not against v6.

Referring game (`refgame`, `refgame-oracle`): clones `referring-game` and uses `serverless/refgame.py`. The policy
sees an image with a red box and describes the region in at most 20 words; the frozen base model (LoRA disabled)
sees only 4 crops (the region and 3 look-alikes, no positions) and the description, and its probability of the
right letter, averaged over two candidate orders, is the reward (minus 0.05 per word beyond 20). There is no
recall or content term, so a longer or invented description cannot pay. `refgame`: one same-image region and two
colour/sharpness-matched regions of other images; `refgame-oracle`: COCO-label-matched regions. The step-0 eval
also reports blind (no description) and colour-only listener floors. Base-model checks without training:
`python -m serverless.refgame phase0 --images <val2017> --out-dir <dir> --scheme name='{...}'`.
