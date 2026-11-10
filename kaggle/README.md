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
phase 1), `ssl-r1` (all five SSL-R1 tasks). Any `CONFIG` key can be overridden as `KEY=VALUE`.
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

Not ported yet: the masked caption game (it needs a frozen captioner and judge per rollout, roughly 3x the
generation cost on one T4).
