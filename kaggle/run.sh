#!/usr/bin/env bash
# Push the SSLVLM GRPO notebook to Kaggle, run it on a GPU, wait, and download the outputs.
#
#   ./run.sh <kaggle_username> [ARM] [KEY=VALUE ...]
#   ./run.sh alice A-spatial STEPS=200 HF_REPO=alice/sslvlm-kaggle
#   ./run.sh alice B-oracle FINETUNE_VISION=false
#
# ARM: A-spatial | B-oracle | C-mixed | D-random | ssl-r1 (see notebook cell 2). KEY=VALUE overrides any CONFIG
# entry; values are parsed as JSON when possible (numbers, true/false, quoted strings), else kept as strings.
# Each arm gets its own kernel (sslvlm-<arm>), so arms can run side by side within Kaggle's GPU session limit.
# Needs the kaggle CLI with an API token in ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY.
# Optional Kaggle secrets (set in the notebook's Add-ons > Secrets): HF_TOKEN, WANDB_API_KEY.
# Env: ACCELERATOR (default NvidiaTeslaT4), POLL (seconds, default 120), NO_WAIT=1 to push and exit.
set -euo pipefail
[[ $# -ge 1 ]] || { sed -n 2,14p "$0"; exit 1; }
KUSER=$1; ARM=${2:-A-spatial}; shift $(( $# >= 2 ? 2 : 1 ))
HERE=$(cd "$(dirname "$0")" && pwd)
SLUG="sslvlm-$(echo "$ARM" | tr 'A-Z_.' 'a-z--')"
BUILD="$HERE/build/$SLUG"; OUTDIR="$HERE/output/$SLUG"
ACCELERATOR=${ACCELERATOR:-NvidiaTeslaT4}; POLL=${POLL:-120}
command -v kaggle >/dev/null || { echo "kaggle CLI missing: pip install kaggle"; exit 1; }
mkdir -p "$BUILD" "$OUTDIR"

python3 - "$HERE/sslvlm_kaggle.ipynb" "$BUILD/sslvlm_kaggle.ipynb" "$ARM" "$@" <<'PY'
import json, sys
src, dst, arm, *kv = sys.argv[1:]
over = {"ARM": arm}
for item in kv:
    key, sep, val = item.partition("=")
    if not sep:
        sys.exit(f"bad override {item!r}: use KEY=VALUE")
    try:
        val = json.loads(val)
    except json.JSONDecodeError:
        pass
    over[key] = val
nb = json.load(open(src))
hits = 0
for cell in nb["cells"]:
    s = "".join(cell["source"])
    if "\nCONFIG_OVERRIDES = {}\n" in s:
        s = s.replace("\nCONFIG_OVERRIDES = {}\n", f"\nCONFIG_OVERRIDES = {over!r}\n")
        cell["source"] = s.splitlines(keepends=True)
        hits += 1
if hits != 1:
    sys.exit("could not find the CONFIG_OVERRIDES line in the notebook")
json.dump(nb, open(dst, "w"), indent=1)
print("overrides:", over)
PY
sed -e "s/KAGGLE_USERNAME/$KUSER/" -e "s/KERNEL_SLUG/$SLUG/g" "$HERE/kernel-metadata.json" > "$BUILD/kernel-metadata.json"

ACC_ARGS=()
if kaggle kernels push --help 2>&1 | grep -q -- --accelerator; then
  ACC_ARGS=(--accelerator "$ACCELERATOR")
else
  echo "warning: this kaggle CLI has no --accelerator; Kaggle picks the default GPU (pip install -U kaggle)"
fi
kaggle kernels push -p "$BUILD" "${ACC_ARGS[@]}"
echo "pushed https://www.kaggle.com/code/$KUSER/$SLUG"
[[ ${NO_WAIT:-0} == 1 ]] && exit 0

while true; do
  sleep "$POLL"
  st=$(kaggle kernels status "$KUSER/$SLUG" 2>&1 || true)
  echo "$(date +%H:%M:%S) $st"
  low=$(echo "$st" | tr 'A-Z' 'a-z')
  case "$low" in  # e.g. 'user/slug has status "complete"' (transient CLI errors keep polling)
    *"has status"*complete*|*"has status"*error*|*"has status"*cancel*) break ;;
  esac
done
kaggle kernels output "$KUSER/$SLUG" -p "$OUTDIR"
echo "outputs in $OUTDIR (adapter: out/$ARM/latest, metrics: out/$ARM/train_metrics.jsonl)"
case "$low" in *complete*) exit 0 ;; *) exit 1 ;; esac
