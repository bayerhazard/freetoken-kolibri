# freetoken-kolibri

FreeToken CUDA serving image with **Kolibri-1** model support.

Derived from the official runtime `beclab/leamon2code-freetoken:0.1.3-cu132`
(FreeToken 0.1.3, CUDA 13.2, full accel stack in `/opt/venv`) — this repo only
adds the `kolibri1` model module and registers it.

Image: `ghcr.io/bayerhazard/freetoken-kolibri:<tag>`

## Build

GitHub Actions → **Build freetoken-kolibri image** → Run workflow (or via API).

## Run

```
ft serve --model audreyt/Kolibri-1-NVFP4-W4A16 \
  --host 0.0.0.0 --port 8080 \
  --moe-strategy hybrid --moe-cache-auto \
  --kv-reserve-tokens 131072 --max-extend-length 8192 \
  --reasoning-parser kolibri1
```

## Notes

- Kolibri arch: GQA 48/4 + qk-norm; 40 sliding (513, RoPE) + 10 full (RNoPE)
  layers; 384 experts top-6 + 1 shared, sigmoid+bias routing; sandwich norms.
- The module is written against FreeToken `main`; the base image is 0.1.3 —
  validate imports on first run.
