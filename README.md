# 🔬 WaferDefect Studio

**Diffusion-based minority-class augmentation + lightweight ViT classifier with Grad-CAM explainability for mixed-type wafer defect detection — a cross-dataset robustness study (WM-811K → MixedWM38).**

Deep learning course project · TensorFlow/Keras · Streamlit console

---

## Research framing

| # | Contribution | What it means |
|---|---|---|
| 1 | **Latent-diffusion minority-class augmentation** | A Stable-Diffusion-*style* model (VAE → conditional UNet in latent space, classifier-free guidance) trained **from scratch** on wafer maps. It synthesizes rare defect classes (Scratch, Donut, Near-full, …) to balance training. |
| 2 | **Lightweight ViT classifier** | A ~470K-param Vision Transformer (8×8 patches → 42 tokens → 6 encoder blocks) benchmarked against the classic CNN baseline. |
| 3 | **Cross-dataset robustness + explainability protocol** | Train on WM-811K, test **zero-shot** on MixedWM38 and sample-wafermap; evaluate with macro-F1, minority-class recall, Grad-CAM (CNN) and attention rollout (ViT). |

**Why it's defensible:** diffusion (not GAN) augmentation, a transformer (not another CNN), and an explicit robustness/explainability evaluation are three distinct, literature-grounded components — while using existing, well-documented public datasets.

---

## Datasets

| Dataset | Size | Role | Source |
|---|---|---|---|
| **WM-811K** | 811,457 maps (25,415 labeled) | primary training | Kaggle `qingyi/wm811k-wafer-map` (`LSWMD.pkl`) |
| **MixedWM38** | 38,015 mixed-type maps | zero-shot cross-dataset test | Kaggle `co1d7era/mixedtype-wafer-defect-datasets` |
| **Sample-WaferMap** | auxiliary | extra benchmark | Kaggle `emphymachine/sample-wafermap-data` |

Place data under `data/` (or use kagglehub on Colab — see the notebook):

```
data/
├── LSWMD.pkl
├── MixedWM38/          # .mat files from the kagglehub download
└── sample_wafermap/
```

**Shared label space (9):** Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none. MixedWM38 mixed-type labels (e.g. `Center-Loc`) map to their primary component.

**Shared die encoding (all datasets):** fail = 1.0 · pass = 0.0 · outside-wafer = 0.5, resized 52×52 nearest-neighbor — so classifiers can't cheat off dataset-specific fingerprints.

---

## Quick start

### 1. Environment
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

### 2. Train the diffusion augmentation model
```bash
python src/train_diffusion.py     # VAE → class-conditional UNet
```
Saves `outputs/diffusion/{vae.keras,unet.keras}` + a per-class preview grid.

### 3. Train the three classifier arms
```bash
python src/train.py               # cnn · vit · vit_diff
python src/train.py --arms vit    # single arm
python src/train.py --max-per-class 300   # fast smoke run
```
Identical train/val split for every arm → fair ablation. Exports `outputs/ablation_results.csv`.

### 4. Evaluate (ablation + cross-dataset + explainability)
```bash
python src/evaluate.py            # all three stages
python src/evaluate.py --skip-explain
```
Generates:
- `outputs/cross_dataset_results.csv` — robustness matrix (macro-F1 train→test)
- `outputs/per_class_f1.csv`
- `outputs/explain/grid_{cnn,vit,vit_diff}.png` — Grad-CAM / rollout grids
- per-arm confusion matrices

### 5. Streamlit console
```bash
streamlit run app.py
```
**WaferDefect Studio** — 6 pages:
| Page | What it shows |
|---|---|
| 🔮 Predict | live inference, confidence gauge, top-k bars, inline Grad-CAM / rollout |
| 🌫 Diffusion Lab | class-conditional wafer-map generation with CFG + step control |
| 📊 Experiments | 3-arm ablation dashboard (acc / macro-F1 / params), per-class F1 chart |
| 🌐 Cross-Dataset | robustness heatmap matrix + raw table |
| 🔍 Explainability | exported explanation grids + upload-your-own explaining |
| ℹ About | methods summary |

5 switchable dark palettes (Nebula Dusk, Terminal Ice, Forest Ember, Sakura Dark, Solar Flare).

---

## Project structure

```
Wafer-MAP/
├── app.py                        # Streamlit research console
├── train_colab.ipynb             # Cloud-GPU training notebook (kagglehub download)
├── data/                         # datasets (see above)
├── src/
│   ├── config.py                 # classes, paths, model-arm registry
│   ├── data/
│   │   └── multi.py              # unified multi-dataset loader + encoding
│   ├── models/
│   │   ├── vit.py                # lightweight ViT (468K params)
│   │   └── cnn.py                # CNN baseline
│   ├── diffusion/
│   │   ├── latent_diffusion.py   # VAE + conditional UNet + DDIM + CFG
│   │   └── sample.py             # app-facing sampling helpers
│   ├── explain.py                # attention rollout + unified dispatch
│   ├── gradcam.py                # Grad-CAM for the CNN arm
│   ├── train_diffusion.py        # diffusion training entry point
│   ├── train.py                  # 3-arm classifier training
│   └── evaluate.py               # ablation + cross-dataset + explain exports
├── outputs/                      # checkpoints, CSVs, figures (created at runtime)
└── requirements.txt
```

---

## Method details

**Latent diffusion (augmentation engine).** Wafer maps are near-binary with huge flat backgrounds, so a tiny VAE (83K params) compresses 52×52×1 → 13×13×4. Diffusion then runs in latent space: cosine noise schedule (400 steps), class-conditional UNet (1.07M params) with FiLM-style embedding injection and 10% label dropout, sampled with 50-step DDIM + classifier-free guidance (scale 2.5). Synthetic minority-class maps are **appended** to the training set (never replacing real data), capped per class.

**Lightweight ViT.** 8×8 non-overlapping patches → linear projection to dim-96 tokens → 6 pre-norm transformer blocks (4 heads, GELU MLP ×2) → mean token pooling → 2-layer head. Attention layers are named (`block_{i}_mha`) so rollout explainability can introspect them.

**Attention rollout.** Per Abnar & Zuidema (2020): recursively multiply normalized attention matrices with the identity (residual), average over heads, then read per-patch importance — reshaped to the 6×6 patch grid and upscaled for overlay.

**Fair-comparison protocol.** Same split, same epochs budget, same class-weighting for CNN and ViT; the `vit_diff` arm differs *only* by the extra diffusion-synthesized samples. `--max-per-class` gives a fast end-to-end dry run before the real training.

---

## Results workflow (fill after training)

| Arm | Params | Val Acc | Macro-F1 | Minority Recall |
|---|---|---|---|---|
| CNN baseline | 356K | — | — | — |
| Lightweight ViT | 469K | — | — | — |
| ViT + Diffusion aug | 469K | — | — | — |

Cross-dataset (macro-F1, zero-shot):

| Train ↓ / Test → | WM-811K | MixedWM38 | Sample |
|---|---|---|---|
| WM-811K | — | — | — |
| MixedWM38 | — | — | — |

---

## Dataset sources

- WM-811K: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- MixedWM38: https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets
- Sample-WaferMap: https://www.kaggle.com/datasets/emphymachine/sample-wafermap-data

## References
- Wu, M.-J. et al. — Mixed-type wafer defect pattern recognition (MixedWM38)
- Nakazawa & Kulkarni — Wafer map defect pattern classification and image retrieval (WM-811K)
- Abnar & Zuidema — Quantifying attention flow in transformers (2020)
- Ho et al. — Denoising diffusion probabilistic models (2020)
- Rombach et al. — High-resolution image synthesis with latent diffusion models (2022)
- Dosovitskiy et al. — An image is worth 16×16 words (ViT, 2021)
