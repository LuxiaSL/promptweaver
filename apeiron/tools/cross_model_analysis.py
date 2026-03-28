#!/usr/bin/env python3
"""Quick analysis: how correlated are OpenCLIP and T5-XXL similarity judgments?"""

import numpy as np
import torch
from apeiron.tools.embeddings import load_components_yaml

categories = load_components_yaml("apeiron/data/curated_v2.yaml")

all_words = []
cat_labels = []
cat_ranges = {}
for cat, words in sorted(categories.items()):
    start = len(all_words)
    all_words.extend(words)
    cat_labels.extend([cat] * len(words))
    cat_ranges[cat] = (start, len(all_words))

print(f"{len(all_words)} words")

# OpenCLIP
print("Loading OpenCLIP ViT-bigG/14...")
import open_clip
device = "cpu"
model_oc, _, _ = open_clip.create_model_and_transforms(
    "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device
)
tok_oc = open_clip.get_tokenizer("ViT-bigG-14")
model_oc.eval()

print("Encoding OpenCLIP...")
with torch.no_grad():
    oc_embs = []
    for i in range(0, len(all_words), 64):
        batch = all_words[i:i+64]
        tokens = tok_oc(batch).to(device)
        feat = model_oc.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        oc_embs.append(feat.cpu().float().numpy())
    oc_emb = np.concatenate(oc_embs)

del model_oc
print("OpenCLIP done.")

# T5-XXL
print("Loading T5-v1_1-XXL...")
from transformers import T5EncoderModel, AutoTokenizer
t5_model = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl").to(device).eval()
t5_tok = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl", legacy=True)

print("Encoding T5-XXL...")
with torch.no_grad():
    t5_embs = []
    for i in range(0, len(all_words), 32):
        batch = all_words[i:i+32]
        inputs = t5_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = t5_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hidden = out.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        t5_embs.append(pooled.cpu().float().numpy())
    t5_emb = np.concatenate(t5_embs)

del t5_model
print("T5-XXL done.")

# Analysis
print()
print("=" * 70)
print("OPENCLIP vs T5-XXL ORTHOGONALITY ANALYSIS")
print("=" * 70)

header = f"{'Category':<25} {'OC Mean':>7} {'T5 Mean':>7} {'Pearson r':>9} {'Redund Jaccard':>14}"
print(f"\n{header}")
print("-" * 65)

overall_oc = []
overall_t5 = []

for cat, (start, end) in sorted(cat_ranges.items()):
    oc_cat = oc_emb[start:end]
    t5_cat = t5_emb[start:end]
    n = end - start

    oc_sim = oc_cat @ oc_cat.T
    t5_sim = t5_cat @ t5_cat.T

    mask = ~np.eye(n, dtype=bool)
    oc_flat = oc_sim[mask]
    t5_flat = t5_sim[mask]

    r = float(np.corrcoef(oc_flat, t5_flat)[0, 1])

    # Redundancy agreement (Jaccard on pairs > 0.85)
    oc_r = set(zip(*np.where((oc_sim > 0.85) & mask)))
    t5_r = set(zip(*np.where((t5_sim > 0.85) & mask)))
    both = len(oc_r & t5_r)
    either = len(oc_r | t5_r) or 1
    jaccard = both / either

    print(f"  {cat:<23} {oc_flat.mean():>7.3f} {t5_flat.mean():>7.3f} {r:>9.3f} {jaccard:>13.1%}")

    overall_oc.extend(oc_flat.tolist())
    overall_t5.extend(t5_flat.tolist())

r_global = float(np.corrcoef(overall_oc, overall_t5)[0, 1])
print(f"\nGlobal Pearson r: {r_global:.3f}")
print("  r < 0.3 = largely independent (dual-space very valuable)")
print("  r 0.3-0.6 = moderately correlated (dual-space adds signal)")
print("  r > 0.6 = highly correlated (diminishing returns from dual)")

# Biggest disagreements
print()
print("=" * 70)
print("BIGGEST DISAGREEMENTS")
print("=" * 70)

n = len(all_words)
oc_full = oc_emb @ oc_emb.T
t5_full = t5_emb @ t5_emb.T
diff = oc_full - t5_full

mask_full = ~np.eye(n, dtype=bool)
diff_masked = diff.copy()
diff_masked[~mask_full] = 0

# OpenCLIP says similar, T5 says different
top_oc = np.unravel_index(np.argsort(diff_masked.ravel())[-10:], diff.shape)
print("\nOpenCLIP groups together, T5-XXL separates:")
print(f"  {'Word A':<28} {'Word B':<28} {'OC':>5} {'T5':>5} {'Gap':>5}")
print(f"  {'-'*73}")
seen = set()
for i, j in zip(top_oc[0][::-1], top_oc[1][::-1]):
    pair = tuple(sorted([i, j]))
    if pair in seen:
        continue
    seen.add(pair)
    print(f"  {all_words[i]:<28} {all_words[j]:<28} {oc_full[i,j]:>5.2f} {t5_full[i,j]:>5.2f} {diff[i,j]:>+5.2f}")

# T5 says similar, OpenCLIP says different
top_t5 = np.unravel_index(np.argsort(diff_masked.ravel())[:10], diff.shape)
print("\nT5-XXL groups together, OpenCLIP separates:")
print(f"  {'Word A':<28} {'Word B':<28} {'OC':>5} {'T5':>5} {'Gap':>5}")
print(f"  {'-'*73}")
seen = set()
for i, j in zip(top_t5[0], top_t5[1]):
    pair = tuple(sorted([i, j]))
    if pair in seen:
        continue
    seen.add(pair)
    print(f"  {all_words[i]:<28} {all_words[j]:<28} {oc_full[i,j]:>5.2f} {t5_full[i,j]:>5.2f} {diff[i,j]:>+5.2f}")

print("\nDone!")
