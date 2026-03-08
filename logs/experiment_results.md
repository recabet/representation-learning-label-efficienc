
# Experiment Results — Full Analysis

All **10 experiments** completed successfully (5 label percentages × 2 SimCLR variants).

---

## Accuracy Summary Table

| Labels | Scratch (no SimCLR) | ImageNet Fine-tune | SimCLR Probe (scratch) | SimCLR Probe (PT init) |
|-------:|--------------------:|-------------------:|-----------------------:|-----------------------:|
| **10%** | 41.1% | **79.7%** | 74.0% | 75.5% |
| **25%** | 54.2% | **83.8%** | 78.9% | 79.4% |
| **50%** | 61.5% | **84.3%** | 80.7% | 81.6% |
| **75%** | 69.1% | **86.9%** | 81.2% | 82.7% |
| **100%** | 73.2% | **87.3%** | 82.2% | **83.1%** |

> **First 5 experiments** = `resnet18` variant (SimCLR pretrained from scratch)
> **Last 5 experiments** = `resnet18_pt` variant (SimCLR pretrained with ImageNet init)

---

## Key Findings

### ✅ 1. SimCLR is working — huge improvement over training from scratch

| Labels | Scratch → SimCLR Probe (scratch) | Improvement |
|-------:|---------------------------------:|------------:|
| 10% | 41.1% → 74.0% | **+32.9%** |
| 25% | 54.2% → 78.9% | **+24.7%** |
| 50% | 61.5% → 80.7% | **+19.2%** |
| 75% | 69.1% → 81.2% | **+12.1%** |
| 100% | 73.2% → 82.2% | **+9.0%** |

SimCLR dominates at low label regimes. With only 10% labels, SimCLR achieves **74%** vs scratch's **41%** — nearly 2× better!

### ✅ 2. SimCLR with ImageNet init gives ~1% boost over scratch init

| Labels | SimCLR (scratch init) | SimCLR (ImageNet init) | Δ |
|-------:|----------------------:|-----------------------:|---:|
| 10% | 74.0% | 75.5% | +1.5% |
| 25% | 78.9% | 79.4% | +0.5% |
| 50% | 80.7% | 81.6% | +0.9% |
| 75% | 81.2% | 82.7% | +1.5% |
| 100% | 82.2% | 83.1% | +0.9% |

The pretrained init is **consistently but only marginally better** (~1%). This aligns with the loss analysis — both converged to nearly the same loss.

### ⚠️ 3. ImageNet fine-tuning still beats SimCLR probe

| Labels | ImageNet Fine-tune | Best SimCLR Probe | Gap |
|-------:|-------------------:|------------------:|----:|
| 10% | **79.7%** | 75.5% | -4.2% |
| 25% | **83.8%** | 79.4% | -4.4% |
| 50% | **84.3%** | 81.6% | -2.7% |
| 75% | **86.9%** | 82.7% | -4.2% |
| 100% | **87.3%** | 83.1% | -4.2% |

ImageNet fine-tuning beats SimCLR by ~3-4% at all label percentages. This is expected because:
- ImageNet was pretrained on **1.2M labeled images** (massive supervised signal)
- SimCLR only had STL-10's **100K unlabeled images**
- The linear probe **freezes** the encoder — fine-tuning adapts all layers

### ⚠️ 4. Scratch baselines are inconsistent between the two runs

| Labels | Scratch (run 1) | Scratch (run 2) | Δ |
|-------:|----------------:|----------------:|---:|
| 10% | 41.1% | 39.1% | -2.0% |
| 25% | 54.2% | 56.0% | +1.8% |
| 50% | 61.5% | 63.1% | +1.6% |
| 75% | 69.1% | 65.0% | -4.1% |
| 100% | 73.2% | 71.1% | -2.1% |

The scratch baselines vary by ±2-4% between runs. This is because `np.random.choice` samples different training subsets each time (no fixed seed for subset selection). This variance also affects the SimCLR/ImageNet results slightly.

---

## Training Observations

### Overfitting in Scratch Models
- **10% labels (500 samples)**: Train acc reaches ~85% but test acc is only 41% → **severe overfitting**
- **100% labels (5000 samples)**: Train acc reaches ~93% but test acc is 73% → **still overfitting**
- The scratch model memorizes the small training set

### Linear Probe Convergence
- SimCLR probe training loss plateaus around **0.53–0.63** at epoch 100
- Train accuracy plateaus at ~81-82% — it cannot overfit because the encoder is **frozen**
- This is the correct behavior for a linear probe

### ImageNet Fine-tuning
- Uses lr=1e-4 (10× lower than scratch) to avoid destroying pretrained features
- Train acc reaches 96-98% — no convergence issues
- Best overall performance at all label levels

---

## Verdict

| Aspect | Status |
|--------|--------|
| SimCLR pretraining | ✅ **Working correctly** — 83% with 100% labels |
| Label efficiency | ✅ **Proven** — SimCLR at 10% labels (74%) beats scratch at 100% labels (73%) |
| ImageNet init for SimCLR | ⚠️ **Marginal** — only ~1% better, not worth the conceptual impurity |
| vs ImageNet fine-tuning | ⚠️ **~4% gap** — expected given ImageNet's supervised advantage |
| Scratch baseline | ⚠️ **High variance** — needs fixed seed for reproducibility |

### The headline result:
> **SimCLR with just 10% labeled data (74%) matches a fully supervised model trained on 100% labels (73%)**

This is exactly the point of your project — *representation learning improves label efficiency*.

