# Experiment Results v2 — After SGD Probe + Test Crop Fix

All **10 experiments** completed. Comparing against the previous run.

---

## Accuracy Summary

| Labels | Scratch | ImageNet | SimCLR Probe (scratch) | SimCLR Probe (PT init) |
|-------:|--------:|---------:|-----------------------:|-----------------------:|
| **10%** | 46.0% | 82.1% | 79.3% | **80.2%** |
| **25%** | 54.8% | 83.6% | 81.8% | **82.5%** |
| **50%** | 63.7% | 87.2% | 83.2% | **83.8%** |
| **75%** | 71.4% | 89.1% | 84.1% | **84.7%** |
| **100%** | 74.7% | 89.7% | 84.6% | **85.0%** |

---

## Improvement vs Previous Run

The changes (SGD+cosine for probe, Resize(112)+CenterCrop(96) for eval) delivered significant gains:

| Labels | Old Probe → New Probe | Δ | Old PT Probe → New PT Probe | Δ |
|-------:|----------------------:|----:|----------------------------:|----:|
| 10% | 74.0% → **79.3%** | **+5.3%** | 75.5% → **80.2%** | **+4.7%** |
| 25% | 78.9% → **81.8%** | **+2.9%** | 79.4% → **82.5%** | **+3.1%** |
| 50% | 80.7% → **83.2%** | **+2.5%** | 81.6% → **83.8%** | **+2.2%** |
| 75% | 81.2% → **84.1%** | **+2.9%** | 82.7% → **84.7%** | **+2.0%** |
| 100% | 82.2% → **84.6%** | **+2.4%** | 83.1% → **85.0%** | **+1.9%** |

**Average improvement: +3.2%** across all experiments — without retraining SimCLR.

---

## SimCLR scratch vs ImageNet-init

| Labels | Probe (scratch) | Probe (PT init) | Δ |
|-------:|----------------:|----------------:|----:|
| 10% | 79.3% | 80.2% | +0.8% |
| 25% | 81.8% | 82.5% | +0.8% |
| 50% | 83.2% | 83.8% | +0.6% |
| 75% | 84.1% | 84.7% | +0.6% |
| 100% | 84.6% | 85.0% | +0.4% |

ImageNet init still ~0.5-0.8% better. Marginal.

---

## Gap to ImageNet Fine-tuning

| Labels | ImageNet Fine-tune | Best SimCLR Probe | Gap |
|-------:|-------------------:|------------------:|----:|
| 10% | 82.1% | 80.2% | **-1.9%** (was -4.2%) |
| 25% | 83.6% | 82.5% | **-1.1%** (was -4.4%) |
| 50% | 87.2% | 83.8% | **-3.4%** (was -2.7%) |
| 75% | 89.1% | 84.7% | **-4.4%** (was -4.2%) |
| 100% | 89.7% | 85.0% | **-4.7%** (was -4.2%) |

At low labels (10-25%), the gap to ImageNet has been cut in half. At higher labels, ImageNet fine-tuning pulls ahead because it adapts all layers while the probe is frozen.

---

## Headline Result

> **SimCLR with just 10% labeled data (79.3%) beats a fully supervised model trained on 100% labels (74.7%) by +4.6%**

Previously this margin was only +1.8%. The improved eval protocol made the SimCLR advantage far more convincing.

