# ToTo Lottery Prediction Report

**Generated:** 2026-01-27 21:54:57

**Model:** Deep Learning LSTM with Attention Mechanism

---

## Predicted Next Draw

### Winning Numbers
```
13  15  16  21  36  43
```

### Additional Number
```
17
```

---

## Statistical Analysis

| Metric | Value |
|--------|-------|
| Sum | 144 |
| Average | 24.00 |
| Low/High Ratio | 4 / 2 |
| Odd/Even Ratio | 4 / 2 |
| Range 1-10 | 0 |
| Range 11-20 | 3 |
| Range 21-30 | 1 |
| Range 31-40 | 1 |
| Range 41-50 | 1 |

---

## Model Performance

| Dataset | Loss | Win Acc | Addl Acc |
|---------|------|---------|----------|
| Training | 2.350787 | 0.877296 | 0.031226 |
| Validation | 2.407519 | 0.877551 | 0.028037 |
| Holdout | 2.390631 | 0.877551 | 0.016760 |

**Loss:** Binary Cross-Entropy (wins) + Categorical Cross-Entropy (addl)

**Architecture:**
- Bidirectional LSTM (256 + 128 units)
- Self-Attention Mechanism
- Dense Layers (256 → 128 → 64)
- Multi-head Outputs (wins multi-label + addl softmax)
- Dropout Regularization (tuned)
- Batch Normalization

**Checkpoint:** models\toto_model_best_20260127_202736.keras

**Selected Hyperparameters:**
- Sequence Length: 30
- Dropout Rate: 0.3
- L2 Weight: 0.0

- Hyperparameter Selection: Walk-forward (5 folds)
- Final Holdout Fraction: 0.10

**Feature Engineering:**
- Feature Scaler: standard
- Rolling Frequency Window: 20 draws
- Lagged Metrics: Enabled
- Seasonality Features: Day-of-week & Month (sin/cos)
- Time-since-last-seen Features: Enabled
- Target: Win multi-label + additional one-hot

**Training Details:**
- Historical Draws Analyzed: 1811
- Training Sequences: 1281
- Validation Sequences: 321
- Holdout Sequences: 179
- Sequence Length: 30 draws
- Features Used: 131
- Epochs Trained: 58

---

## Disclaimer

> **Important:** This prediction is generated using advanced deep learning algorithms
> analyzing historical patterns. However, lottery draws are designed to be random,
> and past performance does not guarantee future results. This model identifies
> statistical patterns but cannot predict truly random events with certainty.
> 
> Use this prediction for entertainment and educational purposes only.
