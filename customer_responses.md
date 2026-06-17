# ML Engineer Analysis: Pipe Manufacturing Plant

Answers based on customer input (Head of Quality — Marek Nowak) via http://158.160.187.32/ and the plant's design document.

---

## 1. Problem Definition

### 1.1 What type of problem are we solving?

**Ranking + multi-label classification, with anomaly detection under the hood.**

The customer wants a priority-ordered list of coils per shift with suspected defect types. That maps cleanly to two ML tasks:

| Task | Type | What it solves |
|------|------|----------------|
| **Risk scoring** | Regression (probabilistic) → Ranking | Each coil gets a defect probability score. Coils are sorted by score to produce the inspection queue. |
| **Defect type tagging** | Multi-label classification (8 classes) | For each high-risk coil, predict which of the 8 defect families are likely present. A coil can have multiple defects. |
| **Underlying signal** | Anomaly detection | The core pattern: a coil is "suspicious" when its process parameters deviate from normal trajectories. The ranking score is essentially an anomaly score. |

Why not pure classification? Because a binary "defective / not defective" label doesn't solve the operator's queue problem — they need to know which coils to check first within their ~60-coil capacity. A ranking/regression approach lets us threshold the score to match capacity dynamically.

### 1.2 Describe the output of the model(s)

Five outputs per coil, delivered within the shift window:

| Output | Format | Consumer |
|--------|--------|----------|
| **Priority level** | Enum: high / medium / low | Operator — decides which coils to focus on |
| **Inspection score** | Float 0–1 | System — sorts the queue, adjustable thresholds |
| **Suspected defect types** | Multi-hot vector over 8 classes (+ confidence per class) | Operator — knows what to look for (crack vs. lamination check is different) |
| **Rationale** | 2–3 plain-language strings (e.g., "Cooling deviation Stage 3 + Edge thickness spike") | Operator & Supervisor — trust and actionability |
| **Recommended action** | Enum + optional params (e.g., `VISUAL_CHECK + LEFT_SIDE`, `ULTRASONIC_TEST`) | Operator — standard operating procedure |

Key constraint from the customer: **false alarms are tolerable, misses are expensive.** This means the probability thresholds should be set defensively — better to flag 70 coils and let the operator deprioritize than to flag 40 and miss bad coils. We tune thresholds against the ~60-coil capacity budget.

### 1.3 Explain how the model output will be used in the decision-making process

**Advisory only. No automatic holds. No line stoppages.**

The flow per coil:
```
Coil exits line → Model scores within minutes → Appears in operator's ranked queue →
Operator inspects top-N coils visually (+ NDT if recommended) →
Decision: pass / hold-for-lab / remelt → Recorded in quality system → Coil moves on
```

If the system goes down, operators fall back to the current manual process. This is critical for plant adoption — the model must earn trust before anyone depends on it.

### 1.4 Should the model be trained only on missed defects?

**No. Train on all coils — defective and non-defective, caught and missed.**

The customer's reasoning is sound and I agree with all five points. Adding my technical perspective:

| Concern if training only on misses | Consequence |
|------------------------------------|-------------|
| Tiny dataset | ~19 missed defects/day out of 1,200 coils. That's ~1.6% positive rate on misses alone. The model would barely have signal. |
| Label bias encoded as ground truth | "Missed" means "the operator didn't catch it," not "impossible to detect from process data." The model would learn operator behavior, not process physics. |
| No calibration on caught defects | Without caught defects in training, the model can't learn what "detectable" looks like vs. "undetectable." You lose the ability to rank. |

The full dataset should include: caught defective, missed defective, and non-defective coils. Missed-vs-caught becomes a post-hoc analysis dimension, not a training filter.

---

## 2. Data and Features

### 2.1 What data sources are available and will be used?

Four sources, merged on coil ID:

| Source | Granularity | Joins on |
|--------|-------------|----------|
| Process sensor time-series | Per-stage, multi-sensor, ~Hz | Coil ID + timestamps → aggregated to coil-level features |
| Coil tracking/events | One row per coil | Coil ID (primary key) |
| Quality history | One row per coil (sparse — only defective coils have full entries) | Coil ID |
| Shift/context | One row per shift | Timestamp range |

The merge point is coil ID. Time-series sensor data gets aggregated (see 2.4). Quality labels are the target variable. Shift context is a feature, not a target.

**Data volume estimate:** ~1,200 coils/day × 365 days = ~438,000 coils/year. With ~4% defect rate, that's ~17,500 defective coils/year. That's a workable dataset for tree-based models.

### 2.2 Can the target value of the previous coil be used to predict the current coil?

**Yes. Essential, not optional.**

The customer confirmed that process issues span consecutive coils — roll wear, cooling drift, chemistry setup errors. From a modeling standpoint, this means:

- The data is **not i.i.d.** — rows are temporally correlated. This affects train/test splitting (see Section 4).
- We need **lag features**: the defect status, process parameters, and anomaly flags from the previous 1–5 coils.
- The target itself (previous coil's defect outcome) is a valid feature. If the last 3 coils were defective, the current one is more likely to be. But we must not leak future labels during inference — only use previously confirmed outcomes available at decision time.

Implementation: rolling window of N previous coils (N=3–5), with each previous coil represented by a compressed feature vector (score, top deviations, defect flags).

### 2.3 What feature groups will be used in the model?

Nine groups, agreed with customer input, organized by source:

| # | Group | Source | Example features |
|---|-------|--------|-----------------|
| 1 | Setpoints | Tracking system | Target thickness, width, rolling speed setpoint, cooling rate target |
| 2 | Per-stage aggregates | Sensor time-series | Mean/max/min/std temp, force, speed per stage |
| 3 | Deviations from targets | Derived (actual − setpoint) | Thickness error (mean, peak), temp overshoot duration |
| 4 | Stability metrics | Sensor time-series | Variance, range, spike count within coil production |
| 5 | Timing/context | Tracking system | Stage duration, time since shift start, time since last changeover |
| 6 | Equipment signals | Sensor time-series | Vibration RMS, load factor, alarm count, maintenance counter |
| 7 | Recent history (cross-coil) | Derived (lag features) | Previous coil defect status, rolling avg of last 5 coils' parameters, defect streak flag |
| 8 | Quality/inspection context | Quality system | Inspecting operator ID, shift, side viewed, recent defect-type frequency |
| 9 | Lab results (sparse) | Quality system | Chemistry readings, UT results — only available for tested coils |

### 2.4 Feature engineering — approach and derived features

The raw data is time-series. We aggregate to one row per coil. Categories of derived features:

**Statistical aggregates (per stage, per sensor):**
- Mean, median, std, min, max, 5th/95th percentiles
- Inter-quartile range, skew, kurtosis (for distribution shape anomalies)

**Out-of-spec metrics:**
- Seconds spent outside [lower_limit, upper_limit]
- Count of excursions (crossings of the limit boundary)
- Max excursion magnitude
- Cumulative area under the excursion curve (integrated deviation)

**Drift/trend features:**
- Slope of linear fit over the coil's production duration (per sensor)
- Start-to-end delta (first 10% avg vs. last 10% avg)
- Number of direction changes (sign flips in first derivative)

**Cross-coil sequence features:**
- Delta vs. previous coil for each aggregate (captures step changes)
- Rolling mean/std of the last 3/5/10 coils
- Flags: "any defect in last 3 coils?", "same defect type in last 5 coils?"
- Cumulative deviation from running average (CUSUM-lite)

**Signal quality:**
- % missing values per sensor per stage
- Sensor reset count (value drops to zero/near-zero and recovers)
- Flatline detection (variance ≈ 0 for extended period = stuck sensor)

**Event counters:**
- Number of manual overrides/adjustments during coil production
- Alarm events per stage
- Changeover flag (product grade or dimension change between coils)

---

## 3. Model Selection

### 3.1 Justify the choice of model class

**Primary recommendation: Gradient-boosted trees (LightGBM or CatBoost).**

| Factor | Why GBDT fits |
|--------|---------------|
| **Data structure** | Tabular, mixed dtypes, ~100–300 features. GBDTs are state-of-the-art for this shape. |
| **Missing values** | Sensors fail. CatBoost and LightGBM handle NaN natively — no imputation pipeline required. |
| **Class imbalance** | ~4% positive rate. GBDTs handle this with `scale_pos_weight` or custom loss. |
| **Interpretability** | SHAP values give per-prediction feature contributions → maps directly to the "2–3 plain-language reasons" requirement. Feature importance is comprehensible to quality engineers. |
| **Latency** | Sub-millisecond inference per coil. Runs on CPU. No GPU needed. |
| **Deployment** | Serialize to a single file (~few MB). Load with `joblib` or native save. No serving infrastructure. |
| **Retraining** | Train on historical data monthly. Fast (minutes on ~400k rows). No hyperparameter drift. |

**What about deep learning?**

Not needed here. The data is tabular after aggregation, not images or raw waveforms. A transformer or LSTM on raw time-series would be overkill — more complex to deploy, harder to interpret, and unlikely to outperform GBDT on this data size. If the plant later adds camera data (surface images), we revisit this.

**What about simple rules?**

The baseline document already plans to keep hard-limit rules for obvious out-of-spec conditions. The ML model augments this — it catches the subtle multi-parameter interactions that rules miss.

### 3.2 Model architecture

Two-model setup:

```
┌─────────────────────┐     ┌──────────────────────────┐
│ Model A: Risk Scorer │ ──► │ Priority score (0–1)       │
│ (Binary classifier)  │     │ → Ranked inspection queue  │
└─────────────────────┘     └──────────────────────────┘

┌─────────────────────┐     ┌──────────────────────────┐
│ Model B: Defect      │ ──► │ Multi-hot vector over      │
│ Type Tagger          │     │ 8 defect classes           │
│ (Multi-label)        │     │ → What to look for         │
└─────────────────────┘     └──────────────────────────┘
```

- **Model A:** LightGBM binary classifier (defect: yes/no). Output = probability. This probability IS the ranking score. Calibrated via isotonic regression so thresholds are interpretable ("score > 0.7 = high priority").
- **Model B:** One-vs-rest LightGBM for each of 8 defect classes, OR a single multi-output classifier. Output = vector of 8 probabilities. Only computed for coils flagged by Model A above a configurable threshold.

Rationale for two models: Model A runs on all 1,200 coils/day. Model B only runs on the ~60–100 flagged coils. Saves compute and keeps the pipeline simple. Also, the defect type labels are sparse and imbalanced per class — separate training strategies may be needed.

**Explainability layer:**
- SHAP on Model A → top contributing features → mapped to plain-language templates ("Temperature at Stage 2 was 15% above setpoint throughout the coil").
- If no single feature dominates → fallback rationale: "Multiple small deviations across stages."

---

## 4. Validation and Data Splitting Strategy

### 4.1 How will the train / validation / test split be constructed?

**Time-based, sequence-aware, no future leakage.**

```
┌────────────┬──────────────┬─────────────┬──────────────┐
│ Train      │ Validation   │ Test        │ Future       │
│ Jan–Dec    │ Jan–Mar      │ Apr–Jun     │ (production) │
│ 2024       │ 2025         │ 2025        │              │
└────────────┴──────────────┴─────────────┴──────────────┘
```

Rules:
1. **Time ordering:** All train data < all validation data < all test data. No random shuffling across time.
2. **Sequence integrity:** Consecutive coils from the same production run stay in the same split. Detect run boundaries via changeover events or time gaps > threshold. This prevents the "sister coil leakage" problem the customer flagged.
3. **Decision-time masking:** During evaluation, only features that would have been available at prediction time are used. Specifically:
   - Previous coil's defect outcome: only use IF the outcome was confirmed BEFORE the current coil's prediction time.
   - Lab results: only use IF the test was completed and result entered BEFORE prediction time.
   - This is enforced by a timestamp-aware feature pipeline, not just a fixed split.
4. **Holdout is sacred:** Test set is used exactly once — for the final evaluation before the pilot. Validation set is used for threshold tuning and model selection.

### 4.2 Which ML metrics will be used for offline model evaluation?

**Three tiers: ranking metrics (primary), classification metrics (diagnostic), business metrics (decision).**

**Tier 1 — Ranking (does the queue work?):**

| Metric | Why |
|--------|-----|
| **Recall@K** where K = 60 (operator capacity) | "If the operator checks the top 60, what % of defective coils are caught?" This IS the business question. |
| **Precision@K** where K = 60 | "What % of those 60 flagged coils actually have defects?" Controls operator trust. Too low = "the system cries wolf." |
| **Average Precision (AP)** | Summary across all K values. Standard ranking metric for imbalanced retrieval. |
| **NDCG@60** | Rewards ranking truly defective coils higher in the list. Better than AP when we care about ordering within the queue. |

**Tier 2 — Classification (diagnostic):**

| Metric | Why |
|--------|-----|
| **Recall by defect type** | Are we equally good at catching cracks vs. surface defects? Some types are inherently harder. This identifies blind spots. |
| **Precision by defect type** | Do we over-flag certain types? |
| **F2-score** (β=2, recall-weighted) | Single number for threshold comparison. β=2 because recall is ~2× more important than precision per the customer's risk tolerance. |
| **PR-AUC** | Better than ROC-AUC for 4% positive rate. ROC would look deceptively good. |
| **Confusion matrix (8×8)** | Do we confuse laminations with cracks? This matters for operator guidance. |

**Tier 3 — Business (does it work in practice?):**

| Metric | Baseline | Target |
|--------|----------|--------|
| Miss rate (% of defective coils not flagged) | ~40% | ≤20% |
| High-priority alerts per shift | N/A (manual) | ≤60 (operator capacity) |
| Alerts per defect caught | N/A | ≤10 (max 10 false flags per real catch) |
| Dwell time increase | 0% | ≤10% |
| Miss rate stability across shifts/lines | N/A | Within ±5% across groups |

**What we do NOT optimize for:**
- Accuracy — meaningless at 96% majority class.
- ROC-AUC — inflated by the 96% negatives.
- Pure precision — we'd rather catch 100% of defects with 30% precision than catch 60% with 90% precision.

**Validation cadence:**
- Offline: train on 2024, validate on Q1 2025, test on Q2 2025.
- Online pilot: run in shadow mode (guidance displayed but not acted on) for 2 weeks → compare model flags vs. operator findings → then go live as advisory.
