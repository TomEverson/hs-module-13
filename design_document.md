# ML System Design Document: Pipe Manufacturing Defect Detection at Coil Stage

## 1. Business Problem Description

### Manufacturing Process
The pipe manufacturing process follows these steps:
1. A steel slab is heated.
2. It is hot-rolled through a series of rollers, producing a long, thin steel sheet.
3. The sheet is wound into a coil.
4. Coils are uncoiled and formed/welded into pipes.
5. Pipes undergo final testing (e.g., hydrostatic testing).

### The Critical Nature of the Coil Stage
The coil stage is the **last point in the process where defects are still relatively inexpensive to address**. If a defect is detected at the coil stage, the material can be remelted and reused. However, if a defective coil passes inspection and proceeds to pipe production, it may result in hundreds of meters of defective pipes that cannot be recovered and must be scrapped — leading to significant financial losses.

### How Defects Occur
Defects originate from:
- Casting and chemistry issues
- Rolling setup parameters and forces
- Cooling processes
- Handling and storage conditions

**Defect types (8 families):**
- Chemical composition out of specification
- External rust/corrosion
- Internal corrosion, scale, and inclusions
- Cracks
- Laminations/delaminations
- Thickness and profile deviations
- Width, edge, and waviness issues
- Surface defects (scratches, dents, roll marks)

### Consequences of Missed Defects
When a defective coil is not detected at the coil stage:
- It proceeds to pipe forming, where the defect typically manifests during forming or hydrostatic testing.
- The entire batch produced from that coil must be scrapped.
- Additional costs include lost production time, schedule disruption, and extra material handling.
- Defect confirmation can take hours to up to 7 days after rolling.

### Current Losses
| Scenario | Cost per coil |
|----------|--------------|
| Caught at coil stage (remelting) | ~€4,000 |
| Missed — found after pipe production | ~€20,000 |
| **Loss difference per missed defective coil** | **~€16,000** |

- ~4% of all coils have defects.
- Currently, ~40% of defective coils are **missed** at the coil stage.
- Production volume: ~1,200 coils/day, ~30 days/month (~36,000 coils/month).

## 2. AS-IS Analysis: Current Inspection Process

### How Inspection Works Today
- **Primary method:** Visual inspection by a single quality control operator per shift.
- **Capacity limitation:** A careful inspection is feasible for only ~60 coils per shift, but all coils must be checked as they flow through.
- **Detection limitations:**
  - Detection rate depends on defect type and which side of the coil is visible.
  - Some defects (internal cracks, laminations, certain chemical issues) are invisible to the naked eye and only show up during pipe forming or hydrostatic testing.
- **Additional testing methods** (ultrasonic, magnetic particle, penetrant, chemical analysis) are used only **selectively** due to lab capacity, cost, and turnaround time constraints.

### Current Reliability
- ~4% of coils have defects.
- ~40% of those defective coils (~1.6% of total) are missed at the coil stage.
- Prioritization is entirely manual, based on operator experience.

### Personnel
- On-shift **Quality Control Operator** performs visual inspection.
- **Quality Shift Supervisor (Anna Kowalczyk)** organizes inspection assignments within the shift window.

## 3. Stakeholders and Customers

| Role | Person | Interest |
|------|--------|----------|
| **Primary Customer/Owner** | Marek Nowak — Head of Quality | Accountable for missed defects and write-offs |
| **Users** | Quality Control Operators & Anna Kowalczyk (Quality Shift Supervisor) | Would use the system's guidance and prioritization |
| **Production** | Tomasz Zieliński — Head of Production | Benefits from fewer disruptions; insists on no added line stoppages |
| **IT/Infrastructure** | Jakub Mazur | Responsible for integrations and 24/7 reliability; needs a supportable system |

### Who Benefits
- **Quality team and the plant overall:** Fewer missed defects and write-offs.
- **Production:** More stable flow, fewer scrap-related interruptions.

## 4. Justification for Using Machine Learning

The problem is **too complex for fixed rules** to work well — simple thresholds would either miss too many defects or flag too many good coils.

### Why ML is Superior to Rule-Based Approaches

1. **Multiple interacting signals:** Defects arise from complex combinations of temperature, speed, force, cooling, and equipment behavior. Single-threshold rules miss "bad combinations" where individual values appear normal.

2. **Diverse defect types:** Eight defect families have different signatures. Hand-crafting and maintaining separate rules for each type becomes unmanageable.

3. **Context-dependent thresholds:** Product grades, equipment setups, and seasons shift the normal operating range. Static limits either create constant false alarms or remain silent when conditions drift.

4. **Temporal patterns:** Defects often depend on sequences across stages ("if A happens, then B occurs later"), not single-point readings. Simple checks cannot capture these patterns.

5. **Noisy, incomplete data:** Sensors have gaps and noise. Hard rules either trigger excessively or get tuned so loosely they miss real issues.

6. **Limited inspection capacity:** The system must produce a ranked priority list so operators can focus careful inspection and NDT (non-destructive testing) on the highest-risk coils. Rule sets tend to produce many equal-priority flags with no clear ranking.

7. **Delayed ground truth:** Some defects are confirmed only days later. ML models can learn from historical links between process data and later outcomes; static rules cannot benefit from this feedback.

8. **Lower maintenance burden:** Rules need constant retuning as equipment wears and product mix changes. A data-driven model can be retrained/refreshed based on new historical data rather than requiring manual reprogramming of dozens of limits.

### Hybrid Approach
Simple hard limits will still be retained for obvious out-of-spec conditions. The ML layer adds value through better catch rate and a workable priority list — without slowing the line.

## 5. Baseline Solution

### Status Quo (Current Baseline)
- Operator visually checks all coils at speed.
- Only ~60 coils per shift receive a truly careful inspection.
- Extra NDT/lab tests are used selectively based on operator judgment.
- **~40% of defective coils are missed at the coil stage.**

### Simple Non-ML Baseline
- Basic rule/threshold checks using known process limits to flag obvious out-of-spec coils.
- A manual checklist for targeted re-inspection of flagged coils.
- Advisory only, no automatic holds.
- Minimal integration effort, but limited detection capability.

### Proposed ML-Based Solution
- **Per-coil risk ranking** generated within the shift window.
- **Likely defect category** indicated for each high-risk coil.
- Operators are guided to focus careful inspection and limited NDT resources on the highest-risk coils.
- Advisory-only in the first phase — no automatic line stoppages.
- Must not increase average coil hand-off time.

## 6. Expected Business Impact

### Success Criteria
| Criterion | Target |
|-----------|--------|
| **Primary:** Missed defective coil rate | Reduce by ≥50% within ~1 year |
| **Flow impact:** Dwell time from line exit to next stage | No more than 10% increase |
| **Usage model:** Guidance only | No automatic holds; operators receive clear per-shift priority list with likely defect type |
| **Reliability:** System availability | 24/7 without disrupting production |

### Business Metrics to Track
1. Missed-defect rate at the coil stage (missed / total defective)
2. Coil-stage detection rate by defect type
3. False positive rate: coils sent for remelting that would have been acceptable
4. Downstream scrap: number of coils causing pipe scrap and associated cost (€)
5. Remelting volume and cost (€)
6. Average dwell time from line exit to next stage; % of coils exceeding current SLA
7. Operator re-inspection workload; NDT and lab resource usage
8. Overall monthly quality-related losses (€)

### Potential Value

**Unit economics:**
- Catching a defective coil at coil stage → ~€4,000 loss (remelting)
- Missing it until after pipe production → ~€20,000 loss (scrap)
- **Savings per avoided miss = ~€16,000**

**Current state:**
- ~4% defect rate = ~48 defective coils/day (out of 1,200 total)
- ~40% miss rate = ~19.2 missed defective coils/day

**Target state (50% reduction in misses):**
- Miss rate reduced to ~20% = ~9.6 missed defective coils/day
- **~9.6 additional defective coils caught per day at coil stage**
- **Daily savings ≈ 9.6 × €16,000 = ~€153,600/day**
- **Monthly savings ≈ ~€4.6 million**
- **Annual savings ≈ ~€55 million**

Additionally, softer benefits include:
- Fewer production schedule disruptions
- Reduced handling and logistics overhead for scrap
- More predictable quality outcomes
- Better resource allocation for NDT/lab testing
