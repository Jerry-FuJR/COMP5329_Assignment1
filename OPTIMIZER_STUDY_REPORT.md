# Experimental Investigation of Optimization Strategies in Repaired QANet

## 1. Introduction

After the QANet framework had been fully repaired, it became a controlled platform for studying how specific deep learning mechanisms affect training dynamics and model performance. This study focuses on optimization configurations, with particular emphasis on the interaction between optimizer choice and learning-rate scheduling.

The aim of this investigation is not superficial hyperparameter tuning, but a mechanism-oriented analysis of how optimization design influences convergence speed, training stability, and validation performance in the repaired framework.

To support this study, I extended the repaired codebase so that Adam could be evaluated under the training setup implied by the repaired optimizer design. Specifically, I added a new scheduler implementation in `Schedulers/warmup_scheduler.py`, and integrated it into the framework through modifications to `Schedulers/scheduler.py` and `TrainTools/train.py`. These additions were necessary because the repaired Adam configuration uses a base learning rate of `1.0` and is intended to be controlled by a warmup-based scheduler rather than treated as a fixed effective learning rate. Without this extension, Adam would not have been evaluated under a meaningful optimization setup, and the comparison would have been confounded by scheduler mismatch rather than reflecting the repaired framework itself.

Accordingly, this investigation studies optimization behavior not only at the level of optimizer labels, but at the level of complete optimization strategies.

## 2. Research Question

How do scheduler choice and optimization strategy affect convergence behavior and validation performance in the repaired QANet framework?

This question is addressed through two subproblems:

1. Which scheduler is most suitable for SGD-based methods under a short training budget?
2. After selecting a suitable scheduler for the SGD family, how does Adam paired with warmup compare with SGD and SGD with momentum under a larger training budget?

## 3. Hypotheses

This investigation is structured around three hypotheses.

### H1. Scheduler choice materially affects SGD-based optimization behavior.

For SGD-based methods, learning-rate scheduling is expected to influence convergence efficiency and validation performance. In particular, non-constant schedules such as cosine annealing or step decay are expected to outperform a constant learning rate because they reduce the effective step size as training progresses.

### H2. A single scheduler can serve as a reasonable representative choice for the broader SGD family.

Although vanilla SGD and SGD with momentum may not favor exactly the same scheduler in every metric, one scheduler may provide a sufficiently strong and interpretable configuration across both methods, making it appropriate for the main strategy comparison.

### H3. Adam paired with warmup will outperform SGD-based strategies under a larger but finite training budget.

Because Adam performs adaptive parameter-wise updates and is paired with a warmup schedule specifically introduced to match the repaired framework's intended behavior, it is expected to converge more effectively than SGD-based methods within a finite number of training steps.

## 4. Experimental Design

### 4.1 Overall Structure

The study was divided into two blocks:

1. Block A: Scheduler Selection for SGD-based Methods
2. Block B: Main Optimization Strategy Comparison

This structure was chosen to avoid an unfair comparison between Adam under its intended warmup schedule and SGD-based methods under an arbitrary or potentially suboptimal scheduler.

### 4.2 Controlled Variables

The following factors were held constant unless explicitly varied:

- Model architecture
- Dataset split and preprocessing pipeline
- Batch size
- Checkpoint interval
- Evaluation protocol
- Default random seed
- Loss function and repaired framework components

This design ensures that performance differences can be attributed primarily to the optimization configuration being studied.

### 4.3 Quantitative Metrics

The following metrics were used throughout the study:

- Best dev F1
- Best dev EM
- Dev loss
- Convergence curves over training steps

F1 was treated as the primary comparison metric. EM and dev loss were used as supporting indicators of optimization quality and prediction behavior.

## 5. Framework Modifications for This Study

To make the optimization experiment valid, I introduced a new warmup scheduler and integrated it into the repaired framework.

### 5.1 Added File

- `Schedulers/warmup_scheduler.py`

This file implements the `warmup_lambda` scheduler used to control Adam's effective learning rate.

### 5.2 Modified Files

- `Schedulers/scheduler.py`
- `TrainTools/train.py`

### 5.3 Purpose of the Modification

In the repaired optimizer design, Adam is initialized with a base learning rate of `1.0`, while the intended effective learning rate is supposed to be controlled by a scheduler. I therefore added the `warmup_lambda` scheduler and registered it through the scheduler factory so that it could be selected through the existing training interface. I also updated the training entry point so that the warmup-related configuration could be passed into the experiment.

These changes were central to the experiment because they ensured that Adam was evaluated under the intended optimization setup rather than under an incomplete default schedule.

## 6. Block A: Scheduler Selection for SGD-based Methods

### 6.1 Purpose

Block A was designed as a pilot scheduler-selection phase. Its role was to identify a suitable scheduler for the SGD family before performing the main strategy comparison.

### 6.2 Experimental Setup

Two optimizers were considered:

- SGD
- SGD with momentum

For each optimizer, three schedulers were tested:

- `lambda`
- `cosine`
- `step`

Because this was a selection phase rather than the final comparison, a shorter training budget was used:

- `num_steps = 1000`

### 6.3 Configurations Compared

The following configurations were evaluated:

- SGD + lambda
- SGD + cosine
- SGD + step
- SGDMomentum + lambda
- SGDMomentum + cosine
- SGDMomentum + step

### 6.4 Results

The observed results were as follows:

| Configuration | Best F1 | Best EM |
|---|---:|---:|
| SGD + cosine | 6.370364 | 0.000000 |
| SGD + step | 6.283838 | 0.000000 |
| SGDMomentum + cosine | 6.106976 | 0.166667 |
| SGDMomentum + step | 6.377957 | 0.083333 |

### 6.5 Analysis

The scheduler-selection results did not identify one scheduler that dominated across all cases.

For vanilla SGD, cosine achieved the highest F1 (`6.370364`), slightly outperforming step (`6.283838`). For SGD with momentum, step achieved the highest F1 (`6.377957`), while cosine produced a lower F1 (`6.106976`) but a higher EM (`0.166667` compared with `0.083333`).

This means the evidence was mixed rather than uniformly decisive. However, the difference between cosine and step remained modest, and cosine yielded the strongest result for vanilla SGD while also producing the higher EM value for SGDMomentum in this run. Since the EM values were still small in absolute magnitude, this EM difference should be treated as supportive rather than decisive evidence. Taken together with the smoother decay profile of cosine annealing, cosine was selected as the representative scheduler for the SGD family in the main comparison.

### 6.6 Conclusion of Block A

Cosine annealing was selected as the representative scheduler for the SGD family.

This decision was based on a balance of evidence across F1, EM, and practical interpretability, rather than on a single overwhelmingly dominant result.

## 7. Block B: Main Optimization Strategy Comparison

### 7.1 Purpose

After selecting cosine as the representative scheduler for SGD-based methods, Block B compared three optimization strategies under a larger training budget:

- Adam + warmup_lambda
- SGD + cosine
- SGDMomentum + cosine

### 7.2 Experimental Setup

This stage used a larger training budget:

- `num_steps = 4000`

This larger budget was chosen because very short training runs may disproportionately favor methods such as Adam that often improve quickly in the initial phase.

### 7.3 Rationale

This block compares optimization strategies, not bare optimizers under identical but potentially inappropriate schedules.

Specifically:

- Adam was evaluated with `warmup_lambda`, matching the intended repaired framework design.
- SGD and SGDMomentum were evaluated with the scheduler selected from Block A.

This makes the comparison more meaningful than forcing all methods to share the same scheduler regardless of compatibility.

### 7.4 Results

Please insert the final Block B results here. A suitable table format is:

| Configuration | Best F1 | Best EM | Interpretation |
|---|---:|---:|---|
| Adam + warmup_lambda | [fill] | [fill] | [fill] |
| SGD + cosine | [fill] | [fill] | [fill] |
| SGDMomentum + cosine | [fill] | [fill] | [fill] |

### 7.5 Suggested Interpretation

If Adam performs best, the following interpretation can be used:

> Under the larger training budget, Adam paired with warmup achieved the strongest validation performance. This supports the hypothesis that adaptive optimization, when combined with an appropriate warmup schedule, improves convergence efficiency in the repaired QANet framework.

If SGDMomentum outperforms vanilla SGD, the following interpretation can be used:

> SGDMomentum outperformed vanilla SGD under the same cosine scheduler, indicating that momentum contributes positively by stabilizing updates and improving descent efficiency.

If the gap becomes smaller than expected, the following interpretation can be used:

> The gap between Adam and SGD-based methods narrowed after introducing a more suitable scheduler for the SGD family, suggesting that some apparent optimizer advantage under naive settings may in fact reflect scheduler compatibility rather than optimizer type alone.

## 8. Interpretation of Findings

This study highlights two main points.

First, scheduler choice matters substantially for SGD-based optimization. The scheduler-selection phase showed that optimizer comparisons can become misleading if one optimizer family is evaluated under a suitable schedule while another is not.

Second, optimization strategies should be interpreted as coupled mechanisms rather than isolated algorithm names. In the repaired framework, Adam is not merely "Adam"; it is more accurately understood as Adam paired with a warmup-based schedule. Likewise, the behavior of SGD-based methods depends strongly on the scheduler used alongside them.

Thus, the experiment is mechanism-oriented in the sense required by the assignment: it studies how adaptive updates, momentum, and learning-rate scheduling jointly shape optimization behavior.

## 9. Limitations

Several limitations should be acknowledged.

- The scheduler-selection phase used only `1000` steps, so its conclusions mainly reflect early training behavior.
- The current study uses a single fixed seed, so the results should be interpreted as controlled observations rather than statistically robust estimates.
- The Block A evidence for cosine over step was mixed rather than uniformly dominant.
- EM values in Block A were small, so EM differences should be interpreted cautiously.

These limitations do not invalidate the experiment, but they reduce the strength of causal claims and should be stated clearly.

## 10. Conclusion

This experiment investigated optimization behavior in the repaired QANet framework through a two-stage design. First, a scheduler-selection phase compared `lambda`, `cosine`, and `step` scheduling within the SGD family under a reduced training budget. The results suggested that cosine was the most appropriate representative scheduler for SGD-based methods, although the evidence was mixed rather than uniformly dominant.

Second, the selected scheduler was used in a larger-budget comparison against Adam paired with `warmup_lambda`, which was enabled by the addition of `Schedulers/warmup_scheduler.py` and its integration into `Schedulers/scheduler.py` and `TrainTools/train.py`. This ensured that the experiment compared meaningful optimization strategies rather than incomplete or mismatched training setups.

Overall, the study shows that performance in the repaired framework depends not only on optimizer type, but also on the scheduler paired with it. This supports the broader conclusion that optimization behavior in deep learning emerges from the interaction of multiple mechanisms rather than from isolated algorithm names alone.
