# Experimental Protocol

This document specifies the two reported experiments:

* **Experiment 1**. Measuring the effect of the actor-network size on the task of learning to race in one fixed circuit. 
* **Experiment 2**. Measuring the ability of PPO to generalize to multiple circuits, under different observations (Frenet vs LiDAR).

Experiment 1 combines the 45 completed original runs with 15 new tiny-actor
runs. Experiment 2 is a new, separately recorded rerun; its historical result
directories are retained but are not pooled with this study.

The exact policy, model, target and loss definitions for the training are reported in [`LEARNING.md`](LEARNING.md). 

For all comparisons, all algorithms are given a **fixed budget of interactions**, instead of episodes or number of updates.
This stresses how much each algorithm is capable of making treasure of each transition observed with the environment.

## Preliminary Steps

### 1. Reproducible Randomness Setup

One integer written on a run specification is called its **root identity**. 
It must control the whole run reproducibly, but using one mutable random generator for everything would create accidental coupling. 
For example, adding an evaluation could consume random numbers and change every later training action.

The solution is to derive independent child generators for distinct jobs:
1. actor initialization;
2. critic initialization;
3. stochastic policy actions;
4. environment reset;
5. training-circuit schedule;
6. PPO minibatch order; and
7. evaluation or baseline-policy sampling where sampling exists; and
8. track generation.

This hierarchy is implemented through `numpy.random.SeedSequence`.
Using `SeedSequence`, a single stream of random numbers is identified by the quadruple:
$$ [\mathtt{PROTOCOL\,KEY}, \ \mathtt{Namespace\,Code}, \ \mathtt{Local\,Identity}, \ \mathtt{Stream}] $$
Where:
* $\mathtt{PROTOCOL\,KEY}$ is always set to `0`.
* $\mathtt{Namespace\,Code}$ describes the experiment that is currently being run. 
  It assumes values from $1$ to $12$ according to the following table:
  | Purpose | Namespace code |
  |---|---:|
  | Experiment 1 reported roots | 1 | 
  | Experiment 2 reported roots | 2 | 
  | Learning-rate configuration | 3 | 
  | Capability check | 4 | 
  | Reduced-budget end-to-end validation | 5 |
  | Controlled-problem algorithm validation | 6 | 
  | Experiment 1 circuit candidates | 7 | 
  | Multi-circuit development checks | 8 | 
  | Experiment 2 training circuits | 9 |
  | Experiment 2 validation circuits | 10 | 
  | Experiment 2 test circuits | 11 | 
  | Randomized execution order | 12 |
* $\mathtt{Local\,Identity}$ is the **local identity** of a specific run. 
  It starts from $0$ and grows one by one for each repeated run.
* $\mathtt{Stream}$ represents one of the $8$ child generators, that distinguish the specific stream of random numbers being used. 
  Its values go from $1$ to $8$. 

Every recorded run stores both human-readable logical identities and generated integer states.
Checkpoints retain all mutable generator states required for an exact resume on the supported hardware and software stack.

### 2. Learning Rate Search

To determine a proper **learning rate** for all of the algorithms, different values are tested for each of them, in a short run using:
* An actor network of size `(64, 64)`.
* A budget of $250\,000$ interactions, which is increased to $750\,000$ if no learning rate prevails.
* $3$ seed roots.

The tested values are the following:

- REINFORCE Actor rate:
  $$1\cdot10^{-4}, \quad 3\cdot10^{-4}, \quad 1\cdot10^{-3}$$

- A2C/PPO Actor and Critic rates:
  $$ 
  (1\cdot10^{-4}, 3\cdot10^{-4}), \quad (3\cdot10^{-4}, 1\cdot10^{-3}), \quad (3\cdot10^{-4}, 3\cdot10^{-3}), \\
  (3\cdot10^{-4}, 1\cdot10^{-2}), \quad (1\cdot10^{-3}, 1\cdot10^{-3}), \quad (1\cdot10^{-3}, 3\cdot10^{-3}), \\
  (1\cdot10^{-3}, 1\cdot10^{-2})
  $$
  
The candidates are engineering scales around Adam's $1\cdot10^{-3}$ suggested
default. 

#### Selection Criteria

Selection of the **LR** is done using $4$ criteria, applied one after the other:
1. **Laps**. Number of final deterministic evaluations that complete a full lap.
2. **Mean progress**. If tied, compare mean maximum normalized progress achieved in the final deterministc evaluations.
3. **Mean return**. If still tied, compare mean return achieved in the final deterministic evaluations.
4. **Scale**. If still tied, choose by smaller actor LR, then smaller critic LR.  

#### Recorded Outcomes

| Algorithm | Actor | Critic | Allowance | Laps | Mean progress | Mean return |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE | $1\cdot10^{-4}$ | — | 250k | 0/3 | 0.070 | −4.28 |
| REINFORCE | $3\cdot10^{-4}$ | — | 250k | 0/3 | 0.169 | 4.77 |
| **REINFORCE** | $\mathbf{1\cdot10^{-3}}$ | — | 250k | **2/3** | **0.894** | **152.67** |
| A2C | $1\cdot10^{-4}$ | $3\cdot10^{-4}$ | 750k | 0/3 | 0.265 | 9.54 |
| A2C | $3\cdot10^{-4}$ | $1\cdot10^{-3}$ | 750k | 0/3 | 0.274 | 14.10 |
| A2C | $3\cdot10^{-4}$ | $3\cdot10^{-3}$ | 750k | 0/3 | 0.520 | 32.77 |
| A2C | $3\cdot10^{-4}$ | $1\cdot10^{-2}$ | 750k | 0/3 | 0.582 | 37.23 |
| A2C | $1\cdot10^{-3}$ | $1\cdot10^{-3}$ | 750k | 2/3 | 0.835 | 150.51 |
| **A2C** | $\mathbf{10^{-3}}$ | $\mathbf{3\cdot10^{-3}}$ | 750k | **3/3** | **1.000** | **222.77** |
| A2C | $1\cdot10^{-3}$ | $1\cdot10^{-2}$ | 750k | 3/3 | 1.000 | 218.50 |
| PPO | $1\cdot10^{-4}$ | $3\cdot10^{-4}$ | 250k | 3/3 | 1.000 | 225.52 |
| PPO | $3\cdot10^{-4}$ | $1\cdot10^{-3}$ | 250k | 3/3 | 1.000 | 223.77 |
| PPO | $3\cdot10^{-4}$ | $3\cdot10^{-3}$ | 250k | 3/3 | 1.000 | 227.19 |
| **PPO** | $\mathbf{3\cdot10^{-4}}$ | $\mathbf{10^{-2}}$ | 250k | **3/3** | **1.000** | **231.00** |
| PPO | $1\cdot10^{-3}$ | $1\cdot10^{-3}$ | 250k | 3/3 | 1.000 | 207.57 |
| PPO | $1\cdot10^{-3}$ | $3\cdot10^{-3}$ | 250k | 3/3 | 1.000 | 225.83 |
| PPO | $1\cdot10^{-3}$ | $1\cdot10^{-2}$ | 250k | 2/3 | 0.686 | 148.50 |

**Selected learning rates** therefore are: 

| Algorithm | Actor | Critic |
|---|---:|---:|
| REINFORCE | $1\cdot10^{-3}$ | — |
| A2C | $1\cdot10^{-3}$ | $3\cdot10^{-3}$ |
| PPO | $3\cdot10^{-4}$ | $1\cdot10^{-2}$ |

### 3. Fixed Circuit Choice for Experiment 1

The fixed circuit was chosen by manually filtering interesting generated circuits among the first $20$ seeds.

The selected circuit was the one with `seed=0`.

### 4. Hardware Choice and Timing

Everything runs on the CPU with PyTorch `float32`. 
Neural work is not placed on a GPU, and no GPU path is offered.

The actor and critic have two hidden layers and are stepped in batches of one row per environment worker, which is far too small to amortize host-to-device transfer. 
Over 20,000 interactions with $8$ workers, these were the measured speeds:

| Algorithm | CPU | CUDA |
|---|---:|---:|
| REINFORCE | 3,372 interactions/s | 1,862 interactions/s |
| A2C | 3,694 interactions/s | 1,144 interactions/s |
| PPO | 2,835 interactions/s | 1,146 interactions/s |

All three algorithms use the same worker count ($8$, the number of physical CPU cores), so that reported collection throughput and wall time are comparable. 

Timing categories, measured during the runs, are selected so that they do not overlap:

- Environment collection time;
- Actor/critic optimization time;
- Deterministic evaluation time;
- Checkpoint and metric-persistence time;
- End-to-end time.

Reported runs execute one at a time; no other training process competes for the cores.


## Experiment 1 — Actor Size on a Single Circuit

### Research Question

> **RQ1**: How does actor-network capacity affect training an agent to race a car on one fixed circuit?

**RQ1** is answered by observing, in practice, four measures of training efficiency and efficacy:
* **Final driving performance**. How fast does the agent complete the lap (if they manage to complete it)?
* **Interaction efficiency**. How many interactions with the environment does the agent have to perform before reaching the task threshold?
* **Converge reliability**. How do different runs under the same settings vary in the results?
* **Computational cost**. How long does the training take for the agent to reach the task threshold?

The actor-size comparison is repeated for REINFORCE, A2C+GAE and PPO. 
The algorithm comparison is secondary: it describes the practical effect of adding a critic, GAE and bounded sample reuse.

### Design Matrix

$3$ algorithms and $4$ network sizes are compared, for a total of $3 \times 4=12$ training configurations. Each configuration has five paired root identities, `0` through `4`, for $60$ runs in the combined study.

The original small, medium and large cells account for 45 completed runs. The tiny `(8, 8)` actor adds 15 runs under the same fixed conditions and roots. The report treats the 60 runs as one Experiment 1 while retaining the run records that identify the historical and newly added cells.

| Algorithm \ Size | `(8, 8)` tiny | `(32, 32)` small | `(64, 64)` medium | `(256, 256)` large |
|---|---:|---:|---:|---:|
| REINFORCE | 5 roots | 5 roots | 5 roots | 5 roots |
| A2C+GAE | 5 roots | 5 roots | 5 roots | 5 roots |
| PPO | 5 roots | 5 roots | 5 roots | 5 roots |

### Fixed Conditions

Every run uses:

* The saved `tracks/experiment_1.json` circuit, with its canonical start for the evaluation runs;
* Frenet observations $(d_t,\phi_{e,t},v_t,\delta_t,\bar\kappa_t)$;
* A bounded Gaussian policy, trained with Adam optimizer;
* In the cases of A2C and PPO, one fixed `(64, 64)` critic network;
* The learning rate selected before the experiment for its algorithm;
* A budget of $2\,000\,000$ training interactions;
* A deterministic evaluation over the `experiment_1` circuit every $50\,000$ training interactions;
* A checkpoint saves every $250\,000$ interactions and at the end.

The original timeout treatment is retained for every Experiment 1 run, including the tiny extension. REINFORCE stops its Monte Carlo return at the 40-second task deadline; A2C and PPO bootstrap once from the critic at that deadline. This mismatch is a limitation of comparisons between the algorithms, not a difference between actor sizes.

### Definition of Convergence

Threshold attainment is the first of three consecutive evaluations that all
complete a full lap in at most $34$ seconds. With evaluations 50k interactions
apart, the streak spans 100k interactions, and confirmation occurs 100k after
its first checkpoint. Later regression remains possible.

The $34$ seconds threshold is found by executing a deterministic, reference controller policy:
* On average, the determinist controller takes $22.3s$;
* The threshold is set to $1.5\times$ the reference average. 

The $1.5$ multiplier allows the agent to have a slight margin of error without allowing excessively slow laps.

Reaching convergence does not stop the training, which in any case concludes all $2\,000\,000$ transitions.
Convergence is therefore only considered as a post-hoc measure of training quality.

### Recorded Data

The following are the actual tracked quantities for Experiment 1:

* **Identity and reproducibility settings**. This includes the algorithm, the actor and critic sizes, the used environment, model and optimizer, the evaluation configuration, the python venv dependencies and hardware information. 

* **Episodes outcomes**. For each training and evaluation episode, we record each return, agent step and progress, explicit outcome of the episode, car diagnostics (speed, throttle/brake, ...). 

* **Optimizer updates**. For each update, actor and critic losses, learning rates, gradient norm, weight norm, gradient-estimator dispersion, explained variance, importance-ratio distribution, etc.

* **Computational costs**. Physical time spent in each phase of the training, interactions per second, memory usage, etc.

### Analysis conventions

Experiment 1 uses exhaustive root bootstrap summaries over its five roots. The
combined analysis includes all 60 runs, while the original PPO architecture
selection remains restricted to its original small, medium and large candidates.

Experiment 2 aggregates circuits within each root before making its primary
Frenet-minus-LiDAR comparison. Its root bootstrap uses 10,000 resamples, seed
`0`, and a percentile 95% interval. Stability supplements the unchanged final
endpoint with checkpoints in the prespecified late window $0.8B < t \leq B$,
where $B$ is the run's interaction budget.

## Experiment 2 — Observation Types over Generalized Circuits 

### Question and hypotheses

> **RQ2**: How well does a PPO agent generalize over unseen procedurally generated circuits? 
> **RQ3**: How do Frent observations compare with local LiDAR sensing?

### Design Matrix

$2$ observation types are compared.
The agent architecture is kept fixed, therefore the total number of training configuration is $2$.

Each training configuration is run $10$ times, using paired root identities `0` through `9`.
The total number of training runs is therefore $2 \times 10 = 20$:

| Algorithm | Actor | Observation | Roots |
|---|---|---|---:|
| PPO | Medium `(64, 64)` | Frenet | 10 |
| PPO | Medium `(64, 64)` | LiDAR | 10 |

### Fixed Conditions

Every run uses:

- The medium `(64, 64)` PPO actor selected from the original Experiment 1 candidate set before this rerun; the later tiny actor does not alter that selection;
- The learning rate selected before the experiment for PPO;
- A budget of $1\,000\,000$ training interactions;
- A deterministic evaluation over $16$ fixed circuits, every $50\,000$ training interactions;
- A checkpoint save every $250\,000$ interactions and at the end;
- A final test over $32$ fixed circuits.

The original split manifest is retained: 16 validation circuits, 32 test circuits and 16 training-reference circuits. The original timeout treatment is retained: REINFORCE stops its return at the task deadline, while PPO bootstraps once from the critic at that deadline.

### PPO Actor Selection

The medium PPO architecture was selected from the original three-size Experiment 1 candidate set before this revised experiment:

1. Find the size with highest mean final deterministic return;
2. Compute paired root-level return deficits for every other size;
3. Admit a size whose mean deficit is no greater than one standard error and whose completion count is no more than one root below the best size; 
4. Choose the admitted actor with the fewest parameters.

These criteria define the cost we accept to pay to avoid an architecture with too many parameters. The later tiny actor is an Experiment 1 extension and is not eligible to revise this already fixed architecture choice.

### Definition of Convergence 

Threshold attainment is the first of three consecutive evaluations (spanning
100,000 interactions), with confirmation at the third checkpoint, that:
* Have a validation completion rate of at least `0.75`, meaning that they successfully complete $12$ out of the $16$ circuits.
* Have median validation normalized progress of at least `0.95`. This condition
  is redundant once 12/16 circuits complete: it adds no guarantee about progress
  on the remaining four circuits. The original rule is retained.

### Recorded Data

Experiment 2 tracks the same quantities as Experiment 1, properly adjusted to adapt to the multi-circuit training and validation that characterize Experiment 2.

# Runs Outputs

The directory names state why the run exists:

```text
results/
  pre_experiment_configuration/
    <purpose>/<run-id>/
  reduced_budget_end_to_end_validation/
    <experiment-path>/<run-id>/
  reported_experiments/
    experiment_1/<run-id>/
    experiment_2/<run-id>/
    experiment_2_revised/<run-id>/
```

Every run directory contains:

```text
manifest.json
config.json
metadata.json
episodes.jsonl
updates.jsonl
evaluations.jsonl
checkpoints/
trajectories/
completion.json
```
