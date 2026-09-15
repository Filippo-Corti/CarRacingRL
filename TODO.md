# Approved execution scope — 2026-09-15

The user approved implementation, **all 35 new runs**, and both final reports.
This section supersedes conflicting instructions in the historical handoff below.

- Keep all 45 existing Experiment 1 runs. Add `(8, 8)` actors for all three
  algorithms, roots `0..4`, at 2M interactions. Present all 60 runs together.
- Rerun Experiment 2: medium PPO, Frenet/LiDAR, roots `0..9`, 1M interactions.
  Its final report includes only these 20 new runs. Medium stays fixed even if
  tiny performs better; preserve the original architecture-selection artifact.
- **Do not change timeout targets or any other training semantics.** Document
  the REINFORCE versus A2C/PPO timeout mismatch briefly. Work B is cancelled.
- Keep original rates, physics, rewards, observations, initialization, circuit
  splits, evaluation cadence (50k), checkpoints (250k plus final), eight CPU
  workers and one Torch intra/inter-op thread. Train sequentially.
- Run every new condition; the user manages the deadline. Use GPT-5.6 Terra
  subagents with high reasoning for bounded tasks and review their work.
- Combined Experiment 1 uses `results/reported_experiments/experiment_1/`.
  Preserve historical Experiment 2 in `experiment_2/`; new runs use
  `experiment_2_revised/`. Analysis mirrors these names. Add a results index.
- Replace `docs/EXPERIMENT_1.md` and `docs/EXPERIMENT_2.md` with the requested
  complete reports and update their figure directories.
- Freeze analysis conventions: Experiment 1 exhaustive root bootstrap;
  Experiment 2 10,000 whole-root resamples, seed 0, percentile 95% intervals.
  Pair root-level differences. Circuits are aggregated within each root first.
- Preserve first-of-three threshold attainment and final outcomes. Add recorded
  confirmation cost and late stability at `0.8B < interactions <= B`.
  Common-budget comparisons use 250k, 500k, 750k, 1M (and 2M for Experiment 1).
- Analyze every root's final controls; retain all 32 final test trajectories
  per Experiment 2 run. Keep action/distance alignment, separate straight
  curvature, disclose thresholds, and use representative traces as illustrations.

## Approved implementation and validation checklist

- [x] Configs/matrices and safe result handling: `src/configs/`, experiment
  runners, `experiments/matrix.py`, focused tests.
- [x] Final-test trajectory retention: `experiments/train.py`, logging config
  and focused tests, with original training behaviour preserved.
- [x] Root uncertainty, convergence, size/algorithm panels and controls:
  analysis/plotting utilities, `experiments/analyze_results.py`, focused tests.
- [x] Protocol/notebook consistency and factual corrections: README, protocol,
  MDP/learning/track docs where needed, both experiment notebooks.
- [x] Review delegated changes; Black, applicable tests and repository checks.
- [x] Reduced rehearsals, final matrix dry runs, settings/dependency/code freeze.
- [x] Preserve original-data checksums; complete 15 tiny and 20 Experiment 2 runs.
- [x] Verify results and regenerate combined Experiment 1/new-only Experiment 2.
- [x] Write both reports and figures; commit separate concerns on main and
  record validation/results with commit references in `docs/DIARY.md`.

### Reporting completion — 2026-09-15

Both standalone reports now analyze the complete 60-run Experiment 1 and the
new-only 20-run Experiment 2, with refreshed figures and all-root control data.
The reports answer the capacity, common-budget, attainment/confirmation,
late-stability, replication, generalization and learned-control questions below.

The reporting audit found that `src/utils/analysis.py` uses a 100-second lap
cutoff when constructing Experiment 1 threshold inputs, although the protocol
specifies 34 seconds. The report recomputes the 34-second streaks directly from
saved evaluations; two roots' attainment boundaries change. The corrected
60-root table is `docs/tables/experiment_1_thresholds.csv`, and the report's
threshold figure uses it. The processed bundle and analysis helper retain their
existing values, so regenerating them without correcting the helper will not
reproduce the report's corrected threshold costs. Training is unaffected.

Experiment 2 additionally includes a matched-successful-lap check, with circuit
identities and source checksums in `docs/tables/experiment_2_matched_laps.csv`.
See the report evidence sections and diary for validation and provenance.

---

# Historical handoff and review notes

**Created:** 2026-09-15. **Status:** implementation and experiment execution pending.

This is a self-contained handoff for the next agent. It records the scientific
review, the user's time constraints, and the proposed implementation scope. The
request that created this file was to document the plan; it did not execute the
changes or launch training. When asked to implement it, follow the applicable
`AGENTS.md` and any newer user instructions. Update the checklists with evidence
as work is completed.

## Objective and constraints

Prepare a defensible, clearly explained comparison for a university RL exam.
The project asks how policy-network complexity affects driving performance,
interactions needed to learn, reliability, and training time. A second study
compares Frenet observations with LiDAR on unseen generated circuits.

The user has little time for further work and wants to reserve the final day for
deciding how to present the results. Prioritize correct claims, consistent
training targets for new runs, and readable comparisons with driving behaviour.
Do not expand this into a new research programme.

### Scope established in the discussion

- Correct factual errors and claims unsupported by the available evidence.
- Reconcile the treatment of future value at episode timeouts for new training.
- Preserve the 45 completed Experiment 1 runs. Do not rerun that full matrix or
  increase its five training roots. More roots are future work for Experiment 1.
- Add one tiny actor to Experiment 1 only if it can use the original training
  settings and be compared fairly with the existing actors.
- Improve comparisons of sizes within each algorithm and algorithms within each
  size, including convergence and controls.
- **Rerun Experiment 2 only**, with more roots and a smaller per-run interaction
  budget. The user explicitly clarified that this does not mean rerunning both
  experiments.
- Analyze the learned controls in both experiments.

### Explicit numerical proposals carried by this handoff

The preceding plan proposed a tiny `(8, 8)` actor and an Experiment 2 rerun with
ten paired roots at one million training interactions each. These were proposed
follow-up values, not settings of the original completed experiments. Use them
as the concrete implementation proposal; do not describe them as original
preregistered choices. Freeze the values in the implementation plan before
reported runs, applying any subsequent user approval or correction.

| Study | Actors / observations | Algorithms | Roots | Interactions per run | Timeout treatment |
|---|---|---|---|---:|---|
| Original Experiment 1 | `(32, 32)`, `(64, 64)`, `(256, 256)`; Frenet | REINFORCE, A2C+GAE, PPO | Original `0..4` | 2,000,000 | Original treatment retained |
| Experiment 1 extension | Proposed `(8, 8)`; Frenet | Same three | Same paired `0..4` | 2,000,000 | Exactly the original treatment |
| Original Experiment 2 | Medium `(64, 64)`; Frenet and LiDAR | PPO | `0..4` | 2,000,000 | Preserve records as historical results |
| Revised Experiment 2 | Same medium actor and observations | PPO | Proposed `0..9` | 1,000,000 | Zero future value at the task deadline |

Keep the original medium PPO architecture selection for Experiment 2. Treat the
tiny-actor study as a follow-up, without allowing its result to silently change
the architecture used in the revised observation comparison.

### Time and fallback

The reported original runtimes were 316.7 minutes for Experiment 1's 45 runs and
135.8 minutes for Experiment 2's ten runs. On the same machine, estimate:

- Tiny actor: 15 new runs, roughly 1.5–1.75 hours.
- Revised Experiment 2: 20 new runs, roughly 2.25 hours; its total training
  budget remains 20 million interactions, as in the original Experiment 2.
- Total computation: roughly four hours, plus implementation, checks, data
  handling, and interpretation. These are estimates, not guarantees; changed
  episode behaviour and per-run overhead can change runtime.

Prepare analysis while training runs, but run training jobs sequentially on the
same experiment machine for comparable timings. If time becomes tight, omit the
tiny-actor extension first. Do not sacrifice truthful reporting or the user's
presentation day to complete an optional extension.

## Read first and locate the data

1. [AGENTS.md](AGENTS.md): repository workflow and coding rules. Use `.venv`,
   follow Black formatting, keep concerns in separate commits on `main`, and
   document completed work in the diary with its commit reference. Respect the
   user's existing authorization rather than asking again for already approved
   steps. This handoff is not an instruction to spawn agents automatically.
2. [README.md](README.md): project objective and entry points. Some roadmap text
   is stale; the old Phase 2 plan is archived, and LiDAR is already implemented.
3. [EXPERIMENT.md](docs/EXPERIMENT.md): original protocol and calibration.
4. [EXPERIMENT_1.md](docs/EXPERIMENT_1.md) and
   [EXPERIMENT_2.md](docs/EXPERIMENT_2.md): reported tables, interpretations,
   figures, and the user's original TODO questions.
5. [LEARNING.md](docs/LEARNING.md), [MDP.md](docs/MDP.md), and
   [TRACK.md](docs/TRACK.md): algorithm equations, objective, physics, and track
   representation. Derive algorithm changes from these specifications and the
   references below rather than assuming a library's usual implementation.
6. [DIARY.md](docs/DIARY.md): development and calibration history. Historical
   interpretations are not all established causal findings.
7. [Experiment 1 notebook](notebooks/experiment_1.ipynb),
   [Experiment 2 notebook](notebooks/experiment_2.ipynb), and their executable
   counterparts under `experiments/`. Keep the supported paths consistent.

At handoff creation, the two experiment reports already had uncommitted user
edits. Inspect current `git status` and preserve the user's work. The raw
`results/` directory is untracked and was on another device during the review.
You should now have access to it.

Original data locations are:

- `results/reported_experiments/experiment_1/`
- `results/reported_experiments/experiment_2/`
- `results/analysis/reported_experiments/experiment_1/`
- `results/analysis/reported_experiments/experiment_2/`

Each run normally has `manifest.json`, `config.json`, `metadata.json`, episode,
update and evaluation JSONL streams, checkpoints, selected trajectories, and
`completion.json`. Record the source code/configuration and data identity used
for any regenerated table. When raw data is unavailable, complete the code,
checks, and exact execution instructions, and clearly leave data-dependent
tasks pending. Run the final matrices on the original experiment machine.

## Scientific context that the implementation must preserve

The vehicle uses deterministic grip-limited bicycle dynamics, a steering-rate
limit and drag. Actions are normalized throttle/brake and steering. Training
uses stochastic tanh-squashed Gaussian actions and randomized start poses;
evaluation uses `tanh(mean)` from a canonical zero-speed start with frozen
normalization. Evaluation interactions do not count toward the training budget.

Frenet supplies five inputs: lateral distance, heading error, speed, steering
angle and preview curvature. LiDAR supplies speed, steering angle and sixteen
ranges: eighteen inputs. Both representations omit information from the full
environment state. Do not claim that only LiDAR is partially observable or that
the Frenet vector necessarily contains all information available in LiDAR.

Original learning rates, kept for the follow-up:

| Algorithm | Actor rate | Critic rate |
|---|---:|---:|
| REINFORCE | `1e-3` | None |
| A2C+GAE | `1e-3` | `3e-3` |
| PPO | `3e-4` | `1e-2` |

The critic has hidden widths `(64, 64)`. Discount is `1.0`; GAE lambda is `0.95`.
A2C/PPO collect 2,048 transitions per rollout. PPO uses minibatches of 64, four
epochs, clip epsilon 0.2 and target KL 0.02. REINFORCE uses eight complete
episodes per update. Read the configs for the remaining initialization,
normalization, optimizer and policy-dispersion settings; preserve them.

Original deterministic evaluation occurs every 50,000 training interactions;
checkpoints are saved every 250,000 interactions and at the final budget. Retain
this cadence for the proposed new runs. Do not change evaluation cadence in the
tiny extension and then claim identical resolution to the original size sweep.

### What the published summaries currently support

- Experiment 1: A2C small/medium have mean final laps near 24.9 seconds; A2C large
  is near 30.7 seconds, with about a 26-point paired return deficit. Actor width
  therefore does affect learned performance under these training settings.
- At medium width, REINFORCE / A2C / PPO report laps of 29.92 / 24.90 / 26.52
  seconds, all with 5/5 final completions. Their threshold interaction ranges
  are 250k–750k / 400k–1,150k / 100k respectively.
- PPO medium and large have similar reported final outcomes within substantial
  uncertainty. Similar endpoints do not establish equal learning curves or
  equivalent performance.
- REINFORCE large has one root that never learned; PPO small has one final
  failure after earlier successful checkpoints. Both count in the results.
- Experiment 2 reports test completion of 0.894 for Frenet and 0.838 for LiDAR.
  The paired root-level interval for their completion difference is
  `[-0.131, +0.325]`: the relative performance remains uncertain.
- The results support useful generalization within the frozen generator's
  distribution. They do not establish generalization to other circuit families,
  real F1 driving, or the absence of any generalization gap.

These are reported outcomes, not independently reproduced measurements. Keep
the original tables traceable when correcting the accompanying explanations.

## Work A — Correct documentation and interpretation

- [ ] Revise both experiment reports and any corresponding notebook prose.
  Preserve original outcome measurements unless a correction can be derived
  unambiguously or regenerated from the source records.
- [ ] Add explicit original-versus-follow-up protocol descriptions. A correction
  to the prose must not imply that an old run used corrected training targets.
- [ ] Link this follow-up from the README and correct directly relevant stale
  roadmap claims without starting a broad documentation rewrite.

> WRONG: I don't care about stating original-versus-follow-up. Just make it so that it looks like one full experiment. Don't correct the target issues with REINFORCE vs A2C/PPO. Just document that it exists.

### Specific corrections already identified

1. **Capacity:** remove the claim that width has no effect. Describe A2C's large
   actor deficit and the algorithm-dependent response. Fixed learning rates
   selected on the medium actor establish performance under that recipe, not
   each architecture's best performance after independent tuning. Uneven
   parameter-count ratios do not imply proportional performance effects.
2. **Uncertainty:** replace "statistically identical", "as well as", and
   equivalence claims based only on intervals containing zero. Failed roots are
   part of the reliability distribution, not disposable outliers or SD
   artifacts. Exhaustively enumerating a five-root bootstrap does not make its
   inferential coverage exact. Treat multiple comparisons as exploratory.
3. **Pairing:** shared root identities preserve a designed pairing, but do not
   guarantee that root variation cancels. Different architectures and policies
   produce different trajectories. Experiment 2 pairs each worker's circuit
   sequence; different episode lengths can yield unequal total circuit exposure.
4. **Experiment 2 replication:** there are five original training-root pairs,
   each evaluated on the same 32 test circuits, not 160 independent trained
   pairs or 160 distinct circuits. The five printed differences imply a
   root-level SE of about 0.134 for completion. The pooled circuit SE of 0.038
   cannot stand in for uncertainty across training roots.
5. **Convergence:** the metric is the first checkpoint of a three-checkpoint
   qualifying streak. It does not establish permanent convergence. Three
   checkpoints 50k apart span 100k from first to third; confirmation occurs
   100k after the reported first checkpoint. Separate initial attainment,
   confirmation cost, final performance and later regressions.
6. **Convergence cost:** complete-run runtime at two million interactions is not
   runtime to the task threshold. Use recorded `convergence_duration`; do not
   infer that PPO is slower to reach the threshold from its longer full run.
7. **Experiment 2 threshold:** its completion rule is at least 12/16 validation
   circuits, plus median progress at least 0.95. Once 12 circuits complete, the
   median is already 1.0. The second condition adds no guarantee about the four
   failures. Preserve the historical rule and explain the redundancy; do not
   silently substitute a new threshold for the follow-up.
8. **Experiment 2 timing:** Frenet's slowest reported first qualifying checkpoint
   is 650k of 2M, or 32.5%; not every root converges within 10% of the budget.
9. **Frozen policy versus changing checkpoints:** deterministic evaluations at
   successive checkpoints use different policies and normalization states.
   Alternating success/failure shows training instability, not one frozen
   "bimodal policy". Keep the final failure as the final result and supplement
   it with late-window stability. Repeating an identical frozen evaluation does
   not add independent evidence.
10. **Critic quality:** low explained variance against recorded targets, together
    with good driving, does not prove that accuracy is irrelevant or that the
    critic effectively reduces variance. GAE with lambda below one bootstraps,
    so value errors can bias advantages. Stronger claims need an ablation.
11. **PPO noise:** reaching the log-standard-deviation ceiling is worth reporting,
    but a constrained optimum can lie on a boundary. Pre-squash Gaussian scale
    is not actual bounded-action variation. Do not attribute a plateau or
    steering behaviour to the ceiling without a controlled comparison. Identical
    bounds do not guarantee identical effects across conditions.
12. **Gradient diagnostics:** a common gradient-norm clipping threshold does not
    impose identical Adam parameter updates. Gradient dimension and loss
    reduction affect raw norms; avoid causal comparisons based on them alone.
13. **Parameter counts:** the Frenet critic has 4,609 parameters, which exceeds
    the small actor's 1,316 but is below the medium actor's 4,676. The LiDAR
    critic has 5,441 parameters, not 4,609. Both actor and critic input layers
    change width in Experiment 2. A small parameter difference is not proof that
    it cannot affect performance; retain equal hidden widths as the control.
14. **PPO architecture selection:** small's 4/5 completions satisfy the
    within-one-root completion criterion relative to 5/5. It fails the return
    criterion, not both criteria. Preserve the original medium selection.
15. **Curvature:** the saved Experiment 1 track is 58.9% straight. Its absolute
    curvature quartile edges are approximately `(0, 0, 0.0548572)`.
    `searchsorted(..., side="right")` puts zero curvature in q3, leaving q1/q2
    empty. Of 1,070 centerline samples, the bins contain `(0, 0, 760, 310)`.
    "Curved almost everywhere" is false; the existing q3 mixes straights and
    milder corners. The old q3/q4 means are not four informative quartiles.
16. **Driving mechanisms:** mean signed throttle near zero can indicate coasting
    or alternating throttle/brake; positive throttle can merely offset drag.
    Mean absolute steering does not demonstrate oscillation, a clean racing line,
    or speed being scrubbed off. Inspect signed traces and the actual dynamics.
17. **Memory:** `peak_process_memory()` excludes environment workers. The matrix
    runs sequentially in one process, so its process-lifetime high-water mark
    can carry over between runs. Do not attribute identical peaks to worker
    costs, or the reported 132 MB LiDAR difference to wider observation buffers.
    Historical memory cannot be repaired from those peak numbers alone.
18. **Generalization/geometry:** a near-zero point gap is not proof of no gap.
    Coarse length/curvature bins do not prove geometry is irrelevant, and the
    validation split's lower score is not established to be sampling noise.
    A higher crash rate does not establish that LiDAR "misread" its ranges.
    The LiDAR per-root table includes 0.781 as well as 0.438; the assertion that
    all four other roots lie between 0.906 and 1.000 is incorrect.

Also check the MDP reward illustration: it omits approximately 100 points of
accumulated progress reward when comparing a 22-second lap with a 33-second
lap. Including it gives approximately 241 versus 191.5, still just over the
stated 20% relative separation. The exact total depends on terminal-step
accounting. Correct the explanation from the implemented reward without
changing its coefficients. Remove unsupported assertions that another discount
would necessarily produce better results.

**Acceptance:** interpretations distinguish observations from hypotheses, every
comparison names its budget and training treatment, and no historical claim
quietly changes its meaning through a prose correction.

## Work B — Make timeout treatment explicit and compatible

> WRONG: Don't do any of this. It will be slightly incompatible but that's ok. Add a small note to remember about this possible issue. We can't fix it now. I would have to re-run everything.

### Why this matters

The task ends after 1,000 agent steps, at 0.04 seconds each: a 40-second deadline.
The objective is the undiscounted return within that episode, and the finish
bonus depends on the unused episode clock. REINFORCE's Monte Carlo return stops
at the deadline. Currently, A2C/PPO use zero bootstrap only when `terminated` is
true; the deadline is marked `truncated`, so they add a critic value there.
That is the target mismatch to correct for new training.

This differs from cutting a rollout to perform an optimizer update: that cut
does not end the underlying episode and must retain its bootstrap.

- [ ] Introduce an explicit timeout treatment in the appropriate configuration,
  recorded with each run and included in compatibility checks. Names should
  describe behaviour; avoid experiment-number switches inside agent equations.
- [ ] Preserve original behaviour for the original Experiment 1 configuration
  and its tiny extension: REINFORCE stops its return; A2C/PPO bootstrap at the
  deadline. Document this historical limitation honestly.
- [ ] For revised Experiment 2, use zero bootstrap at the actual task deadline.
  Make the corrected treatment available consistently to both A2C and PPO for
  future use; REINFORCE already has the required finite-episode return.
- [ ] Keep genuine termination bootstrap zero under both treatments. Keep
  bootstrap at a nonterminal rollout cut. Stop the GAE recursion at episode
  boundaries and never recurse into a reset episode or another worker's history.
- [ ] Reconcile the environment flags, transition semantics, documentation and
  targets. If retaining Gymnasium `truncated` for the deadline to preserve the
  environment interface, explain the explicit task-deadline target rule; do not
  blindly zero every possible future external truncation.
- [ ] Update the full-state formalization to acknowledge the episode clock,
  lap/start progress and stall history. Keep actor observation dimensions fixed
  for this scope and explicitly describe their partial observability. Zeroing
  timeout bootstrap alone does not make either observation Markov.
- [ ] Read old saved configurations as their original semantics where safe.
  Do not silently reinterpret old runs or resume an old checkpoint into a new
  learning contract. Preserve old checkpoint evaluation where practical without
  redesigning checkpoint infrastructure.

**Small test oracle:** with `reward=1`, `current_value=2`, `next_value=10` and
`discount=1`, the final TD error is `-1` at a task deadline under the corrected
treatment, but `9` under the original timeout-bootstrap treatment. At an
ordinary nonterminal rollout cut it is `9`; at a true terminal state it is `-1`
under both treatments. These are synthetic test values, not hyperparameters.

**Compatibility gate:** compare short deterministic original-setting runs
before and after the change, ignoring timestamps/resource measurements. They
must retain the same substantive records. If preserving Experiment 1 requires
broad changes or cannot be verified, omit its tiny extension; do not present
mixed training treatments as a controlled size comparison.

## Work C — Protect records and extend the matrices

- [ ] Give revised Experiment 2 a separate results root and analysis output.
  Use an explicit study identity in analysis so old and new root `0` runs cannot
  collide or be pooled. Preserve the original reports/results as a named study.
- [ ] Strengthen result compatibility for changed budget, timeout treatment and
  other result-affecting settings actually used by these runners. Current
  `learning_contract()` checks environment settings and discounts but does not
  fully describe these changes. A changed implementation with unchanged discount
  can currently be skipped as if its old run were valid.
- [ ] Account for the fact that `matrix.execute()` can delete a run directory
  before restarting it. A mismatch must not destroy historical results. Prefer
  a fresh explicit output location and non-destructive handling of old runs.
- [ ] Add proposed tiny `(8, 8)` to actor typing, exports, runner selection,
  supported notebooks and comparison generation. It has 140 trainable actor
  parameters for five Frenet inputs and the existing two log scales.
- [ ] Offer a way to execute only the tiny extension. Verify the original 45
  runs are neither scheduled unnecessarily nor invalidated by new metadata.
  Its seeds, circuit, reward, two-million budget, rates, observations, critic,
  normalization, initialization and evaluation remain the original settings.
- [ ] Mark the extension as exploratory and preserve the original PPO selection
  artifact/rule over its original candidate set. Do not let reanalysis with the
  fourth actor silently overwrite the medium selection used by Experiment 2.
- [ ] Configure proposed revised Experiment 2 as twenty runs: roots `0..9` by
  two observations, one million interactions each, medium actor, corrected
  timeout targets. Retain the original generator, split manifest, rates, physics,
  reward, initialization, PPO schedule and evaluation cadence.
- [ ] Preserve per-worker circuit pairing between observations, independent RNG
  streams, and explicit reproducible seeds. Using logical roots `0..4` again
  does not make old and new training runs independent replicates to pool.
- [ ] Freeze the revised protocol before examining new test results. The old
  test outcomes already informed this follow-up, so describe the reuse of the
  existing split transparently; do not call it a new untouched confirmation set.
  Do not tune settings or select checkpoints using the new test outcomes.

**Acceptance:** dry-run listings show 15 tiny-only runs and 20 revised
Experiment 2 runs with correct settings and separate destinations. Original
data remains readable and unmodified. Loading a mixed/incompatible study fails
clearly instead of creating a misleading aggregate.

## Work D — Comparisons, uncertainty and controls

### Statistical and convergence outputs

- [ ] Replace or extend `descriptive_statistics()`: its exhaustive bootstrap
  currently rejects more than six roots. Use a tractable seeded method that
  resamples whole roots and preserves paired differences. Make the resample
  count, seed and interval method explicit; keep the historical analysis
  reproducible rather than silently changing its interval method.
- [ ] Treat Experiment 2's primary observations as one circuit-aggregated result
  per root. Show per-root paired differences and their interval. Keep per-circuit
  contrasts descriptive; if inferring over new circuit draws as well, explain
  and implement a resampling method that respects both root and circuit
  dependence. Do not multiply the number of training replicates by 32.
- [ ] Add panels comparing widths within algorithms and algorithms within widths.
  Include individual roots and readable summaries; avoid another nine- or
  twelve-curve figure with overlapping bands.
- [ ] Report final completion counts, progress, return and lap times together.
  Lap times are conditional on completion; always show the denominator.
- [ ] Report threshold attainment in interactions, episodes and actual recorded
  training duration, retaining censored runs and labeling any budget-capped
  summary. Never silently drop failed/nonconverged roots or treat the budget as
  an observed convergence time.
- [ ] Supplement the original threshold rule with late-training stability and
  common-budget comparisons. Preserve the original final endpoint. Define any
  new checkpoint sample or late window explicitly before viewing its outcome;
  no retrospectively selected best checkpoint and no forced smoothing away of
  failures. Label the first-versus-confirmation convention.
- [ ] Generalize pair generation to include tiny. The current
  `experiments/analyze_results.py::_experiment_pairs()` explicitly lists the
  three original size pairs; adding an actor constant is insufficient.

### Driving outputs and retention

- [ ] Plot pre-action speed, signed throttle/brake and signed steering against
  distance around the same circuit. Mark geometric corners and include XY
  trajectories where they help explain the driving line. Preserve time/order
  alignment: logged `progress` is post-action while speed/position are pre-action.
- [ ] Fix curvature binning. Give straight samples their own group and handle
  positive-curvature bins without duplicate boundaries. State any numerical
  tolerance. The generated centerline samples have exact zero on straights;
  projected/interpolated observations may require care at transitions.
- [ ] Report braking fraction, throttle and speed distributions, steering
  changes/sign reversals, and existing saturation diagnostics. Define any new
  threshold explicitly. Normalized requested steering, actual wheel angle and
  effective grip-limited turning are different quantities; label what is plotted.
- [ ] Inspect corner approaches, corners and exits: braking before a corner can
  be missed by conditioning only on current curvature. A constant positive
  throttle may balance drag; a small mean can hide alternating commands.
- [ ] Aggregate controls within a root/circuit first, then compare roots. State
  whether values are weighted by time, distance or episode. Longer failed or
  slower trajectories must not acquire unexplained statistical weight.
- [ ] Include all roots in summaries. A representative trajectory is an
  illustration, not evidence of a mechanism shared by the whole condition.
  Keep failures and their shorter observed coverage visible.
- [ ] In revised Experiment 2, retain matched final test trajectories for every
  root and observation on a fixed declared circuit subset, or all 32 if storage
  is acceptable. Label illustrative failures separately from the fixed subset.
  Ensure that a global quota is not consumed by validation evaluations first.
- [ ] For original Experiment 1, use saved final trajectories. If additional
  trajectories are needed, evaluate compatible frozen checkpoints on the results
  machine and save a separate analysis artifact; that does not require training.
  Do not substitute newly trained agents for missing historical trajectories.

**Existing retention trap:** `LoggingConfig.trajectory_circuits_per_boundary`
defaults to two. `_write_engine_records()` in `experiments/train.py` counts
retained trajectories by interaction boundary, across all splits. Final
validation evaluations occur before training-reference and test evaluations, so
the quota can prevent any final test trajectories from reaching disk. The
existing curvature analysis also selects only one representative root per cell
and does not establish that its final trajectory came from the test split.

**Optional only if time remains:** overlay the existing scripted controller on
the same circuit. A constant-target-speed baseline could test the user's
"sailing at a speed that works" hypothesis, but building/tuning a new controller
is not required for the core handoff.

**Acceptance:** tables include all roots and outcomes, uncertainty uses the
proper replicate, tiny appears in all relevant comparisons, and controls are
traceable to named checkpoints/circuits/splits with correct action alignment.

## Implementation map

These are likely touch points, not permission for a broad refactor. Inspect
current call sites and nested instructions before editing.

| Concern | Primary files and packages |
|---|---|
| Reports and protocol | `README.md`, `docs/EXPERIMENT.md`, `docs/EXPERIMENT_1.md`, `docs/EXPERIMENT_2.md`, `docs/LEARNING.md`, `docs/MDP.md`, `docs/TRACK.md` where needed |
| Diary and handoff status | `docs/DIARY.md`, `TODO.md` |
| Actor/algorithm/matrix settings | `src/configs/training.py`, `src/configs/algorithms.py`, `src/configs/experiments.py`, `src/configs/__init__.py`; serialization only if needed |
| Targets and callers | `src/agents/targets.py`, `src/agents/implementations/a2c.py`, `src/agents/implementations/ppo.py`, relevant environment/transition code |
| Run/checkpoint compatibility | `experiments/matrix.py`, `experiments/train.py`, `src/training/checkpointing.py`, relevant recording/analysis identities |
| Executable studies | `experiments/experiment_1.py`, `experiments/experiment_2.py`, `notebooks/experiment_1.ipynb`, `notebooks/experiment_2.ipynb` |
| Analysis and figures | `src/utils/analysis.py`, `src/utils/plotting.py`, `experiments/analyze_results.py`, `experiments/reporting.py` |
| Evaluation and retention | `experiments/train.py`, `src/evaluation/deterministic.py`, `src/evaluation/scheduler.py`, `src/recording/records.py` only if existing fields are insufficient |
| Resource interpretation | `src/utils/resources.py`; correct reporting first, avoid a new profiling project |
| Existing focused tests | `tests/training/test_buffers.py`, agent tests, config tests, checkpointing tests, `tests/experiments/test_matrix.py`, analysis/plotting/reporting tests |

## Validation and execution order

1. [ ] Inspect the working tree and available results, then state the concrete
   implementation plan/files under the repository's plan-before-code workflow.
   Preserve user edits and record numerical decisions in configuration/docs.
2. [ ] Correct documentation; implement explicit compatible timeout treatment,
   safe result identities, proposed matrices and sufficient trajectory retention.
3. [ ] Test target boundary cases, original-setting reproducibility, incompatible
   resume/skip behaviour, and survival of historical output directories.
4. [ ] Implement and check analysis on small fixtures: more than six roots,
   paired differences, censored cases, tiny comparisons, tied curvature edges,
   split-aware trajectory retention and action alignment. Run Black and the
   applicable repository checks. Avoid unnecessary full-budget test training.
5. [ ] Run a reduced rehearsal of each changed path under the existing reduced
   validation category, with separate seeds/output. Check the full
   recording-to-analysis path, including corrected PPO under both observations.
   Ensure the rehearsal has enough evaluation checkpoints for convergence/AUC.
6. [ ] Print/inspect the actual final matrices and destinations. Freeze the
   execution manifest, settings, code commit and exact dependency environment.
   The current experiment CLIs do not yet expose arbitrary roots, budgets or
   separate revised destinations; add the necessary interface before publishing
   commands. Do not invent flags in the documentation.
7. [ ] On the experiment machine, execute revised Experiment 2 and, if the time
   and compatibility gates pass, the tiny-only extension. Use the same machine
   and worker configuration as the originals, with one training job at a time.
   Analysis preparation may proceed separately without competing training jobs.
8. [ ] Regenerate tables/figures from the resulting records. Check run counts,
   budgets, treatments, complete markers, root pairing and actual retained test
   circuits before interpreting plots. Keep original and revised studies apart.
9. [ ] Write concise findings that answer the original TODO questions. Retain
   five-root uncertainty as an Experiment 1 limitation, distinguish new
   observations from explanations, and record what was omitted due to time.
10. [ ] Commit separate concerns to `main` using `type: summary [ai]`, with diary
    entries referencing their commits. Do not commit raw results or unrelated
    user edits. Record which commit produced each reported experiment.

The current analysis CLI can consume explicit data roots; check its help before
using it. For reference, its present shape is:

```bash
.venv/bin/python experiments/analyze_results.py \
  --results-root PATH_TO_ONE_STUDY \
  --output PATH_TO_ITS_ANALYSIS \
  --experiment 2 \
  --geometry-specification tracks/experiment_2_splits.json
```

The uppercase path tokens are placeholders. Final execution instructions must
name the real verified directories and implemented runner arguments.

## Completion criteria and presentation handoff

- [ ] Existing reports have corrected facts and appropriately bounded claims.
- [ ] Original Experiment 1 results remain intact, with its timeout limitation
  disclosed. Tiny results use the original settings, or their omission is stated.
- [ ] Revised Experiment 2 has the frozen, explicitly recorded new settings and
  twenty completed run records, or any missing/failed runs are reported honestly.
- [ ] Old and revised results are not mixed in a final table or uncertainty
  calculation. The old medium architecture selection is preserved.
- [ ] Comparisons answer both "which size for this algorithm?" and "which
  algorithm at this size?", using final performance, threshold cost and stability.
- [ ] Control summaries and matched traces support statements about actual
  driving; unavailable evidence is identified rather than replaced with a story.
- [ ] The user receives reproducible commands, figure/table locations, concise
  conclusions, limitations, and a short list of useful presentation figures.
- [ ] The final day remains available for the user's exam presentation work.

### Explicit future work

More Experiment 1 roots; broader actor/critic and learning-rate sweeps; a global
rerun of Experiment 1 with corrected timeout targets; recurrent or time-aware
observations; richer geometry/other generators; exploration-ceiling ablations;
new control baselines; and accurate isolated per-run memory profiling. Do not
silently add these to the present scope.

### Primary references used in the review

- [Pardo et al., Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378):
  distinguishes task deadlines from artificial collection cutoffs.
- [Schulman et al., Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438):
  value approximation, bootstrapping and the bias/variance trade-off.
- [Agarwal et al., Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264):
  uncertainty and the limitations of a small number of training runs.
- [Patterson et al., Empirical Design in Reinforcement Learning](https://arxiv.org/abs/2304.01315):
  experimental budgets, performance variation, tuning and interpretation.

Use the repository's actual equations for implementation. These references
provide methodological context; they do not establish causes for these agents'
reported behaviour.
