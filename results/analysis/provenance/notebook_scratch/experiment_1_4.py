import sys
import warnings
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="pygame.pkgdata",
)

from configs import (
    LARGE_ACTOR_CONFIG,
    MEDIUM_ACTOR_CONFIG,
    SMALL_ACTOR_CONFIG,
    A2CConfig,
    EnvironmentConfig,
    ExecutionConfig,
    LoggingConfig,
    PPOConfig,
    ReinforceConfig,
    physical_cpu_count,
)
from matrix import RunSpecification, execute, learning_contract, summarize
from recording import RunCategory
from reporting import describe, read_table, show_figure, show_table
from train import run_a2c_training, run_ppo_training, run_reinforce_training
from utils.analysis import ppo_actor_selection_rows, selected_ppo_actor

# ---------------------------------------------------------------- scale ----
# The only switch in this notebook. A rehearsal exercises every cell of the
# matrix and the whole analysis in minutes; the protocol is 45 runs of two
# million interactions and takes roughly nine hours on eight workers.
#
# The run category travels with the choice on purpose. A rehearsal writes under
# `reduced_budget_end_to_end_validation/`, which the recording schema refuses to
# load as reported data, and it draws from a different seed namespace -- so a
# rehearsal can neither be mistaken for a result nor share randomness with one.
REHEARSAL = True

if REHEARSAL:
    TRAINING_INTERACTION_BUDGET = 60_000
    EVALUATION_INTERVAL = 5_000
    ROOTS = (0, 1)
    RUN_CATEGORY = RunCategory.REDUCED_VALIDATION
else:
    TRAINING_INTERACTION_BUDGET = 2_000_000
    EVALUATION_INTERVAL = 50_000
    ROOTS = (0, 1, 2, 3, 4)
    RUN_CATEGORY = RunCategory.REPORTED

# ------------------------------------------------------------- fixtures ----
TRACK_PATH = PROJECT_ROOT / "tracks" / "experiment_1.json"
RESULTS_ROOT = PROJECT_ROOT / "results" / RUN_CATEGORY.value / "experiment_1"
ANALYSIS_ROOT = PROJECT_ROOT / "results" / "analysis" / RUN_CATEGORY.value / "experiment_1"
EXECUTION_CONFIG = ExecutionConfig(environment_workers=physical_cpu_count())
STEERING_THRESHOLD = LoggingConfig().near_saturated_steering_threshold

ACTORS = {
    "small": SMALL_ACTOR_CONFIG,     # (32, 32)
    "medium": MEDIUM_ACTOR_CONFIG,   # (64, 64)
    "large": LARGE_ACTOR_CONFIG,     # (256, 256)
}
# Selected before the experiment; see the configuration check in EXPERIMENT.md.
ACTOR_LEARNING_RATE = {"reinforce": 1e-3, "a2c": 1e-3, "ppo": 3e-4}
CRITIC_LEARNING_RATE = {"a2c": 3e-3, "ppo": 1e-2}

print(f"budget {TRAINING_INTERACTION_BUDGET:,} | roots {ROOTS} | {RUN_CATEGORY.value}")
print(f"workers {EXECUTION_CONFIG.environment_workers} | results -> {RESULTS_ROOT}")
# A finished run is only reused if it was produced under these constants.
CONTRACT = learning_contract(
    EnvironmentConfig(), ReinforceConfig(), A2CConfig(), PPOConfig()
)

print(f"runs in this matrix: {3 * len(ACTORS) * len(ROOTS)}")
