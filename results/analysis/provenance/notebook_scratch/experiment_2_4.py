import json
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

from circuits import CircuitSplit, TrainingCircuitSchedule, load_split_circuits
from configs import (
    LARGE_ACTOR_CONFIG,
    MEDIUM_ACTOR_CONFIG,
    SMALL_ACTOR_CONFIG,
    EnvironmentConfig,
    ExecutionConfig,
    LoggingConfig,
    ObservationRepresentation,
    PPOConfig,
    physical_cpu_count,
)
from matrix import RunSpecification, execute, learning_contract, summarize
from recording import RunCategory
from reporting import describe, read_table, show_figure, show_table
from train import run_ppo_training

# ---------------------------------------------------------------- scale ----
# The only switch in this notebook. A rehearsal exercises every cell of the
# matrix and the whole analysis in minutes; the protocol is 10 runs of two
# million interactions and takes roughly three hours on eight workers.
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
SPLITS_PATH = PROJECT_ROOT / "tracks" / "experiment_2_splits.json"
RESULTS_ROOT = PROJECT_ROOT / "results" / RUN_CATEGORY.value / "experiment_2"
ANALYSIS_ROOT = PROJECT_ROOT / "results" / "analysis" / RUN_CATEGORY.value / "experiment_2"
EXPERIMENT_1_ANALYSIS = (
    PROJECT_ROOT / "results" / "analysis" / RUN_CATEGORY.value / "experiment_1"
)
EXECUTION_CONFIG = ExecutionConfig(environment_workers=physical_cpu_count())
ENVIRONMENT_CONFIG = EnvironmentConfig()
STEERING_THRESHOLD = LoggingConfig().near_saturated_steering_threshold

# Selected before the experiment; see the configuration check in EXPERIMENT.md.
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 1e-2
TRAINING_REFERENCE_CIRCUITS = 16

ACTORS = {
    "small": SMALL_ACTOR_CONFIG,
    "medium": MEDIUM_ACTOR_CONFIG,
    "large": LARGE_ACTOR_CONFIG,
}
OBSERVATIONS = {
    "frenet": ObservationRepresentation.FRENET,
    "lidar": ObservationRepresentation.LIDAR,
}
# A finished run is only reused if it was produced under these constants.
CONTRACT = learning_contract(ENVIRONMENT_CONFIG, PPOConfig())

print(f"budget {TRAINING_INTERACTION_BUDGET:,} | roots {ROOTS} | {RUN_CATEGORY.value}")
