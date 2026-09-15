"""Run a design matrix of training runs, resumably.

An experiment here is dozens of independent runs that together take hours, so
the loop that drives them has to survive being interrupted. It does that by
treating the results tree as the record of what has been done: a run whose
`completion.json` exists is finished and is skipped, and everything else is
started from scratch.

Nothing about the experiments themselves lives here. The matrices, their
configurations and their meaning belong to the notebooks that define them; this
only knows how to execute a list of specifications and report what happened.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(frozen=True, slots=True)
class RunSpecification:
    """
    Name one run of a design matrix and say how to start it.

    Fields:
        * run_id: Identity of the run, unique within its matrix.
        * path: Directory the run writes its records into.
        * launch: Starts the run; called with no arguments.
        * expected_contract: Run-specific settings a completed record must match.
    """

    run_id: str
    path: Path
    launch: Callable[[], object]
    expected_contract: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """
    Report what happened to one specification.

    Fields:
        * run_id: Identity of the run this describes.
        * status: One of `completed`, `skipped` or `failed`.
        * duration: Wall seconds spent, zero for a skipped run.
        * error: Formatted traceback when the status is `failed`.
    """

    run_id: str
    status: str
    duration: float
    error: str | None = None


def is_complete(path: Path) -> bool:
    """
    Return whether a run directory holds a finished run.

    Completion is the last thing a run writes, so its presence means every
    record before it was written too. A directory without it is the debris of
    an interrupted run rather than a result.
    """
    return (Path(path) / "completion.json").is_file()


def learning_contract(
    environment_config: Any, *algorithm_configs: Any
) -> dict[str, Any]:
    """
    Name the recorded fields whose change invalidates an existing run.

    A results tree is the record of what has been done, and a matrix skips
    whatever it finds finished there. That is what makes an interrupted run
    resumable, and it is also how a run produced under different constants gets
    silently reused after a contract change. This is the list that makes the
    difference visible: the reward function, the physics, the circuit geometry
    and every algorithm's discount.

    The dotted paths address `config.json`, which every run writes before it
    trains. Every run records all three algorithm configurations regardless of
    which one it used, so one contract describes any matrix.
    """
    environment = environment_config.to_dict()
    contract: dict[str, Any] = {
        f"environment.{name}": environment[name]
        for name in ("reward", "simulation", "vehicle", "track")
    }
    for config in algorithm_configs:
        name = type(config).__name__.removesuffix("Config").lower()
        contract[f"training.{name}.discount"] = config.discount
    return contract


def contract_mismatch(path: Path, contract: Mapping[str, Any]) -> str | None:
    """
    Return why a finished run disagrees with the contract, or `None` if it does not.
    """
    document = Path(path) / "config.json"
    if not document.is_file():
        return "the run has no recorded configuration"
    recorded = json.loads(document.read_text(encoding="utf-8"))
    for dotted, expected in contract.items():
        value: Any = recorded
        for key in dotted.split("."):
            if not isinstance(value, dict) or key not in value:
                return f"{dotted} is not recorded"
            value = value[key]
        if value != expected:
            return f"{dotted}{_first_difference(value, expected)}"
    return None


def _first_difference(recorded: Any, expected: Any) -> str:
    """
    Describe the first differing field, rather than printing both documents.

    A reward or physics document has a dozen fields and one of them changed;
    naming it is the whole of what a reader needs to decide what to do.
    """
    if isinstance(recorded, dict) and isinstance(expected, dict):
        for key in sorted(set(recorded) | set(expected)):
            if recorded.get(key) != expected.get(key):
                return (
                    f".{key} was {recorded.get(key)!r} "
                    f"and is now {expected.get(key)!r}"
                )
    return f" was {recorded!r} and is now {expected!r}"


def execute(
    specifications: Sequence[RunSpecification],
    *,
    skip_complete: bool = True,
    contract: Mapping[str, Any] | None = None,
    report: Callable[[str], None] = print,
) -> list[RunOutcome]:
    """
    Run every specification that is not already finished.

    A failing run does not stop the matrix. Overnight, one bad specification
    ending a nine-hour queue would be worse than finishing the rest and looking
    at the failure in the morning, and because finished runs are skipped, the
    repaired specification can simply be run again. The caller is handed every
    outcome and decides what a failure means.

    A finished run that fails the contract is refused. It is evidence produced
    under another setting, and deleting it would erase the distinction the
    contract exists to protect. An incomplete directory is moved aside before
    restart because the recorder refuses to write into a non-empty directory.

    A `contract` from `learning_contract` turns the skip into a *checked* skip:
    a finished run whose recorded configuration no longer matches is refused.
    Without it, changing a constant leaves a results tree that mixes two
    contracts and looks complete.
    """
    outcomes: list[RunOutcome] = []
    total = len(specifications)
    for index, specification in enumerate(specifications, start=1):
        prefix = f"[{index}/{total}] {specification.run_id}"
        if is_complete(specification.path):
            expected_contract = {
                **(contract or {}),
                **(specification.expected_contract or {}),
            }
            mismatch = (
                None
                if not expected_contract
                else contract_mismatch(specification.path, expected_contract)
            )
            if mismatch is not None:
                error = f"recorded under a different contract ({mismatch})"
                report(f"{prefix}: REFUSED: {error}")
                outcomes.append(RunOutcome(specification.run_id, "failed", 0.0, error))
                continue
            if skip_complete:
                report(f"{prefix}: already complete, skipped")
                outcomes.append(RunOutcome(specification.run_id, "skipped", 0.0))
                continue
        if specification.path.exists():
            archived = _archive_incomplete(specification.path)
            report(
                f"{prefix}: moved incomplete directory to "
                f"{archived.relative_to(specification.path.parent)}"
            )
        report(f"{prefix}: running")
        started = perf_counter()
        try:
            specification.launch()
        # Deliberately blind: a matrix runner cannot know what a run might
        # raise, and the whole point is that one failure does not end the
        # queue. The traceback is kept and reported rather than swallowed.
        except Exception:  # noqa: BLE001
            duration = perf_counter() - started
            report(f"{prefix}: FAILED after {duration / 60:.1f} min")
            outcomes.append(
                RunOutcome(
                    specification.run_id, "failed", duration, traceback.format_exc()
                )
            )
            continue
        duration = perf_counter() - started
        report(f"{prefix}: done in {duration / 60:.1f} min")
        outcomes.append(RunOutcome(specification.run_id, "completed", duration))
    return outcomes


def _archive_incomplete(path: Path) -> Path:
    """
    Move interrupted output aside without destroying its partial evidence.
    """
    archive_root = path.parent / ".incomplete"
    archive_root.mkdir(exist_ok=True)
    index = 1
    while True:
        archived = archive_root / f"{path.name}-{index}"
        if not archived.exists():
            path.rename(archived)
            return archived
        index += 1


def summarize(
    outcomes: Iterable[RunOutcome], report: Callable[[str], None] = print
) -> int:
    """
    Print one line per status and return how many runs failed.
    """
    outcomes = list(outcomes)
    counts = {status: 0 for status in ("completed", "skipped", "failed")}
    for outcome in outcomes:
        counts[outcome.status] += 1
    spent = sum(outcome.duration for outcome in outcomes)
    report(
        f"{counts['completed']} completed, {counts['skipped']} skipped, "
        f"{counts['failed']} failed, {spent / 60:.1f} min spent"
    )
    for outcome in outcomes:
        if outcome.status == "failed":
            report(f"\n--- {outcome.run_id} ---\n{outcome.error}")
    return counts["failed"]
