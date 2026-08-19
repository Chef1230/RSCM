"""Package DFS-processed DBB tasks into the RDBPFN training H5 format."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, Callable, Mapping

import numpy as np
import yaml


_LOGGER = logging.getLogger(__name__)
_SUPPORTED_FEATURE_DTYPES = {"float", "category"}


@dataclass(frozen=True, slots=True, kw_only=True)
class H5ExportConfig:
    processed_root: Path
    output_path: Path
    total_rows: int = 600
    max_columns: int = 60
    seed: int = 42
    overwrite: bool = False
    dataset_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.processed_root, Path):
            raise TypeError("processed_root must be pathlib.Path")
        if not isinstance(self.output_path, Path):
            raise TypeError("output_path must be pathlib.Path")
        for name in ("total_rows", "max_columns", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.total_rows < 2:
            raise ValueError("total_rows must be at least two")
        if self.max_columns < 1:
            raise ValueError("max_columns must be positive")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        if not isinstance(self.dataset_names, tuple) or any(
            not isinstance(name, str) or not name for name in self.dataset_names
        ):
            raise TypeError("dataset_names must be a tuple of non-empty strings")


@dataclass(frozen=True, slots=True, kw_only=True)
class H5ExportResult:
    output_path: Path
    dataset_count: int
    task_count: int
    skipped_task_count: int


def run_rdbpfn_dfs(
    *,
    raw_root: Path,
    preprocessing_root: Path,
    depth: int,
    jobs: int,
    bash_command: str = "bash",
    progress_interval: float = 60.0,
) -> Path:
    """Run RDBPFN's batch DFS script and return its processed root.

    The batch script processes every dataset under ``raw_root`` with
    ``jobs`` parallel workers, which can take hours for large corpora.
    Its combined stdout/stderr is written to a log file next to
    ``raw_root`` and a heartbeat reporting the number of completed
    datasets is emitted every ``progress_interval`` seconds so the step
    remains observable under LOG_LEVEL=WARNING. On success the batch
    script's intermediate tree ``<raw_root>-tmp`` is removed; on failure
    it is kept for debugging.
    """
    if depth not in (1, 2):
        raise ValueError("DFS depth must be 1 or 2")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("DFS jobs must be a positive integer")
    raw_root = raw_root.resolve()
    preprocessing_root = preprocessing_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"RDBPFN raw dataset root does not exist: {raw_root}")
    script = preprocessing_root / f"benchmark_preprocess_depth{depth}.sh"
    if not script.is_file():
        raise FileNotFoundError(f"RDBPFN DFS script does not exist: {script}")

    processed_root = Path(f"{raw_root}-processed")
    if processed_root.exists():
        _LOGGER.warning(
            "RDBPFN processed root already exists; its batch script may reuse "
            "existing datasets: %s",
            processed_root,
        )
    total_datasets = _count_directories(raw_root)
    log_path = raw_root.parent / f"{raw_root.name}_dfs_depth{depth}.log"
    command = [bash_command, str(script), str(raw_root), str(jobs)]
    _LOGGER.warning(
        "starting RDBPFN DFS: depth=%d jobs=%d datasets=%s input=%s; "
        "subprocess output is written to %s (tail -f it for live progress)",
        depth,
        jobs,
        total_datasets if total_datasets >= 0 else "?",
        raw_root,
        log_path,
    )
    env = os.environ.copy()
    # RDBPFN preprocessing scripts use LOG_CFG / LOG_LEVEL for their own
    # logger; silence INFO and DEBUG so only warnings and errors surface.
    env["LOG_LEVEL"] = "WARNING"
    env["LOG_CFG"] = "WARNING"
    interval = progress_interval if progress_interval > 0 else 60.0
    stop = threading.Event()
    watcher = threading.Thread(
        target=_dfs_progress_watchdog,
        args=(processed_root, depth, total_datasets, interval, stop),
        daemon=True,
        name=f"dfs-depth{depth}-progress",
    )
    watcher.start()
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=preprocessing_root,
                check=False,
                env=env,
                # Never let child processes block waiting on the terminal
                # (e.g. GNU parallel's first-run citation prompt).
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
    finally:
        stop.set()
        watcher.join(timeout=5.0)
    if result.returncode != 0:
        _LOGGER.error(
            "RDBPFN DFS failed (exit %d); last log lines from %s:\n%s",
            result.returncode,
            log_path,
            _read_log_tail(log_path),
        )
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=None,
            stderr=None,
        )
    if not processed_root.is_dir():
        raise RuntimeError(
            f"RDBPFN DFS completed without creating {processed_root}"
        )
    _remove_dfs_tmp(raw_root)
    _LOGGER.warning("RDBPFN DFS complete: output=%s", processed_root)
    return processed_root


def _remove_dfs_tmp(raw_root: Path) -> None:
    """Remove the DFS batch script's intermediate tree, ``<raw_root>-tmp``.

    The batch script writes per-dataset ``-pre-dfs``/``-post-dfs`` staging
    directories there and never cleans them up. They are expendable once
    the run succeeds (the ``<raw_root>-processed`` outputs are what the H5
    packaging consumes), so removal is best effort: a failure logs a
    warning instead of failing the pipeline, and on a failed DFS run this
    is never called so the intermediates stay available for debugging.
    """
    tmp_root = Path(f"{raw_root}-tmp")
    if not tmp_root.is_dir():
        return
    try:
        entry_count = sum(1 for _ in tmp_root.iterdir())
    except OSError:
        entry_count = -1
    try:
        shutil.rmtree(tmp_root)
    except OSError as error:
        _LOGGER.warning(
            "could not remove DFS intermediate directory %s: %s",
            tmp_root,
            error,
        )
        return
    _LOGGER.warning(
        "removed DFS intermediate directory: %s (%s entries)",
        tmp_root,
        entry_count if entry_count >= 0 else "?",
    )


def _dfs_progress_watchdog(
    processed_root: Path,
    depth: int,
    total_datasets: int,
    interval: float,
    stop: threading.Event,
) -> None:
    suffix = f"-dfs-{depth}"
    while not stop.wait(interval):
        completed = -1
        if processed_root.is_dir():
            completed = 0
            try:
                completed = sum(
                    1
                    for path in processed_root.iterdir()
                    if path.is_dir() and path.name.endswith(suffix)
                )
            except OSError:
                completed = -1
        _LOGGER.warning(
            "RDBPFN DFS in progress: %s dataset(s) completed",
            f"{completed}/{total_datasets}"
            if completed >= 0 and total_datasets >= 0
            else (completed if completed >= 0 else "?"),
        )


def _count_directories(root: Path) -> int:
    try:
        return sum(1 for path in root.iterdir() if path.is_dir())
    except OSError:
        return -1


def _read_log_tail(log_path: Path, max_chars: int = 4000) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(log file unreadable)"
    tail = text.strip()[-max_chars:]
    return tail or "(empty log)"


def export_processed_dbb_to_h5(
    config: H5ExportConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> H5ExportResult:
    """Stream classification tasks from processed DBB datasets into one H5."""
    if not isinstance(config, H5ExportConfig):
        raise TypeError("config must be H5ExportConfig")
    dataset_paths = _discover_datasets(config.processed_root, config.dataset_names)
    descriptors = _task_descriptors(dataset_paths)
    if not descriptors:
        raise RuntimeError(
            f"no DBB tasks found under processed root {config.processed_root}"
        )
    # DBB directories and their task lists are discovered in deterministic
    # path order. The training loader streams H5 rows sequentially, so
    # writing that order directly creates long, correlated runs of schemas
    # and mechanisms. Shuffle the descriptor order once, deterministically,
    # with the export seed; task contents remain deterministic because each
    # task sample has its own derived RNG.
    descriptor_order = np.random.default_rng(config.seed).permutation(
        len(descriptors)
    )

    output_path = config.output_path.resolve()
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"RDBPFN H5 already exists: {output_path}; use overwrite=True"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    h5py = _require_h5py()
    accepted = 0
    skipped = 0
    _LOGGER.info(
        "starting RDBPFN H5 packaging: tasks=%d rows=%d columns=%d output=%s",
        len(descriptors),
        config.total_rows,
        config.max_columns,
        output_path,
    )
    try:
        with h5py.File(temporary, "w") as handle:
            datasets = _create_h5_datasets(
                handle,
                total_rows=config.total_rows,
                max_columns=config.max_columns,
            )
            handle.attrs["format"] = "rdbpfn-task-prior-v1"
            handle.attrs["source"] = str(config.processed_root.resolve())
            handle.attrs["task_order"] = "seeded_permutation"
            handle.attrs["task_order_seed"] = config.seed
            for completed, descriptor_index in enumerate(
                descriptor_order, start=1
            ):
                descriptor = descriptors[int(descriptor_index)]
                dataset_path, dataset_name, task_metadata = descriptor
                task_name = str(task_metadata.get("name", "unknown"))
                try:
                    sample = _prepare_task_sample(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        task_metadata=task_metadata,
                        total_rows=config.total_rows,
                        max_columns=config.max_columns,
                        seed=config.seed,
                    )
                except (OSError, TypeError, ValueError, KeyError) as error:
                    sample = None
                    _LOGGER.warning(
                        "skipping H5 task %s/%s: %s",
                        dataset_name,
                        task_name,
                        error,
                    )
                if sample is None:
                    skipped += 1
                else:
                    _append_sample(datasets, accepted, sample, config.max_columns)
                    accepted += 1
                if progress is not None:
                    progress(completed, len(descriptors), f"h5:{dataset_name}")
            if accepted == 0:
                raise RuntimeError("no usable classification tasks were found for H5")
            handle.attrs["task_count"] = accepted
            handle.attrs["skipped_task_count"] = skipped
            handle.flush()
        temporary.replace(output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    _LOGGER.info(
        "RDBPFN H5 packaging complete: tasks=%d skipped=%d output=%s",
        accepted,
        skipped,
        output_path,
    )
    return H5ExportResult(
        output_path=output_path,
        dataset_count=len(dataset_paths),
        task_count=accepted,
        skipped_task_count=skipped,
    )


def _discover_datasets(
    processed_root: Path,
    dataset_names: tuple[str, ...],
) -> tuple[Path, ...]:
    root = processed_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"processed DBB root does not exist: {root}")
    if dataset_names:
        paths = tuple(root / name for name in dataset_names)
        missing = [path for path in paths if not (path / "metadata.yaml").is_file()]
        if missing:
            rendered = ", ".join(str(path) for path in missing[:5])
            raise FileNotFoundError(f"processed DBB datasets are missing: {rendered}")
        return paths
    return tuple(
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "metadata.yaml").is_file()
    )


def _task_descriptors(
    dataset_paths: tuple[Path, ...],
) -> tuple[tuple[Path, str, Mapping[str, Any]], ...]:
    descriptors: list[tuple[Path, str, Mapping[str, Any]]] = []
    for dataset_path in dataset_paths:
        metadata = yaml.safe_load(
            (dataset_path / "metadata.yaml").read_text(encoding="utf-8")
        )
        if not isinstance(metadata, Mapping):
            raise ValueError(f"invalid DBB metadata: {dataset_path}")
        dataset_name = metadata.get("dataset_name", dataset_path.name)
        tasks = metadata.get("tasks")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError(f"invalid DBB dataset_name: {dataset_path}")
        if not isinstance(tasks, list):
            raise ValueError(f"invalid DBB task list: {dataset_path}")
        for task in tasks:
            if not isinstance(task, Mapping):
                raise ValueError(f"invalid DBB task metadata: {dataset_path}")
            descriptors.append((dataset_path, dataset_name, task))
    return tuple(descriptors)


def _prepare_task_sample(
    *,
    dataset_path: Path,
    dataset_name: str,
    task_metadata: Mapping[str, Any],
    total_rows: int,
    max_columns: int,
    seed: int,
) -> dict[str, Any] | None:
    if task_metadata.get("task_type") != "classification":
        return None
    task_name = _required_string(task_metadata, "name")
    target = _required_string(task_metadata, "target_column")
    columns = task_metadata.get("columns")
    if not isinstance(columns, list):
        raise ValueError("task columns must be a list")
    feature_metadata = [
        column
        for column in columns
        if isinstance(column, Mapping)
        and column.get("name") != target
        and column.get("dtype") in _SUPPORTED_FEATURE_DTYPES
    ]
    if not feature_metadata:
        return None

    rng = _task_rng(seed, dataset_name, task_name)
    available_features = len(feature_metadata)
    if available_features > max_columns:
        selected = np.sort(
            rng.choice(available_features, size=max_columns, replace=False)
        )
        feature_metadata = [feature_metadata[int(index)] for index in selected]
    feature_names = [_required_string(column, "name") for column in feature_metadata]

    train = _load_task_split(dataset_path, task_metadata, "train")
    validation = _load_optional_task_split(
        dataset_path,
        task_metadata,
        "validation",
    )
    test = _load_task_split(dataset_path, task_metadata, "test")
    support = _concatenate_splits(train, validation)
    encoders = _build_encoders(support, feature_metadata)
    support_x = _transform_split(support, feature_names, encoders)
    query_x = _transform_split(test, feature_names, encoders)
    support_y, query_y = _binarize_labels(
        _required_column(support, target),
        _required_column(test, target),
        rng,
    )
    if support_y is None or query_y is None:
        return None
    if support_x.shape[0] == 0 or query_x.shape[0] == 0:
        return None

    support_count = round(
        total_rows * support_x.shape[0] / (support_x.shape[0] + query_x.shape[0])
    )
    support_count = max(1, min(int(support_count), total_rows - 1))
    query_count = total_rows - support_count
    support_indices = _sample_binary_rows(support_y, support_count, rng)
    query_indices = _sample_binary_rows(query_y, query_count, rng)
    category_mask = np.asarray(
        [1 if encoders[name][0] == "categorical" else 0 for name in feature_names],
        dtype=np.uint8,
    )
    return {
        "X": np.concatenate(
            [support_x[support_indices], query_x[query_indices]], axis=0
        ).astype(np.float32, copy=False),
        "y": np.concatenate(
            [support_y[support_indices], query_y[query_indices]], axis=0
        ).astype(np.int32, copy=False),
        "num_features": len(feature_names),
        "num_available_features": available_features,
        "single_eval_pos": support_count,
        "category_mask": category_mask,
    }


def _load_task_split(
    dataset_path: Path,
    task_metadata: Mapping[str, Any],
    split: str,
) -> dict[str, np.ndarray]:
    source = _required_string(task_metadata, "source").replace("{split}", split)
    path = dataset_path / source
    if task_metadata.get("format") != "numpy" or path.suffix != ".npz":
        raise ValueError(f"unsupported DBB task source: {path}")
    with np.load(path, allow_pickle=True) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _load_optional_task_split(
    dataset_path: Path,
    task_metadata: Mapping[str, Any],
    split: str,
) -> dict[str, np.ndarray] | None:
    source = _required_string(task_metadata, "source").replace("{split}", split)
    if not (dataset_path / source).is_file():
        return None
    return _load_task_split(dataset_path, task_metadata, split)


def _concatenate_splits(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    if second is None:
        return {name: values.copy() for name, values in first.items()}
    if set(first) != set(second):
        raise ValueError("train and validation task columns do not match")
    return {
        name: np.concatenate([first[name], second[name]])
        for name in first
    }


def _build_encoders(
    support: Mapping[str, np.ndarray],
    feature_metadata: list[Mapping[str, Any]],
) -> dict[str, tuple[str, object]]:
    encoders: dict[str, tuple[str, object]] = {}
    for column in feature_metadata:
        name = _required_string(column, "name")
        values = _required_column(support, name)
        if values.size == 0:
            raise ValueError(f"feature {name} has no support rows")
        if column.get("dtype") == "float":
            numeric = values.astype(np.float32, copy=False)
            finite = numeric[np.isfinite(numeric)]
            fill_value = float(finite.mean()) if finite.size else 0.0
            encoders[name] = ("numeric", fill_value)
        else:
            unique = np.unique(values.astype(str))
            encoders[name] = (
                "categorical",
                {value: index for index, value in enumerate(unique)},
            )
    return encoders


def _transform_split(
    split: Mapping[str, np.ndarray],
    feature_names: list[str],
    encoders: Mapping[str, tuple[str, object]],
) -> np.ndarray:
    rows = len(_required_column(split, feature_names[0]))
    columns: list[np.ndarray] = []
    for name in feature_names:
        values = _required_column(split, name)
        if len(values) != rows:
            raise ValueError("task feature columns do not align")
        encoder_type, payload = encoders[name]
        if encoder_type == "numeric":
            encoded = values.astype(np.float32, copy=True)
            encoded[~np.isfinite(encoded)] = float(payload)
        else:
            mapping = payload
            assert isinstance(mapping, dict)
            unknown = len(mapping)
            encoded = np.asarray(
                [mapping.get(value, unknown) for value in values.astype(str)],
                dtype=np.float32,
            )
        columns.append(encoded.reshape(rows, 1))
    return np.concatenate(columns, axis=1)


def _binarize_labels(
    support: np.ndarray,
    query: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    support_values = support.astype(str)
    query_values = query.astype(str)
    unique = np.unique(np.concatenate([support_values, query_values]))
    if unique.size < 2:
        return None, None
    candidates: list[set[str]] = []
    if unique.size == 2:
        candidates.append({str(unique[1])})
    else:
        for _ in range(max(8, unique.size * 2)):
            shuffled = unique.copy()
            rng.shuffle(shuffled)
            candidates.append(
                {str(value) for value in shuffled[: max(1, unique.size // 2)]}
            )
    for positive in candidates:
        support_encoded = np.asarray(
            [1 if value in positive else 0 for value in support_values],
            dtype=np.int32,
        )
        query_encoded = np.asarray(
            [1 if value in positive else 0 for value in query_values],
            dtype=np.int32,
        )
        if np.unique(support_encoded).size == 2 and np.unique(query_encoded).size == 2:
            return support_encoded, query_encoded
    return None, None


def _sample_binary_rows(
    labels: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count < 1:
        raise ValueError("sample count must be positive")
    if count == 1:
        return rng.choice(len(labels), size=1, replace=len(labels) < 1)
    anchors = [
        int(rng.choice(np.flatnonzero(labels == label)))
        for label in (0, 1)
    ]
    remaining = rng.choice(
        len(labels),
        size=count - len(anchors),
        replace=len(labels) < count - len(anchors),
    )
    selected = np.concatenate(
        [np.asarray(anchors, dtype=np.int64), remaining.astype(np.int64)]
    )
    rng.shuffle(selected)
    return selected


def _create_h5_datasets(
    handle: Any,
    *,
    total_rows: int,
    max_columns: int,
) -> dict[str, Any]:
    datasets = {
        "X": handle.create_dataset(
            "X",
            shape=(0, total_rows, max_columns),
            maxshape=(None, total_rows, max_columns),
            chunks=(1, total_rows, max_columns),
            dtype="float32",
            compression="lzf",
        ),
        "y": handle.create_dataset(
            "y",
            shape=(0, total_rows),
            maxshape=(None, total_rows),
            chunks=(1, total_rows),
            dtype="int32",
            compression="lzf",
        ),
        "num_features": _vector_dataset(handle, "num_features", "int32"),
        "num_available_features": _vector_dataset(
            handle, "num_available_features", "int32"
        ),
        "num_datapoints": _vector_dataset(handle, "num_datapoints", "int32"),
        "single_eval_pos": _vector_dataset(handle, "single_eval_pos", "int32"),
        "feature_is_categorical": handle.create_dataset(
            "feature_is_categorical",
            shape=(0, max_columns),
            maxshape=(None, max_columns),
            chunks=(1, max_columns),
            dtype="uint8",
            compression="lzf",
        ),
    }
    handle.create_dataset("max_num_classes", data=np.asarray([1], dtype=np.int32))
    return datasets


def _vector_dataset(handle: Any, name: str, dtype: str) -> Any:
    return handle.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        chunks=(128,),
        dtype=dtype,
    )


def _append_sample(
    datasets: Mapping[str, Any],
    index: int,
    sample: Mapping[str, Any],
    max_columns: int,
) -> None:
    for dataset in datasets.values():
        dataset.resize(index + 1, axis=0)
    num_features = int(sample["num_features"])
    padded_x = np.zeros(
        (sample["X"].shape[0], max_columns),
        dtype=np.float32,
    )
    padded_x[:, :num_features] = sample["X"]
    category_mask = np.zeros(max_columns, dtype=np.uint8)
    category_mask[:num_features] = sample["category_mask"]
    datasets["X"][index] = padded_x
    datasets["y"][index] = sample["y"]
    datasets["num_features"][index] = num_features
    datasets["num_available_features"][index] = int(
        sample["num_available_features"]
    )
    datasets["num_datapoints"][index] = sample["X"].shape[0]
    datasets["single_eval_pos"][index] = int(sample["single_eval_pos"])
    datasets["feature_is_categorical"][index] = category_mask


def _task_rng(seed: int, dataset_name: str, task_name: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{seed}\0{dataset_name}\0{task_name}".encode("utf-8")
    ).digest()
    return np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:8], "big")))


def _required_string(mapping: Mapping[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_column(
    split: Mapping[str, np.ndarray],
    name: str,
) -> np.ndarray:
    if name not in split:
        raise KeyError(f"task split is missing column {name}")
    values = split[name]
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise ValueError(f"task column {name} must be one-dimensional")
    return values


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - declared dependency.
        raise RuntimeError("h5py is required for RDBPFN H5 export") from error
    return h5py


__all__ = [
    "H5ExportConfig",
    "H5ExportResult",
    "export_processed_dbb_to_h5",
    "run_rdbpfn_dfs",
]
