from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.export.h5 import (
    H5ExportConfig,
    _remove_dfs_tmp,
    export_processed_dbb_to_h5,
    run_rdbpfn_dfs,
)


def _find_working_bash() -> str | None:
    """Locate a bash that can actually run POSIX scripts.

    shutil.which("bash") is not enough on Windows: PATH may resolve to
    the WSL launcher stub, which fails on Windows-style paths. Probe each
    candidate once and keep the first one that executes ``echo ok``.
    """
    candidates: list[str] = []
    env_bash = os.environ.get("BASH")
    if env_bash:
        candidates.append(env_bash)
    seen: set[str] = set()
    for directory in os.environ.get("PATH", os.defpath).split(os.pathsep):
        if not directory:
            continue
        for name in ("bash.exe", "bash"):
            candidate = str(Path(directory, name))
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    for candidate in candidates:
        try:
            probe = subprocess.run(
                [candidate, "-c", "echo ok"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0 and probe.stdout.strip() == "ok":
            return candidate
    return None


TEST_BASH = _find_working_bash()


class H5ExportTests(unittest.TestCase):
    def test_task_rows_are_shuffled_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            processed = root / "rdbpfn-processed"
            dataset_names = (
                "classification-a",
                "classification-b",
                "classification-c",
            )
            for name in dataset_names:
                self._write_dataset(processed / name, task_type="classification")

            def export(seed: int, filename: str) -> tuple[list[str], list[str]]:
                progress: list[str] = []
                output = root / filename
                export_processed_dbb_to_h5(
                    H5ExportConfig(
                        processed_root=processed,
                        output_path=output,
                        total_rows=12,
                        max_columns=4,
                        seed=seed,
                        dataset_names=dataset_names,
                    ),
                    progress=lambda _completed, _total, name: progress.append(name),
                )
                with h5py.File(output, "r") as handle:
                    attrs = [
                        str(handle.attrs["task_order"]),
                        str(int(handle.attrs["task_order_seed"])),
                    ]
                return progress, attrs

            first, first_attrs = export(0, "prior-0-a.h5")
            repeat, repeat_attrs = export(0, "prior-0-b.h5")
            changed, changed_attrs = export(1, "prior-1.h5")

            self.assertEqual(
                ["h5:classification-c", "h5:classification-a", "h5:classification-b"],
                first,
            )
            self.assertEqual(first, repeat)
            self.assertNotEqual(first, changed)
            self.assertEqual(["seeded_permutation", "0"], first_attrs)
            self.assertEqual(first_attrs, repeat_attrs)
            self.assertEqual(["seeded_permutation", "1"], changed_attrs)
    def test_streams_processed_classification_tasks_in_training_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            processed = root / "rdbpfn-processed"
            self._write_dataset(
                processed / "classification-dfs-1",
                task_type="classification",
            )
            self._write_dataset(
                processed / "regression-dfs-1",
                task_type="regression",
            )
            output = root / "prior.h5"
            progress: list[tuple[int, int, str]] = []

            result = export_processed_dbb_to_h5(
                H5ExportConfig(
                    processed_root=processed,
                    output_path=output,
                    total_rows=12,
                    max_columns=4,
                    seed=17,
                    dataset_names=(
                        "classification-dfs-1",
                        "regression-dfs-1",
                    ),
                ),
                progress=lambda completed, total, name: progress.append(
                    (completed, total, name)
                ),
            )

            self.assertEqual(2, result.dataset_count)
            self.assertEqual(1, result.task_count)
            self.assertEqual(1, result.skipped_task_count)
            self.assertEqual((2, 2), progress[-1][:2])
            with h5py.File(output, "r") as handle:
                self.assertEqual(
                    {
                        "X",
                        "y",
                        "num_features",
                        "num_available_features",
                        "num_datapoints",
                        "single_eval_pos",
                        "feature_is_categorical",
                        "max_num_classes",
                    },
                    set(handle.keys()),
                )
                self.assertEqual((1, 12, 4), handle["X"].shape)
                self.assertEqual((1, 12), handle["y"].shape)
                self.assertEqual(2, int(handle["num_features"][0]))
                self.assertEqual(2, int(handle["num_available_features"][0]))
                self.assertEqual(12, int(handle["num_datapoints"][0]))
                split = int(handle["single_eval_pos"][0])
                self.assertEqual(7, split)
                np.testing.assert_array_equal(
                    np.asarray([0, 1, 0, 0], dtype=np.uint8),
                    handle["feature_is_categorical"][0],
                )
                self.assertEqual({0, 1}, set(handle["y"][0, :split].tolist()))
                self.assertEqual({0, 1}, set(handle["y"][0, split:].tolist()))
                np.testing.assert_array_equal(
                    np.asarray([1], dtype=np.int32),
                    handle["max_num_classes"][:],
                )

            with self.assertRaises(FileExistsError):
                export_processed_dbb_to_h5(
                    H5ExportConfig(
                        processed_root=processed,
                        output_path=output,
                        total_rows=12,
                        max_columns=4,
                        dataset_names=("classification-dfs-1",),
                    )
                )

    @staticmethod
    def _write_dataset(path: Path, *, task_type: str) -> None:
        task_name = path.name
        task_directory = path / task_name
        task_directory.mkdir(parents=True)
        columns = [
            {"name": "feature_float", "dtype": "float", "in_size": 1},
            {
                "name": "feature_category",
                "dtype": "category",
                "num_categories": 3,
            },
            {
                "name": "label",
                "dtype": "category" if task_type == "classification" else "float",
                "num_categories": 2,
            },
        ]
        metadata = {
            "dataset_name": task_name,
            "tables": [],
            "tasks": [
                {
                    "name": task_name,
                    "source": f"{task_name}/{{split}}.npz",
                    "format": "numpy",
                    "columns": columns,
                    "time_column": None,
                    "evaluation_metric": "auroc",
                    "target_column": "label",
                    "target_table": "target",
                    "task_type": task_type,
                    "num_classes": 2,
                }
            ],
        }
        (path / "metadata.yaml").write_text(
            yaml.safe_dump(metadata, sort_keys=False),
            encoding="utf-8",
        )
        splits = {
            "train": (
                np.asarray([0.0, 1.0, np.nan, 3.0], dtype=np.float32),
                np.asarray(["a", "b", "a", "c"], dtype=object),
                np.asarray([0, 1, 0, 1]),
            ),
            "validation": (
                np.asarray([4.0, 5.0], dtype=np.float32),
                np.asarray(["b", "c"], dtype=object),
                np.asarray([0, 1]),
            ),
            "test": (
                np.asarray([6.0, 7.0, 8.0, 9.0], dtype=np.float32),
                np.asarray(["a", "unseen", "b", "c"], dtype=object),
                np.asarray([0, 1, 0, 1]),
            ),
        }
        for split_name, (numeric, category, labels) in splits.items():
            np.savez(
                task_directory / f"{split_name}.npz",
                feature_float=numeric,
                feature_category=category,
                label=labels.astype(
                    np.int64 if task_type == "classification" else np.float32
                ),
            )


@unittest.skipUnless(
    TEST_BASH, "a working POSIX bash is required to run the DFS batch script"
)
class RunRdbpfnDfsTests(unittest.TestCase):
    def test_removes_tmp_tree_after_successful_dfs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "rdbpfn"
            (raw_root / "task_sample_000000_000").mkdir(parents=True)
            stale_tmp = raw_root.parent / "rdbpfn-tmp" / "stale-pre-dfs"
            stale_tmp.mkdir(parents=True)
            preprocessing = root / "preprocessing"
            preprocessing.mkdir()
            script = preprocessing / "benchmark_preprocess_depth1.sh"
            script.write_text(
                "#!/bin/sh\n"
                'mkdir -p "$1-processed/probe-dfs-1"\n'
                'mkdir -p "$1-tmp/probe-pre-dfs"\n',
                encoding="utf-8",
            )

            processed = run_rdbpfn_dfs(
                raw_root=raw_root,
                preprocessing_root=preprocessing,
                depth=1,
                jobs=2,
                bash_command=TEST_BASH,
                progress_interval=3600.0,
            )

            self.assertEqual(Path(f"{raw_root}-processed"), processed)
            self.assertTrue((processed / "probe-dfs-1").is_dir())
            self.assertFalse(Path(f"{raw_root}-tmp").exists())
            self.assertTrue(
                (raw_root.parent / "rdbpfn_dfs_depth1.log").is_file()
            )

    def test_keeps_tmp_tree_when_dfs_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "rdbpfn"
            raw_root.mkdir()
            tmp_root = Path(f"{raw_root}-tmp")
            (tmp_root / "probe-pre-dfs").mkdir(parents=True)
            preprocessing = root / "preprocessing"
            preprocessing.mkdir()
            script = preprocessing / "benchmark_preprocess_depth1.sh"
            script.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")

            with self.assertRaises(subprocess.CalledProcessError) as caught:
                run_rdbpfn_dfs(
                    raw_root=raw_root,
                    preprocessing_root=preprocessing,
                    depth=1,
                    jobs=2,
                    bash_command=TEST_BASH,
                    progress_interval=3600.0,
                )

            self.assertEqual(3, caught.exception.returncode)
            self.assertTrue(tmp_root.is_dir())
            self.assertFalse(Path(f"{raw_root}-processed").exists())


class RemoveDfsTmpTests(unittest.TestCase):
    def test_removes_existing_tmp_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_root = Path(temporary_directory) / "rdbpfn"
            tmp_root = Path(f"{raw_root}-tmp")
            (tmp_root / "task_sample_000000_000-pre-dfs").mkdir(parents=True)
            (tmp_root / "task_sample_000000_000-post-dfs").mkdir()

            _remove_dfs_tmp(raw_root)

            self.assertFalse(tmp_root.exists())

    def test_noop_when_tmp_tree_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_root = Path(temporary_directory) / "rdbpfn"

            _remove_dfs_tmp(raw_root)

            self.assertFalse(Path(f"{raw_root}-tmp").exists())


if __name__ == "__main__":
    unittest.main()
