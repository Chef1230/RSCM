from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.cli import main
from rdb_prior.config import (
    InstanceConfigOverrides,
    RDBPFNExportConfigOverrides,
    SchemaConfigError,
    SchemaConfigOverrides,
    TaskConfigOverrides,
    load_instance_pipeline_config,
    load_schema_pipeline_config,
    load_rdbpfn_export_config,
    load_task_pipeline_config,
)
from rdb_prior.instance.planner import InstancePlannerConfig
from rdb_prior.task.model import TaskMechanism


class SchemaConfigTests(unittest.TestCase):
    def test_template_exposes_random_column_mechanism(self) -> None:
        config = load_task_pipeline_config(
            PROJECT_ROOT / "configs" / "template.yaml"
        )
        weights = dict(config.planner.mechanism_weights)
        self.assertEqual(0.0, weights[TaskMechanism.RANDOM_COLUMN])

    def test_refactor_v2_loads_complete_pipeline(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "refactor_v2.yaml"
        schema = load_schema_pipeline_config(config_path)
        instance = load_instance_pipeline_config(config_path)
        task = load_task_pipeline_config(config_path)
        export = load_rdbpfn_export_config(config_path)

        run_root = (PROJECT_ROOT / "outputs" / "refactor_v2").resolve()
        self.assertEqual(20000, schema.num_schemas)
        self.assertEqual(run_root / "schema", schema.output_root)
        self.assertEqual(8, instance.num_workers)
        self.assertEqual(
            "lognormal",
            instance.planner.entity_rows_distribution,
        )
        self.assertEqual(
            0.70,
            instance.planner.feature_missing_zero_probability,
        )
        self.assertEqual(
            0.12,
            instance.planner.categorical_high_cardinality_probability,
        )
        self.assertEqual(
            0.1,
            instance.planner.categorical_dirichlet_alpha_min,
        )
        self.assertEqual(
            1.0,
            instance.planner.categorical_dirichlet_alpha_max,
        )
        self.assertEqual(
            0.5,
            instance.planner.categorical_signal_strength_min,
        )
        self.assertEqual(
            2.0,
            instance.planner.categorical_signal_strength_max,
        )
        role_scm = {
            role.value: prior for role, prior in instance.planner.role_scm
        }
        self.assertEqual(
            {"entity", "event", "lookup", "bridge", "detail"},
            set(role_scm),
        )
        self.assertEqual(
            0.25,
            {
                family.value: weight
                for family, weight in role_scm["event"].scm_weights
            }["mlp"],
        )
        self.assertEqual(1.20, role_scm["event"].noise_scale_multiplier)
        resolved_role_scm = instance.to_dict()["planner"]["role_scm"]
        self.assertEqual(
            0.80,
            resolved_role_scm["lookup"]["output_scale_multiplier"],
        )
        json.dumps(instance.to_dict())
        self.assertEqual(
            run_root / "schema" / "manifest.json",
            instance.schema_manifest,
        )
        self.assertEqual(run_root / "instance", instance.output_root)
        self.assertEqual(2, task.planner.tasks_per_database)
        self.assertEqual(
            run_root / "instance" / "manifest.json",
            task.instance_manifest,
        )
        self.assertEqual(run_root / "task", task.output_root)
        self.assertEqual(
            run_root / "task" / "manifest.json",
            export.task_manifest,
        )
        self.assertEqual(run_root / "rdbpfn", export.output_root)
        self.assertFalse(export.h5_enabled)
        self.assertTrue(export.h5_run_dfs)
        self.assertEqual(8, export.dfs_jobs)

    def test_reference_yaml_loads_all_schema_sections(self) -> None:
        config = load_schema_pipeline_config(
            PROJECT_ROOT / "configs" / "refactor_v1.yaml"
        )

        self.assertEqual(20, config.num_schemas)
        self.assertEqual(42, config.base_seed)
        self.assertTrue(config.graph.write_dot)
        self.assertIsNone(config.graph.render_format)
        self.assertEqual(tuple(range(3, 16)), config.sampler.table_count_values)
        self.assertEqual(7, len(config.sampler.motif_weights))
        self.assertEqual(1, config.sampler.min_motif_occurrences)
        self.assertEqual(4, config.sampler.max_motif_occurrences)
        self.assertEqual(
            0.35,
            config.sampler.background_attachment_probability,
        )
        self.assertEqual(
            4,
            len(config.compiler.feature_columns_by_table_count),
        )
        self.assertEqual((), config.compiler.feature_columns_by_role)
        self.assertEqual(
            (PROJECT_ROOT / "outputs" / "v1_sample" / "schema").resolve(),
            config.output_root,
        )
        instance = load_instance_pipeline_config(
            PROJECT_ROOT / "configs" / "refactor_v1.yaml"
        )
        task = load_task_pipeline_config(
            PROJECT_ROOT / "configs" / "refactor_v1.yaml"
        )
        export = load_rdbpfn_export_config(
            PROJECT_ROOT / "configs" / "refactor_v1.yaml"
        )
        run_root = (PROJECT_ROOT / "outputs" / "v1_sample").resolve()
        self.assertEqual(
            run_root / "schema" / "manifest.json",
            instance.schema_manifest,
        )
        self.assertEqual(run_root / "instance", instance.output_root)
        self.assertEqual(
            run_root / "instance" / "manifest.json",
            task.instance_manifest,
        )
        self.assertEqual(run_root / "task", task.output_root)
        self.assertEqual(
            run_root / "task" / "manifest.json",
            export.task_manifest,
        )
        self.assertEqual(run_root / "rdbpfn", export.output_root)

    def test_unknown_option_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.yaml"
            path.write_text(
                "config_version: 1\nschema:\n  unknown_knob: 3\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                SchemaConfigError,
                "unknown option",
            ):
                load_schema_pipeline_config(path)

    def test_unknown_role_scm_option_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_role_scm.yaml"
            path.write_text(
                "instance:\n"
                "  role_scm:\n"
                "    event:\n"
                "      noise_multiplier: 1.2\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                SchemaConfigError,
                r"config\.instance\.role_scm\.event contains unknown",
            ):
                load_instance_pipeline_config(path)

    def test_calendar_and_mechanism_keys_load_and_reject_unknown(self) -> None:
        instance = load_instance_pipeline_config(
            PROJECT_ROOT / "configs" / "refactor_v2.yaml"
        )
        planner = instance.planner
        self.assertEqual(1_577_836_800, planner.calendar_start_seconds_min)
        self.assertEqual(1_735_689_600, planner.calendar_start_seconds_max)
        self.assertLess(
            planner.calendar_span_seconds_min,
            planner.calendar_span_seconds_max,
        )
        mechanisms = {
            mechanism.value: weight
            for mechanism, weight in planner.event_temporal_mechanism_weights
        }
        self.assertEqual(
            {"stationary", "burst", "churn", "seasonal"},
            set(mechanisms),
        )
        self.assertAlmostEqual(1.0, sum(mechanisms.values()))
        self.assertEqual(3, planner.burst_max_clusters)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_calendar.yaml"
            path.write_text(
                "config_version: 1\ninstance:\n  calendar_frobnicate: 3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SchemaConfigError,
                "unknown option",
            ):
                load_instance_pipeline_config(path)

    def test_categorical_dirichlet_validation(self) -> None:
        with self.assertRaises(ValueError):
            InstancePlannerConfig(categorical_dirichlet_alpha_min=0.0)
        with self.assertRaises(ValueError):
            InstancePlannerConfig(
                categorical_dirichlet_alpha_min=1.0,
                categorical_dirichlet_alpha_max=0.5,
            )
        with self.assertRaises(ValueError):
            InstancePlannerConfig(categorical_signal_strength_min=0.0)
        with self.assertRaises(ValueError):
            InstancePlannerConfig(
                categorical_signal_strength_min=3.0,
                categorical_signal_strength_max=1.0,
            )

    def test_nested_config_resolves_paths_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory) / "project"
            config_directory = project_root / "configs" / "local"
            config_directory.mkdir(parents=True)
            path = config_directory / "local.yaml"
            path.write_text(
                "\n".join(
                    (
                        "config_version: 1",
                        "paths:",
                        "  output_root: outputs/v1",
                        "generation:",
                        "  num_schemas: 1",
                        "schema:",
                        "  min_tables: 3",
                        "  max_tables: 3",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_schema_pipeline_config(path)

            self.assertEqual(
                (project_root / "outputs" / "v1" / "schema").resolve(),
                config.output_root,
            )

    def test_unknown_motif_is_rejected_during_config_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad_motif.yaml"
            path.write_text(
                "config_version: 1\nmotifs:\n  weights:\n"
                "    imaginary_motif: 1.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                SchemaConfigError,
                "Unknown configured motif",
            ):
                load_schema_pipeline_config(path)

    def test_cli_bounds_override_disables_configured_distributions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "schemas"
            config = load_schema_pipeline_config(
                PROJECT_ROOT / "configs" / "refactor_v1.yaml",
                overrides=SchemaConfigOverrides(
                    output_root=output,
                    num_schemas=2,
                    min_tables=3,
                    max_tables=4,
                    min_feature_columns=1,
                    max_feature_columns=2,
                ),
            )

            self.assertEqual(2, config.num_schemas)
            self.assertEqual((), config.sampler.table_count_values)
            self.assertEqual(
                (),
                config.compiler.feature_columns_by_table_count,
            )
            self.assertEqual(1, config.compiler.min_feature_columns)
            self.assertEqual(2, config.compiler.max_feature_columns)

    def test_schema_cli_generates_from_yaml_with_run_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output"
            config_path = root / "schema.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "config_version: 1",
                        "seed: 7",
                        "generation:",
                        "  num_schemas: 9",
                        "  progress_every: 0",
                        "schema:",
                        "  min_tables: 3",
                        "  max_tables: 3",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    (
                        "schema",
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(output),
                        "--count",
                        "2",
                    )
                )

            self.assertEqual(0, exit_code)
            summary = json.loads(stdout.getvalue().splitlines()[-1])
            self.assertEqual(2, summary["generated_count"])
            self.assertEqual(2, summary["dot_count"])
            self.assertEqual(0, summary["image_count"])
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, len(manifest["entries"]))
            self.assertTrue(
                all(entry["table_count"] == 3 for entry in manifest["entries"])
            )

    def test_instance_worker_config_and_cli_override(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "refactor_v1.yaml"
        config = load_instance_pipeline_config(
            config_path,
            overrides=InstanceConfigOverrides(num_workers=3),
        )
        self.assertEqual(3, config.num_workers)

        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                (
                    "instance",
                    "--config",
                    str(config_path),
                    "--jobs",
                    "2",
                    "--validate-config-only",
                )
            )
        self.assertEqual(0, exit_code)
        resolved = json.loads(stdout.getvalue())
        self.assertEqual(2, resolved["num_workers"])

    def test_task_config_and_cli_override_tasks_per_database(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "refactor_v1.yaml"
        config = load_task_pipeline_config(
            config_path,
            overrides=TaskConfigOverrides(
                tasks_per_database=4,
                num_workers=3,
            ),
        )
        self.assertEqual(4, config.planner.tasks_per_database)
        self.assertEqual(3, config.num_workers)
        self.assertEqual(8, len(config.planner.mechanism_weights))

        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                (
                    "task",
                    "--config",
                    str(config_path),
                    "--tasks-per-database",
                    "5",
                    "--jobs",
                    "4",
                    "--validate-config-only",
                )
            )
        self.assertEqual(0, exit_code)
        resolved = json.loads(stdout.getvalue())
        self.assertEqual(5, resolved["planner"]["tasks_per_database"])
        self.assertEqual(4, resolved["num_workers"])

    def test_rdbpfn_export_config_and_cli_overrides(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "refactor_v1.yaml"
        config = load_rdbpfn_export_config(
            config_path,
            overrides=RDBPFNExportConfigOverrides(
                task_count=7,
                validation_fraction=0.25,
                compress=False,
                h5_enabled=True,
                h5_run_dfs=False,
                h5_total_rows=512,
            ),
        )
        self.assertEqual(7, config.task_count)
        self.assertEqual(0.25, config.validation_fraction)
        self.assertFalse(config.compress)
        self.assertTrue(config.h5_enabled)
        self.assertFalse(config.h5_run_dfs)
        self.assertEqual(512, config.h5_total_rows)
        self.assertEqual(
            (PROJECT_ROOT.parent / "RDBPFN" / "data_preprocessing").resolve(),
            config.rdbpfn_preprocessing_root,
        )

        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                (
                    "rdbpfn-export",
                    "--config",
                    str(config_path),
                    "--count",
                    "3",
                    "--no-compress",
                    "--h5",
                    "--no-h5-run-dfs",
                    "--h5-output",
                    str(PROJECT_ROOT / "outputs" / "test_prior.h5"),
                    "--dfs-depth",
                    "2",
                    "--validate-config-only",
                )
            )
        self.assertEqual(0, exit_code)
        resolved = json.loads(stdout.getvalue())
        self.assertEqual(3, resolved["task_count"])
        self.assertFalse(resolved["compress"])
        self.assertTrue(resolved["h5_enabled"])
        self.assertFalse(resolved["h5_run_dfs"])
        self.assertEqual(2, resolved["dfs_depth"])
        self.assertEqual(
            str((PROJECT_ROOT / "outputs" / "test_prior.h5").resolve()),
            resolved["h5_output"],
        )


class TemplateFallbackTests(unittest.TestCase):
    """Configs deep-merge over configs/template.yaml as fallback defaults."""

    def _write_minimal_config(self, directory: Path) -> Path:
        config_path = directory / "minimal.yaml"
        config_path.write_text(
            "paths:\n"
            "  output_root: outputs/merge_probe\n"
            "schema:\n"
            "  max_tables: 9\n",
            encoding="utf-8",
        )
        return config_path

    def test_explicit_field_wins_and_missing_fields_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = load_schema_pipeline_config(
                self._write_minimal_config(Path(temporary_directory))
            )

        self.assertEqual(9, config.sampler.max_tables)
        # Missing fields and whole missing sections come from the template.
        template = load_schema_pipeline_config(
            PROJECT_ROOT / "configs" / "template.yaml"
        )
        self.assertEqual(
            template.sampler.min_tables, config.sampler.min_tables
        )
        self.assertEqual(template.num_schemas, config.num_schemas)
        self.assertEqual(
            template.sampler.motif_weights, config.sampler.motif_weights
        )

    def test_merging_can_be_disabled_via_environment(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = self._write_minimal_config(Path(temporary_directory))
            os.environ["RDB_PRIOR_CONFIG_TEMPLATE"] = ""
            try:
                config = load_schema_pipeline_config(config_path)
            finally:
                del os.environ["RDB_PRIOR_CONFIG_TEMPLATE"]

        # Without the template, loader-level code defaults apply instead.
        self.assertEqual(9, config.sampler.max_tables)
        self.assertEqual(100, config.num_schemas)
        self.assertEqual(
            (("entity_event", 1.0),),
            tuple(config.sampler.motif_weights[:1]),
        )

    def test_complete_presets_are_unaffected_by_merge(self) -> None:
        import dataclasses
        import os

        preset = PROJECT_ROOT / "configs" / "refactor_v2.yaml"
        merged = load_schema_pipeline_config(preset)
        os.environ["RDB_PRIOR_CONFIG_TEMPLATE"] = ""
        try:
            bare = load_schema_pipeline_config(preset)
        finally:
            del os.environ["RDB_PRIOR_CONFIG_TEMPLATE"]

        self.assertEqual(
            dataclasses.asdict(bare), dataclasses.asdict(merged)
        )

    def test_yaml_bounds_override_resets_inherited_table_count_prior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "small.yaml"
            config_path.write_text(
                "schema:\n  max_tables: 8\n",
                encoding="utf-8",
            )
            config = load_schema_pipeline_config(config_path)

        # Bounds were overridden without a custom list, so the template's
        # fixed table-count prior is dropped and a uniform draw is used.
        self.assertEqual(8, config.sampler.max_tables)
        self.assertEqual((), config.sampler.table_count_values)
        self.assertEqual((), config.sampler.table_count_weights)

    def test_null_feature_columns_by_role_clears_template_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "noroles.yaml"
            config_path.write_text(
                "physical_design:\n  feature_columns_by_role: null\n",
                encoding="utf-8",
            )
            config = load_schema_pipeline_config(config_path)

        self.assertEqual((), config.compiler.feature_columns_by_role)


if __name__ == "__main__":
    unittest.main()
