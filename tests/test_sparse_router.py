from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.instance.planner import InstancePlannerConfig
from rdb_prior.config import (
    load_routed_h5_config,
    load_router_training_config,
)
from rdb_prior.pipeline import (
    InstancePipelineConfig,
    SchemaPipelineConfig,
    generate_database_instances,
    generate_physical_schemas,
)
from rdb_prior.routing.catalog import enumerate_schema_paths
from rdb_prior.routing.config import (
    RoutedH5Config,
    RouterEvaluationConfig,
    RouterModelConfig,
    RouterTrainingConfig,
)
from rdb_prior.routing.data import (
    RoutingTaskTensorizer,
    collate_routed_tasks,
    load_routing_tasks,
)
from rdb_prior.routing.losses import sparse_router_batch_loss, sparse_router_loss
from rdb_prior.routing.checkpoint import load_router_checkpoint
from rdb_prior.routing.network import SparseRelationalPFN
from rdb_prior.routing.trainer import train_sparse_router
from rdb_prior.evaluation.router import evaluate_router_checkpoint
from rdb_prior.export.routed_h5 import export_routed_h5
from rdb_prior.schema.sampler import BlueprintSamplerConfig
from rdb_prior.task.model import RouteRole, TaskMechanism
from rdb_prior.task.pipeline import TaskPipelineConfig, generate_tasks
from rdb_prior.task.planner import TaskPlannerConfig


class SparseRouterTests(unittest.TestCase):
    def _task(self, root: Path):
        schemas = generate_physical_schemas(
            SchemaPipelineConfig(
                output_root=root / "schema",
                num_schemas=1,
                base_seed=717,
                sampler=BlueprintSamplerConfig(min_tables=4, max_tables=4),
            )
        )
        instances = generate_database_instances(
            InstancePipelineConfig(
                schema_manifest=schemas.manifest_path,
                output_root=root / "instance",
                planner=InstancePlannerConfig(
                    entity_rows_min=24,
                    entity_rows_max=32,
                    lookup_rows_min=3,
                    lookup_rows_max=5,
                    max_rows_per_table=96,
                ),
            )
        )
        tasks = generate_tasks(
            TaskPipelineConfig(
                instance_manifest=instances.manifest_path,
                output_root=root / "task",
                planner=TaskPlannerConfig(
                    tasks_per_database=1,
                    mechanism_weights=(
                        (TaskMechanism.RELATION_ATTRIBUTE, 1.0),
                    ),
                    min_support_rows=8,
                    min_query_rows=4,
                ),
            )
        )
        return load_routing_tasks(tasks.manifest_path)[0]

    def test_refactor_v2_fixes_first_version_sparse_bounds(self) -> None:
        training = load_router_training_config(
            PROJECT_ROOT / "configs" / "refactor_v2.yaml"
        )
        self.assertEqual(2, training.model.max_path_depth)
        self.assertEqual(20, training.model.max_candidates)
        self.assertEqual(3, training.model.top_k_paths)
        self.assertEqual(8, training.model.max_source_columns)
        self.assertEqual(16, training.model.min_rows_per_hop)
        self.assertEqual(32, training.model.rows_per_hop)
        self.assertEqual(600, training.model.max_rows_per_task)
        self.assertEqual(8, training.batch_size)
        self.assertEqual(8, training.num_workers)
        self.assertEqual(2, training.prefetch_factor)
        self.assertEqual("bf16", training.mixed_precision)
        routed = load_routed_h5_config(
            PROJECT_ROOT / "configs" / "refactor_v2.yaml"
        )
        self.assertEqual("best.pt", routed.checkpoint.name)

    def test_catalog_is_bounded_and_task_dsl_labels_adjacent_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = self._task(Path(directory))
            paths = enumerate_schema_paths(
                raw.schema,
                target_table_id=raw.task_artifact.task.plan.target_table_id,
                max_depth=2,
                max_candidates=20,
            )
            self.assertLessEqual(len(paths), 20)
            self.assertTrue(all(1 <= len(path.hops) <= 2 for path in paths))
            self.assertTrue(raw.task_artifact.task.plan.route_supervision)
            roles = {
                label.role
                for label in raw.task_artifact.task.plan.route_supervision
            }
            # Relation-attribute tasks have one required path; optional paths
            # are reserved for mechanisms with an explicitly optional
            # relation, such as the item side of interaction_response.
            self.assertNotIn(RouteRole.OPTIONAL, roles)
            self.assertIn(RouteRole.DISTRACTOR, roles)

    def test_end_to_end_loss_backward_and_query_label_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = self._task(Path(directory))
            model_config = RouterModelConfig(
                max_path_depth=2,
                max_candidates=8,
                top_k_paths=2,
                max_source_columns=3,
                min_rows_per_hop=2,
                rows_per_hop=4,
                max_target_columns=8,
                token_dim=16,
                type_embedding_dim=4,
                router_hidden_dim=24,
                transformer_heads=4,
                transformer_layers=1,
                dropout=0.0,
            )
            tensorizer = RoutingTaskTensorizer(model_config)
            descriptors = tensorizer.tensorize_descriptors(raw)
            self.assertFalse(bool(descriptors.source_row_mask.any()))
            model = SparseRelationalPFN(model_config)
            model.eval()
            selection = model.select(descriptors)
            batch = tensorizer.materialize_selected(
                raw,
                selected_path_mask=selection.route_selection.hard_mask,
                selected_column_mask=selection.column_hard_mask,
            )
            unselected_paths = ~selection.route_selection.hard_mask
            self.assertFalse(
                bool(batch.source_row_mask[:, unselected_paths].any())
            )
            unselected_columns = ~selection.column_hard_mask
            expanded_column_mask = unselected_columns[None, :, :, None].expand_as(
                batch.source_row_mask
            )
            self.assertFalse(
                bool(batch.source_row_mask[expanded_column_mask].any())
            )
            output = model(batch, selection)
            self.assertLessEqual(output.relation_token_count, 2 * 3)
            training = RouterTrainingConfig(
                task_manifest=Path("unused.json"),
                output_root=Path("unused"),
                model=model_config,
                epochs=1,
            )
            losses = sparse_router_loss(output, batch, training)
            self.assertTrue(torch.isfinite(losses.total))
            losses.total.backward()
            scorer_gradient = next(model.path_router.scorer.parameters()).grad
            self.assertIsNotNone(scorer_gradient)
            self.assertTrue(torch.isfinite(scorer_gradient).all())

            changed = batch.labels.clone()
            changed[batch.query_mask] += 100
            isolated_batch = replace(batch, labels=changed)
            isolated_descriptors = replace(descriptors, labels=changed)
            with torch.no_grad():
                isolated_selection = model.select(isolated_descriptors)
                isolated = model(isolated_batch, isolated_selection)
            torch.testing.assert_close(
                output.classification_logits,
                isolated.classification_logits,
            )
            torch.testing.assert_close(
                output.regression_prediction,
                isolated.regression_prediction,
            )

    def test_one_epoch_training_writes_loadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self._task(root)
            manifest_path = root / "task" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"] = manifest["entries"] * 2
            manifest["task_count"] = 2
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            model_config = RouterModelConfig(
                max_candidates=6,
                top_k_paths=2,
                max_source_columns=2,
                min_rows_per_hop=2,
                rows_per_hop=4,
                max_target_columns=6,
                token_dim=16,
                type_embedding_dim=4,
                router_hidden_dim=24,
                transformer_heads=4,
                transformer_layers=1,
                dropout=0.0,
            )
            with self.assertLogs(
                "rdb_prior.routing.trainer", level="INFO"
            ) as captured:
                progress_updates = []
                result = train_sparse_router(
                    RouterTrainingConfig(
                        task_manifest=root / "task" / "manifest.json",
                        output_root=root / "router",
                        model=model_config,
                        epochs=1,
                        validation_fraction=0.0,
                        device="cpu",
                        batch_size=2,
                        num_workers=2,
                        prefetch_factor=2,
                    ),
                    progress=lambda completed, total, task_id, detail: (
                        progress_updates.append(
                            (completed, total, task_id, detail)
                        )
                    ),
                )
            self.assertIn("loss=", progress_updates[-1][3])
            self.assertIn("running_loss=", progress_updates[-1][3])
            self.assertTrue(
                any(
                    "router eval epoch=1 batch=1/1" in message
                    and "loss=" in message
                    and "running_loss=" in message
                    for message in captured.output
                )
            )
            self.assertTrue(result.best_checkpoint.is_file())
            evaluation = evaluate_router_checkpoint(
                RouterEvaluationConfig(
                    task_manifest=manifest_path,
                    checkpoint=result.best_checkpoint,
                    output_root=root / "evaluation",
                    device="cpu",
                )
            )
            self.assertEqual(2, evaluation.task_count)
            self.assertGreater(evaluation.query_row_count, 0)
            evaluation_payload = json.loads(
                evaluation.metrics_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                "rdb-prior-router-benchmark-eval-v1",
                evaluation_payload["format"],
            )
            predictions = evaluation.predictions_path.read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(evaluation.query_row_count, len(predictions))
            self.assertIn("row_id", json.loads(predictions[0]))

            routed_path = root / "routed" / "benchmark.h5"
            export_routed_h5(
                RoutedH5Config(
                    task_manifest=manifest_path,
                    checkpoint=result.best_checkpoint,
                    output_path=routed_path,
                    task_count=1,
                    device="cpu",
                )
            )
            with h5py.File(routed_path, "r") as handle:
                group = next(iter(handle["tasks"].values()))
                self.assertIn("row_ids", group)
                self.assertIn("class_values", group.attrs)
            loaded, payload = load_router_checkpoint(result.best_checkpoint)
            self.assertEqual(1, payload["epoch"])
            batch = RoutingTaskTensorizer(model_config).tensorize(raw)
            with torch.no_grad():
                output = loaded.eval()(batch)
            self.assertEqual(len(batch.labels), len(output.regression_prediction))

    def test_padded_two_task_batch_forward_and_backward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = self._task(Path(directory))
            model_config = RouterModelConfig(
                max_candidates=6,
                top_k_paths=2,
                max_source_columns=2,
                min_rows_per_hop=2,
                rows_per_hop=4,
                max_target_columns=6,
                token_dim=16,
                type_embedding_dim=4,
                router_hidden_dim=24,
                transformer_heads=4,
                transformer_layers=1,
                dropout=0.0,
            )
            tensorizer = RoutingTaskTensorizer(model_config)
            descriptors = tuple(
                tensorizer.tensorize_descriptors(raw) for _ in range(2)
            )
            self.assertEqual(0, descriptors[0].source_values.shape[-1])
            descriptor_batch = collate_routed_tasks(descriptors)
            model = SparseRelationalPFN(model_config)
            selection = model.select(descriptor_batch)
            materialized = tuple(
                tensorizer.materialize_selected(
                    raw,
                    selected_path_mask=selection.route_selection.hard_mask[index],
                    selected_column_mask=selection.column_hard_mask[index],
                )
                for index in range(2)
            )
            batch = collate_routed_tasks(materialized)
            output = model(batch, selection)
            training = RouterTrainingConfig(
                task_manifest=Path("unused.json"),
                output_root=Path("unused"),
                model=model_config,
                epochs=1,
                batch_size=2,
            )
            losses = sparse_router_batch_loss(output, batch, training)
            losses.total.backward()

            self.assertEqual(2, output.classification_logits.shape[0])
            self.assertTrue(torch.isfinite(output.classification_logits).all())
            self.assertTrue(torch.isfinite(losses.total))
            self.assertIsNotNone(
                next(model.path_router.scorer.parameters()).grad
            )


if __name__ == "__main__":
    unittest.main()
