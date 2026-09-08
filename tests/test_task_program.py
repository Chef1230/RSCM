from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# Import generation first to preserve the package's historical import order.
from rdb_prior.generation.database import DatabaseGenerator  # noqa: F401
from rdb_prior.task.program import (
    CutoffPolicy,
    HorizonPolicy,
    LabelExpression,
    TaskProgramPlan,
)


class TaskProgramModelTests(unittest.TestCase):
    def test_generic_program_round_trip(self) -> None:
        program = TaskProgramPlan(
            program_id="program_001",
            family="history_gated_future_active",
            target_table_id="table_entity",
            expression=LabelExpression(
                kind="Exists",
                args=(
                    LabelExpression(kind="Event"),
                    {"window": "future"},
                ),
            ),
            cutoff_policy=CutoffPolicy(
                family="calendar_fraction",
                parameters=(("fraction", 0.5),),
            ),
            horizon_policy=HorizonPolicy(
                family="calendar_fraction",
                parameters=(("fraction", 0.25),),
            ),
            required_bundle_ids=("bundle_001",),
            seed=17,
        )
        restored = TaskProgramPlan.from_dict(program.to_dict())
        self.assertEqual(program, restored)

    def test_legacy_payload_is_upgraded_with_bundle_alias(self) -> None:
        program = TaskProgramPlan.from_dict(
            {
                "program_id": "legacy",
                "family": "entity_future_event_existence",
                "target_table_id": "entity",
                "source_table_id": "event",
                "foreign_key_id": "fk",
                "time_column_id": "time",
                "cutoff_time": 10,
                "horizon_end_time": 20,
                "required_mechanism_ids": ["bundle_legacy", "event_count"],
                "seed": 3,
                "prior_plan_id": "prior",
            }
        )
        self.assertEqual(("bundle_legacy",), program.required_bundle_ids)
        self.assertEqual(program, TaskProgramPlan.from_dict(program.to_dict()))


if __name__ == "__main__":
    unittest.main()
