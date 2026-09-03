"""Role-aware binding of physical schemas into executable InstancePlans."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

from rdb_prior.compilation.model import PhysicalForeignKey, PhysicalSchema
from rdb_prior.instance.plan import (
    EventTemporalMechanism,
    FeatureSCMFamily,
    InstancePlan,
    PopulationPlan,
    RelationMechanismPlan,
    RootCauseFamily,
    TableMechanismPlan,
    TemporalFamily,
)
from rdb_prior.instance.scm_prior import (
    sample_scm_meta_parameters,
    sample_table_scm_parameters,
)
from rdb_prior.priors.families.legacy_role_scm import bind_legacy_plan
from rdb_prior.priors.families.temporal_event import bind_temporal_event_plan
from rdb_prior.priors.model import DatabasePriorPlan, PriorFamily
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.spec import Optionality, TableRole


_DEFAULT_SCM_WEIGHTS = (
    (FeatureSCMFamily.EXOGENOUS, 0.30),
    (FeatureSCMFamily.LINEAR, 0.40),
    (FeatureSCMFamily.CAM, 0.20),
    (FeatureSCMFamily.MLP, 0.10),
)

_DEFAULT_ROOT_CAUSE_WEIGHTS = (
    (RootCauseFamily.STANDARD_NORMAL, 0.30),
    (RootCauseFamily.LINEAR, 0.20),
    (RootCauseFamily.NONLINEAR, 0.20),
    (RootCauseFamily.LOGNORMAL, 0.15),
    (RootCauseFamily.GAUSSIAN_MIXTURE, 0.15),
)

_LOOKUP_SCM_WEIGHTS = ((FeatureSCMFamily.EXOGENOUS, 1.0),)
_LOOKUP_ROOT_CAUSE_WEIGHTS = ((RootCauseFamily.STANDARD_NORMAL, 1.0),)


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleSCMPrior:
    """Table-role-specific SCM families and scale adjustments."""

    scm_weights: tuple[tuple[FeatureSCMFamily, float], ...] = (
        _DEFAULT_SCM_WEIGHTS
    )
    root_cause_weights: tuple[tuple[RootCauseFamily, float], ...] = (
        _DEFAULT_ROOT_CAUSE_WEIGHTS
    )
    signal_scale_multiplier: float = 1.0
    noise_scale_multiplier: float = 1.0
    activation_scale_multiplier: float = 1.0
    output_scale_multiplier: float = 1.0

    def __post_init__(self) -> None:
        _validate_scm_weights(self.scm_weights, "scm_weights")
        _validate_root_cause_weights(
            self.root_cause_weights,
            "root_cause_weights",
        )
        for name in (
            "signal_scale_multiplier",
            "noise_scale_multiplier",
            "activation_scale_multiplier",
            "output_scale_multiplier",
        ):
            _positive_scalar(name, getattr(self, name))


@dataclass(frozen=True, slots=True, kw_only=True)
class InstancePlannerConfig:
    entity_rows_min: int = 128
    entity_rows_max: int = 512
    entity_rows_distribution: str = "loguniform"
    entity_rows_median: int = 256
    entity_rows_log_sigma: float = 0.8
    lookup_rows_min: int = 4
    lookup_rows_max: int = 32
    lookup_rows_distribution: str = "uniform"
    event_fanout_min: float = 0.75
    event_fanout_max: float = 4.0
    bridge_rows_factor_min: float = 0.5
    bridge_rows_factor_max: float = 2.0
    detail_fanout_min: float = 0.75
    detail_fanout_max: float = 4.0
    entity_child_factor_min: float = 0.5
    entity_child_factor_max: float = 1.5
    population_scale_distribution: str = "uniform"
    population_scale_log_sigma: float = 0.8
    max_rows_per_table: int = 2048
    latent_dimension: int = 4
    optional_rate_min: float = 0.05
    optional_rate_max: float = 0.25
    affinity_strength: float = 1.0
    degree_strength: float = 0.8
    feature_missing_rate_min: float = 0.02
    feature_missing_rate_max: float = 0.15
    feature_missing_zero_probability: float = 0.0
    feature_missing_beta_alpha: float = 1.0
    feature_missing_beta_beta: float = 1.0
    feature_noise_scale_min: float = 0.0001
    feature_noise_scale_max: float = 0.3
    scm_signal_scale_min: float = 0.01
    scm_signal_scale_max: float = 10.0
    scm_meta_relative_std_min: float = 0.01
    scm_meta_relative_std_max: float = 1.0
    scm_activation_scale_min: float = 0.1
    scm_activation_scale_max: float = 100.0
    scm_output_scale_log_std: float = 1.4
    scm_long_tail_probability: float = 0.5
    scm_long_tail_alpha_min: float = 1.1
    scm_long_tail_alpha_max: float = 2.0
    categorical_cardinality_min: int = 3
    categorical_cardinality_max: int = 12
    categorical_high_cardinality_probability: float = 0.0
    categorical_high_cardinality_min: int = 13
    categorical_high_cardinality_max: int = 256
    # Categorical columns are softmax-sampled per row from signal logits plus
    # a per-table Dirichlet class prior. alpha controls class-frequency
    # imbalance (smaller -> concentrated on few classes); strength controls how
    # strongly the signal drives the category. Both are drawn log-uniformly.
    categorical_dirichlet_alpha_min: float = 0.1
    categorical_dirichlet_alpha_max: float = 1.0
    categorical_signal_strength_min: float = 0.5
    categorical_signal_strength_max: float = 2.0
    time_scale_seconds_min: float = 300.0
    time_scale_seconds_max: float = 86_400.0
    # <== 数据库日历区间 (epoch 秒)。事件时间只落在 [start, end] 内。
    # time_scale_seconds 收窄为基准时间尺度：TIME_LAGGED 的 lag 幅度与
    # burst/churn 特征时间，不再控制非滞后路径的无界间隔。
    calendar_start_seconds_min: int = 1_577_836_800  # 2020-01-01
    calendar_start_seconds_max: int = 1_735_689_600  # 2025-01-01
    calendar_span_seconds_min: float = 31_536_000.0  # 1 年
    calendar_span_seconds_max: float = 315_360_000.0  # 10 年
    # <== 事件时间机制权重 (每张带 TIME 列的表采样一次)。
    event_temporal_mechanism_weights: tuple[
        tuple[EventTemporalMechanism, float], ...
    ] = (
        (EventTemporalMechanism.STATIONARY, 0.30),
        (EventTemporalMechanism.BURST, 0.30),
        (EventTemporalMechanism.CHURN, 0.25),
        (EventTemporalMechanism.SEASONAL, 0.15),
    )
    burst_max_clusters: int = 3
    burst_cluster_width_min: float = 0.01
    burst_cluster_width_max: float = 0.12
    churn_exponent_min: float = 1.2
    churn_exponent_max: float = 3.0
    seasonal_period_days_min: int = 30
    seasonal_period_days_max: int = 365
    scm_weights: tuple[tuple[FeatureSCMFamily, float], ...] = (
        _DEFAULT_SCM_WEIGHTS
    )
    # Root-cause latent distribution prior — each table draws one family
    # that controls the shape of its exogenous latent variables.
    root_cause_weights: tuple[tuple[RootCauseFamily, float], ...] = (
        _DEFAULT_ROOT_CAUSE_WEIGHTS
    )
    # Optional per-role overrides. Missing roles inherit the global weights
    # and unit multipliers; Lookup keeps its historical exogenous prior.
    role_scm: tuple[tuple[TableRole, RoleSCMPrior], ...] = ()
    # MLP structural prior ranges — sampled once per database, each MLP
    # table then draws its own depth / hidden factor / dropout realisation.
    mlp_depth_min: int = 1
    mlp_depth_max: int = 4
    mlp_hidden_factor_min: float = 0.5
    mlp_hidden_factor_max: float = 4.0
    mlp_dropout_probability: float = 0.6
    mlp_dropout_rate_min: float = 0.05
    mlp_dropout_rate_max: float = 0.4

    def __post_init__(self) -> None:
        _integer_range(
            "entity rows",
            self.entity_rows_min,
            self.entity_rows_max,
        )
        if self.entity_rows_distribution not in {"loguniform", "lognormal"}:
            raise ValueError(
                "entity_rows_distribution must be loguniform or lognormal"
            )
        if (
            self.entity_rows_distribution == "lognormal"
            and not self.entity_rows_min
            <= self.entity_rows_median
            <= self.entity_rows_max
        ):
            raise ValueError("entity_rows_median must lie within entity row bounds")
        _positive_scalar("entity_rows_log_sigma", self.entity_rows_log_sigma)
        _integer_range(
            "lookup rows",
            self.lookup_rows_min,
            self.lookup_rows_max,
        )
        if self.lookup_rows_distribution not in {"uniform", "loguniform"}:
            raise ValueError(
                "lookup_rows_distribution must be uniform or loguniform"
            )
        for name, low, high in (
            ("event fanout", self.event_fanout_min, self.event_fanout_max),
            (
                "bridge rows factor",
                self.bridge_rows_factor_min,
                self.bridge_rows_factor_max,
            ),
            ("detail fanout", self.detail_fanout_min, self.detail_fanout_max),
            (
                "entity child factor",
                self.entity_child_factor_min,
                self.entity_child_factor_max,
            ),
        ):
            _positive_range(name, low, high)
        if self.population_scale_distribution not in {
            "uniform",
            "loguniform",
            "lognormal",
        }:
            raise ValueError(
                "population_scale_distribution must be uniform, loguniform "
                "or lognormal"
            )
        _positive_scalar(
            "population_scale_log_sigma",
            self.population_scale_log_sigma,
        )
        for name, value in (
            ("max_rows_per_table", self.max_rows_per_table),
            ("latent_dimension", self.latent_dimension),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_rows_per_table < self.entity_rows_min:
            raise ValueError("max_rows_per_table is below entity_rows_min")
        if not 0 <= self.optional_rate_min <= self.optional_rate_max < 1:
            raise ValueError("optional rates must satisfy 0 <= min <= max < 1")
        if self.affinity_strength <= 0 or self.degree_strength < 0:
            raise ValueError("affinity must be positive and degree non-negative")
        if not (
            0
            <= self.feature_missing_rate_min
            <= self.feature_missing_rate_max
            < 1
        ):
            raise ValueError("feature missing rates must satisfy 0 <= min <= max < 1")
        if not 0 <= self.feature_missing_zero_probability <= 1:
            raise ValueError(
                "feature_missing_zero_probability must be between zero and one"
            )
        _positive_scalar(
            "feature_missing_beta_alpha",
            self.feature_missing_beta_alpha,
        )
        _positive_scalar(
            "feature_missing_beta_beta",
            self.feature_missing_beta_beta,
        )
        _positive_range(
            "feature noise scale",
            self.feature_noise_scale_min,
            self.feature_noise_scale_max,
        )
        _positive_range(
            "SCM signal scale",
            self.scm_signal_scale_min,
            self.scm_signal_scale_max,
        )
        _positive_range(
            "SCM meta relative std",
            self.scm_meta_relative_std_min,
            self.scm_meta_relative_std_max,
        )
        _positive_range(
            "SCM activation scale",
            self.scm_activation_scale_min,
            self.scm_activation_scale_max,
        )
        if self.scm_output_scale_log_std <= 0:
            raise ValueError("scm_output_scale_log_std must be positive")
        if not 0 <= self.scm_long_tail_probability <= 1:
            raise ValueError("scm_long_tail_probability must be in [0, 1]")
        _positive_range(
            "SCM long-tail alpha",
            self.scm_long_tail_alpha_min,
            self.scm_long_tail_alpha_max,
        )
        if self.scm_long_tail_alpha_min <= 1:
            raise ValueError("scm_long_tail_alpha_min must be greater than 1")
        _integer_range("MLP depth", self.mlp_depth_min, self.mlp_depth_max)
        if self.mlp_depth_min < 1:
            raise ValueError("mlp_depth_min must be at least 1")
        _positive_range(
            "MLP hidden factor",
            self.mlp_hidden_factor_min,
            self.mlp_hidden_factor_max,
        )
        if not 0 <= self.mlp_dropout_probability <= 1:
            raise ValueError("mlp_dropout_probability must be in [0, 1]")
        if not 0 <= self.mlp_dropout_rate_min <= self.mlp_dropout_rate_max < 1:
            raise ValueError(
                "mlp_dropout_rate must satisfy 0 <= min <= max < 1"
            )
        _integer_range(
            "categorical cardinality",
            self.categorical_cardinality_min,
            self.categorical_cardinality_max,
        )
        if not 0 <= self.categorical_high_cardinality_probability <= 1:
            raise ValueError(
                "categorical_high_cardinality_probability must be between "
                "zero and one"
            )
        _integer_range(
            "high categorical cardinality",
            self.categorical_high_cardinality_min,
            self.categorical_high_cardinality_max,
        )
        if (
            self.categorical_high_cardinality_probability > 0
            and self.categorical_high_cardinality_min
            <= self.categorical_cardinality_max
        ):
            raise ValueError(
                "high categorical cardinality range must start above the "
                "ordinary range"
            )
        _positive_range(
            "categorical dirichlet alpha",
            self.categorical_dirichlet_alpha_min,
            self.categorical_dirichlet_alpha_max,
        )
        _positive_range(
            "categorical signal strength",
            self.categorical_signal_strength_min,
            self.categorical_signal_strength_max,
        )
        _positive_range(
            "time scale seconds",
            self.time_scale_seconds_min,
            self.time_scale_seconds_max,
        )
        _integer_range(
            "calendar start seconds",
            self.calendar_start_seconds_min,
            self.calendar_start_seconds_max,
        )
        if self.calendar_start_seconds_min < 0:
            raise ValueError("calendar_start_seconds_min must be non-negative")
        _positive_range(
            "calendar span seconds",
            self.calendar_span_seconds_min,
            self.calendar_span_seconds_max,
        )
        _validate_mechanism_weights(
            self.event_temporal_mechanism_weights,
            "event_temporal_mechanism_weights",
        )
        if self.burst_max_clusters < 1:
            raise ValueError("burst_max_clusters must be at least 1")
        _positive_range(
            "burst cluster width",
            self.burst_cluster_width_min,
            self.burst_cluster_width_max,
        )
        if not (
            0
            < self.burst_cluster_width_min
            <= self.burst_cluster_width_max
            < 1
        ):
            raise ValueError(
                "burst cluster width must satisfy 0 < min <= max < 1"
            )
        if self.churn_exponent_min <= 1:
            raise ValueError("churn_exponent_min must be greater than 1")
        _positive_range(
            "churn exponent",
            self.churn_exponent_min,
            self.churn_exponent_max,
        )
        _integer_range(
            "seasonal period days",
            self.seasonal_period_days_min,
            self.seasonal_period_days_max,
        )
        if self.seasonal_period_days_min < 1:
            raise ValueError("seasonal_period_days_min must be positive")
        _validate_scm_weights(self.scm_weights, "scm_weights")
        _validate_root_cause_weights(
            self.root_cause_weights,
            "root_cause_weights",
        )
        if not isinstance(self.role_scm, tuple):
            raise TypeError("role_scm must be a tuple")
        roles: list[TableRole] = []
        for item in self.role_scm:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("role_scm items must be role-prior pairs")
            role, prior = item
            if not isinstance(role, TableRole):
                raise TypeError("role_scm keys must be TableRole")
            if not isinstance(prior, RoleSCMPrior):
                raise TypeError("role_scm values must be RoleSCMPrior")
            roles.append(role)
        if len(set(roles)) != len(roles):
            raise ValueError("role_scm roles must be unique")

    def scm_prior_for_role(self, role: TableRole) -> RoleSCMPrior:
        """Resolve one role override without changing legacy defaults."""
        if not isinstance(role, TableRole):
            raise TypeError("role must be TableRole")
        for configured_role, prior in self.role_scm:
            if configured_role is role:
                return prior
        if role is TableRole.LOOKUP:
            return RoleSCMPrior(
                scm_weights=_LOOKUP_SCM_WEIGHTS,
                root_cause_weights=_LOOKUP_ROOT_CAUSE_WEIGHTS,
            )
        return RoleSCMPrior(
            scm_weights=self.scm_weights,
            root_cause_weights=self.root_cause_weights,
        )


class InstancePlanner:
    def __init__(self, config: InstancePlannerConfig | None = None) -> None:
        self.config = config or InstancePlannerConfig()

    def plan(
        self,
        *,
        sample_id: str,
        schema: PhysicalSchema,
        runtime: RuntimeContext,
        prior_plan: DatabasePriorPlan | None = None,
    ) -> InstancePlan:
        if not isinstance(schema, PhysicalSchema):
            raise TypeError("schema must be PhysicalSchema")
        if not isinstance(runtime, RuntimeContext):
            raise TypeError("runtime must be RuntimeContext")

        order = tuple(
            table.table_id
            for table in sorted(
                schema.tables,
                key=lambda table: (table.rank, table.table_id),
            )
        )
        incoming = {
            table.table_id: tuple(
                fk
                for fk in schema.foreign_keys
                if fk.child_table_id == table.table_id
            )
            for table in schema.tables
        }
        calendar_rng = runtime.numpy_rng("instance", "calendar")
        calendar_start = int(
            calendar_rng.integers(
                self.config.calendar_start_seconds_min,
                self.config.calendar_start_seconds_max + 1,
            )
        )
        calendar_span = int(
            round(
                exp(
                    calendar_rng.uniform(
                        log(self.config.calendar_span_seconds_min),
                        log(self.config.calendar_span_seconds_max),
                    )
                )
            )
        )
        calendar_end = calendar_start + calendar_span
        meta = sample_scm_meta_parameters(
            runtime.numpy_rng("instance", "scm-meta"),
            signal_mean_min=self.config.scm_signal_scale_min,
            signal_mean_max=self.config.scm_signal_scale_max,
            noise_mean_min=self.config.feature_noise_scale_min,
            noise_mean_max=self.config.feature_noise_scale_max,
            relative_std_min=self.config.scm_meta_relative_std_min,
            relative_std_max=self.config.scm_meta_relative_std_max,
            activation_scale_min=self.config.scm_activation_scale_min,
            activation_scale_max=self.config.scm_activation_scale_max,
            output_log_std=self.config.scm_output_scale_log_std,
            long_tail_probability=self.config.scm_long_tail_probability,
            long_tail_alpha_min=self.config.scm_long_tail_alpha_min,
            long_tail_alpha_max=self.config.scm_long_tail_alpha_max,
            mlp_depth_min=self.config.mlp_depth_min,
            mlp_depth_max=self.config.mlp_depth_max,
            mlp_hidden_factor_min=self.config.mlp_hidden_factor_min,
            mlp_hidden_factor_max=self.config.mlp_hidden_factor_max,
            mlp_dropout_probability=self.config.mlp_dropout_probability,
            mlp_dropout_rate_min=self.config.mlp_dropout_rate_min,
            mlp_dropout_rate_max=self.config.mlp_dropout_rate_max,
        )
        table_plans: dict[str, TableMechanismPlan] = {}
        for table_id in order:
            table = schema.table(table_id)
            row_count, strategy, multiplier = self._row_count(
                table.role,
                incoming[table_id],
                table_plans,
                runtime,
                table_id,
            )
            temporal = self._temporal_family(
                table.role,
                incoming[table_id],
                schema,
            )
            mechanism = (
                self._event_mechanism(runtime, table_id)
                if temporal is not TemporalFamily.NONE
                else EventTemporalMechanism.STATIONARY
            )
            role_scm = self.config.scm_prior_for_role(table.role)
            family = self._feature_family(role_scm, runtime, table_id)
            root_cause = self._root_cause_family(
                role_scm,
                runtime,
                table_id,
            )
            parameter_rng = runtime.numpy_rng(
                "instance", "table-parameters", table_id
            )
            scm_parameters = sample_table_scm_parameters(
                parameter_rng,
                meta,
                activation_scale_min=self.config.scm_activation_scale_min,
                activation_scale_max=self.config.scm_activation_scale_max,
                signal_scale_multiplier=role_scm.signal_scale_multiplier,
                noise_scale_multiplier=role_scm.noise_scale_multiplier,
                activation_scale_multiplier=(
                    role_scm.activation_scale_multiplier
                ),
                output_scale_multiplier=role_scm.output_scale_multiplier,
            )
            table_plans[table_id] = TableMechanismPlan(
                table_id=table_id,
                role=table.role,
                population=PopulationPlan(
                    strategy=strategy,
                    row_count=row_count,
                    parameters=(("multiplier", multiplier),),
                ),
                latent_dimension=self.config.latent_dimension,
                feature_family=family,
                root_cause_family=root_cause,
                temporal_family=temporal,
                event_mechanism=mechanism,
                latent_seed=runtime.seed("instance", "latent", table_id),
                feature_seed=runtime.seed("instance", "feature", table_id),
                temporal_seed=runtime.seed("instance", "time", table_id),
                parameters=(
                    (
                        "categorical_cardinality",
                        float(self._categorical_cardinality(parameter_rng)),
                    ),
                    (
                        "categorical_dirichlet_alpha",
                        self._categorical_dirichlet_alpha(parameter_rng),
                    ),
                    (
                        "categorical_signal_strength",
                        self._categorical_signal_strength(parameter_rng),
                    ),
                    (
                        "missing_rate",
                        self._missing_rate(parameter_rng),
                    ),
                    (
                        "time_scale_seconds",
                        float(
                            parameter_rng.uniform(
                                self.config.time_scale_seconds_min,
                                self.config.time_scale_seconds_max,
                            )
                        ),
                    ),
                )
                + (
                    (
                        ("burst_max_clusters", float(self.config.burst_max_clusters)),
                        (
                            "burst_cluster_width_min",
                            self.config.burst_cluster_width_min,
                        ),
                        (
                            "burst_cluster_width_max",
                            self.config.burst_cluster_width_max,
                        ),
                        ("churn_exponent_min", self.config.churn_exponent_min),
                        ("churn_exponent_max", self.config.churn_exponent_max),
                        (
                            "seasonal_period_days_min",
                            float(self.config.seasonal_period_days_min),
                        ),
                        (
                            "seasonal_period_days_max",
                            float(self.config.seasonal_period_days_max),
                        ),
                    )
                    if temporal is not TemporalFamily.NONE
                    else ()
                )
                + scm_parameters,
            )

        relations = self._relation_plans(schema, runtime)
        plan = InstancePlan(
            plan_id=f"instance_plan_{sample_id}",
            sample_id=sample_id,
            schema_id=schema.schema_id,
            blueprint_id=schema.blueprint_id,
            global_seed=runtime.seed("instance", "global-latent"),
            generation_order=order,
            tables=tuple(table_plans[table_id] for table_id in order),
            relations=relations,
            parameters=meta.parameters,
            calendar_start_seconds=calendar_start,
            calendar_end_seconds=calendar_end,
        )
        if prior_plan is None:
            return plan
        if prior_plan.semantic_schema.schema_id != schema.schema_id:
            raise ValueError("prior plan does not belong to physical schema")
        if prior_plan.family is PriorFamily.TEMPORAL_EVENT:
            return bind_temporal_event_plan(schema, plan, prior_plan)
        return bind_legacy_plan(plan, prior_plan)

    def _event_mechanism(
        self,
        runtime: RuntimeContext,
        table_id: str,
    ) -> EventTemporalMechanism:
        """Sample one bounded event-time mechanism for a temporal table."""
        rng = runtime.numpy_rng("instance", "event-mechanism", table_id)
        weights = self.config.event_temporal_mechanism_weights
        total = sum(weight for _mechanism, weight in weights)
        draw = rng.uniform(0.0, total)
        cumulative = 0.0
        for mechanism, weight in weights:
            cumulative += weight
            if draw < cumulative:
                return mechanism
        return weights[-1][0]

    def _row_count(
        self,
        role: TableRole,
        incoming: tuple[PhysicalForeignKey, ...],
        existing: dict[str, TableMechanismPlan],
        runtime: RuntimeContext,
        table_id: str,
    ) -> tuple[int, str, float]:
        rng = runtime.numpy_rng("instance", "population", table_id)
        parent_counts = [
            existing[fk.parent_table_id].population.row_count
            for fk in incoming
            if fk.parent_table_id in existing
            and fk.relation_strategy != "lookup_assignment"
        ]
        if role is TableRole.LOOKUP:
            if self.config.lookup_rows_distribution == "loguniform":
                count = round(
                    exp(
                        rng.uniform(
                            log(self.config.lookup_rows_min),
                            log(self.config.lookup_rows_max + 1),
                        )
                    )
                )
            else:
                count = int(
                    rng.integers(
                        self.config.lookup_rows_min,
                        self.config.lookup_rows_max + 1,
                    )
                )
            return count, "fixed_exogenous", 1.0
        if not parent_counts:
            if role is not TableRole.ENTITY:
                raise ValueError(f"root table {table_id} must be Entity/Lookup")
            if self.config.entity_rows_distribution == "lognormal":
                value = rng.lognormal(
                    mean=log(self.config.entity_rows_median),
                    sigma=self.config.entity_rows_log_sigma,
                )
                count = round(value)
            else:
                value = exp(
                    rng.uniform(
                        log(self.config.entity_rows_min),
                        log(self.config.entity_rows_max + 1),
                    )
                )
                # Preserve the historical log-uniform discretisation so old
                # seeds continue to materialize the same database instances.
                count = int(value)
            count = min(self.config.entity_rows_max, max(self.config.entity_rows_min, count))
            return count, "root_entity", 1.0

        if role is TableRole.EVENT:
            low, high = self.config.event_fanout_min, self.config.event_fanout_max
            strategy = "parent_conditioned_event"
            basis = max(parent_counts)
        elif role is TableRole.DETAIL:
            low, high = self.config.detail_fanout_min, self.config.detail_fanout_max
            strategy = "parent_conditioned_detail"
            basis = max(parent_counts)
        elif role is TableRole.BRIDGE:
            low = self.config.bridge_rows_factor_min
            high = self.config.bridge_rows_factor_max
            strategy = "joint_bridge_population"
            basis = int(sqrt(parent_counts[0] * parent_counts[-1]))
        else:
            low = self.config.entity_child_factor_min
            high = self.config.entity_child_factor_max
            strategy = "entity_hierarchy_population"
            basis = max(parent_counts)

        multiplier = self._population_multiplier(rng, low, high)
        count = max(1, min(self.config.max_rows_per_table, round(basis * multiplier)))
        return count, strategy, multiplier

    def _population_multiplier(self, rng, low: float, high: float) -> float:
        distribution = self.config.population_scale_distribution
        if distribution == "loguniform":
            return float(exp(rng.uniform(log(low), log(high))))
        if distribution == "lognormal":
            value = rng.lognormal(
                mean=0.5 * (log(low) + log(high)),
                sigma=self.config.population_scale_log_sigma,
            )
            return float(min(high, max(low, value)))
        return float(rng.uniform(low, high))

    def _missing_rate(self, rng) -> float:
        if (
            self.config.feature_missing_zero_probability == 0
            and self.config.feature_missing_beta_alpha == 1
            and self.config.feature_missing_beta_beta == 1
        ):
            return float(
                rng.uniform(
                    self.config.feature_missing_rate_min,
                    self.config.feature_missing_rate_max,
                )
            )
        if rng.random() < self.config.feature_missing_zero_probability:
            return 0.0
        unit = rng.beta(
            self.config.feature_missing_beta_alpha,
            self.config.feature_missing_beta_beta,
        )
        low = self.config.feature_missing_rate_min
        high = self.config.feature_missing_rate_max
        return float(low + unit * (high - low))

    def _categorical_cardinality(self, rng) -> int:
        high_probability = (
            self.config.categorical_high_cardinality_probability
        )
        if high_probability == 0:
            return int(
                rng.integers(
                    self.config.categorical_cardinality_min,
                    self.config.categorical_cardinality_max + 1,
                )
            )
        if rng.random() < high_probability:
            low = self.config.categorical_high_cardinality_min
            high = self.config.categorical_high_cardinality_max
        else:
            low = self.config.categorical_cardinality_min
            high = self.config.categorical_cardinality_max
        return int(round(exp(rng.uniform(log(low), log(high + 1)))))

    def _categorical_dirichlet_alpha(self, rng) -> float:
        return float(
            exp(
                rng.uniform(
                    log(self.config.categorical_dirichlet_alpha_min),
                    log(self.config.categorical_dirichlet_alpha_max),
                )
            )
        )

    def _categorical_signal_strength(self, rng) -> float:
        return float(
            exp(
                rng.uniform(
                    log(self.config.categorical_signal_strength_min),
                    log(self.config.categorical_signal_strength_max),
                )
            )
        )

    def _feature_family(
        self,
        prior: RoleSCMPrior,
        runtime: RuntimeContext,
        table_id: str,
    ) -> FeatureSCMFamily:
        rng = runtime.python_rng("instance", "scm-family", table_id)
        families, weights = zip(*prior.scm_weights)
        return rng.choices(families, weights=weights, k=1)[0]

    def _root_cause_family(
        self,
        prior: RoleSCMPrior,
        runtime: RuntimeContext,
        table_id: str,
    ) -> RootCauseFamily:
        """Sample the exogenous latent distribution family for one table."""
        rng = runtime.python_rng("instance", "root-cause", table_id)
        families, weights = zip(*prior.root_cause_weights)
        return rng.choices(families, weights=weights, k=1)[0]

    @staticmethod
    def _temporal_family(
        role: TableRole,
        incoming: tuple[PhysicalForeignKey, ...],
        schema: PhysicalSchema,
    ) -> TemporalFamily:
        if role is not TableRole.EVENT:
            return TemporalFamily.NONE
        if any(
            schema.table(fk.parent_table_id).role is TableRole.EVENT
            for fk in incoming
        ):
            return TemporalFamily.TIME_LAGGED
        return TemporalFamily.PARENT_BURST

    def _relation_plans(
        self,
        schema: PhysicalSchema,
        runtime: RuntimeContext,
    ) -> tuple[RelationMechanismPlan, ...]:
        results: list[RelationMechanismPlan] = []
        for table in sorted(schema.tables, key=lambda item: item.table_id):
            incoming = tuple(
                fk
                for fk in schema.foreign_keys
                if fk.child_table_id == table.table_id
            )
            structural = tuple(
                fk for fk in incoming if fk.relation_strategy != "lookup_assignment"
            )
            auxiliary = tuple(
                fk for fk in incoming if fk.relation_strategy == "lookup_assignment"
            )
            if table.role is TableRole.BRIDGE and len(structural) >= 2:
                results.append(
                    self._relation_group(
                        group_id=f"RG_{table.table_id}_bridge",
                        fks=structural,
                        family="affinity_bridge",
                        runtime=runtime,
                        allow_lookup_transition=False,
                    )
                )
            else:
                auxiliary = incoming

            for fk in auxiliary:
                results.append(
                    self._relation_group(
                        group_id=f"RG_{fk.foreign_key_id}",
                        fks=(fk,),
                        family=fk.relation_strategy,
                        runtime=runtime,
                        allow_lookup_transition=(table.role is TableRole.EVENT),
                    )
                )
        return tuple(results)

    def _relation_group(
        self,
        *,
        group_id: str,
        fks: tuple[PhysicalForeignKey, ...],
        family: str,
        runtime: RuntimeContext,
        allow_lookup_transition: bool,
    ) -> RelationMechanismPlan:
        rng = runtime.numpy_rng("instance", "relation-plan", group_id)
        if family == "lookup_assignment":
            choices = ["lookup_cpt", "lookup_softmax"]
            if allow_lookup_transition:
                choices.append("lookup_transition")
            family = choices[int(rng.integers(0, len(choices)))]
        optional_rates = tuple(
            0.0
            if fk.optionality is Optionality.REQUIRED
            else float(
                rng.uniform(
                    self.config.optional_rate_min,
                    self.config.optional_rate_max,
                )
            )
            for fk in fks
        )
        return RelationMechanismPlan(
            relation_group_id=group_id,
            foreign_key_ids=tuple(fk.foreign_key_id for fk in fks),
            parent_table_ids=tuple(fk.parent_table_id for fk in fks),
            child_table_id=fks[0].child_table_id,
            family=family,
            optional_rates=optional_rates,
            seed=runtime.seed("instance", "relation", group_id),
            parameters=(
                ("affinity_strength", self.config.affinity_strength),
                ("degree_strength", self.config.degree_strength),
            ),
        )


def _integer_range(name: str, low: int, high: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (low, high)):
        raise TypeError(f"{name} bounds must be integers")
    if low < 1 or high < low:
        raise ValueError(f"invalid {name} bounds")


def _positive_range(name: str, low: float, high: float) -> None:
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (low, high)):
        raise TypeError(f"{name} bounds must be numeric")
    if low <= 0 or high < low:
        raise ValueError(f"invalid {name} bounds")


def _positive_scalar(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_scm_weights(
    weights: tuple[tuple[FeatureSCMFamily, float], ...],
    name: str,
) -> None:
    if not isinstance(weights, tuple) or not weights:
        raise ValueError(f"{name} must be a non-empty tuple")
    for item in weights:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name} items must be family-weight pairs")
    families = tuple(family for family, _weight in weights)
    if len(set(families)) != len(families):
        raise ValueError(f"{name} families must be unique")
    for family, weight in weights:
        if not isinstance(family, FeatureSCMFamily):
            raise TypeError(f"{name} keys must be FeatureSCMFamily")
        _positive_scalar(f"{name} weight", weight)


def _validate_root_cause_weights(
    weights: tuple[tuple[RootCauseFamily, float], ...],
    name: str,
) -> None:
    if not isinstance(weights, tuple) or not weights:
        raise ValueError(f"{name} must be a non-empty tuple")
    for item in weights:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name} items must be family-weight pairs")
    families = tuple(family for family, _weight in weights)
    if len(set(families)) != len(families):
        raise ValueError(f"{name} families must be unique")
    for family, weight in weights:
        if not isinstance(family, RootCauseFamily):
            raise TypeError(f"{name} keys must be RootCauseFamily")
        _positive_scalar(f"{name} weight", weight)


def _validate_mechanism_weights(
    weights: tuple[tuple[EventTemporalMechanism, float], ...],
    name: str,
) -> None:
    if not isinstance(weights, tuple) or not weights:
        raise ValueError(f"{name} must be a non-empty tuple")
    for item in weights:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name} items must be mechanism-weight pairs")
    mechanisms = tuple(mechanism for mechanism, _weight in weights)
    if len(set(mechanisms)) != len(mechanisms):
        raise ValueError(f"{name} mechanisms must be unique")
    for mechanism, weight in weights:
        if not isinstance(mechanism, EventTemporalMechanism):
            raise TypeError(f"{name} keys must be EventTemporalMechanism")
        _positive_scalar(f"{name} weight", weight)


__all__ = ["RoleSCMPrior", "InstancePlannerConfig", "InstancePlanner"]
