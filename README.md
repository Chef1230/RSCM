# rdb-prior staged generation pipeline

Stage 01 samples a validated anonymous `SchemaBlueprint`, preserves private
motif provenance, and compiles a randomized `PhysicalSchema`. Stage 02 binds
that schema to a reproducible `InstancePlan` and materializes validated table
rows, keys, features, and event times. Stage 03 derives supervised tasks from
the frozen database without changing its schema or rows. Stage 04 exports each
task as one RDBPFN `dbinfer_bench` dataset with its task-specific visible view.

## Run

The schema command reads `configs/refactor_v1.yaml`. The reference config is
derived from the schema-related priors in `syn_data/configs/default.yaml` and
contains only options implemented by the current four-stage pipeline.

`configs/refactor_v2.yaml` is the complete production-size preset: 20k
schemas, eight stage-02 workers, two tasks per database, and every DBB/DFS/H5
switch exposed. Run all stages with:

```bash
bash scripts/v1/generate_v1.sh configs/refactor_v2.yaml
```

H5 remains disabled in that preset until `rdbpfn_export.h5_enabled` is set to
`true`, because it requires the external RDBPFN preprocessing environment.

```bash
bash scripts/v1/01_schema.sh
```

Validate and print the fully resolved configuration without writing files:

```bash
VALIDATE_CONFIG_ONLY=1 bash scripts/v1/01_schema.sh
```

Generate a small isolated test batch:

```bash
NUM_SCHEMAS=5 \
OUTPUT_DIR=outputs/smoke \
OVERWRITE=1 \
bash scripts/v1/01_schema.sh
```

Run the intended 20k schema feasibility batch:

```bash
NUM_SCHEMAS=20000 \
BASE_SEED=42 \
OUTPUT_DIR=outputs/v1_20k \
OVERWRITE=0 \
bash scripts/v1/01_schema.sh
```

`01_schema.sh` accepts these run-level environment overrides:

- `CONFIG_PATH`
- `RDB_PRIOR_CONFIG` (legacy alias, lower precedence than `CONFIG_PATH`)
- `OUTPUT_DIR`
- `NUM_SCHEMAS`
- `BASE_SEED`
- `START_INDEX`
- `SAMPLE_ID_PREFIX`
- `PROGRESS_EVERY`
- `SCHEMA_DOT`
- `SCHEMA_GRAPH_FORMAT` (`none`, `png`, `svg`, or `pdf`)
- `GRAPHVIZ_COMMAND`
- `OVERWRITE`
- `VALIDATE_CONFIG_ONLY`
- `PYTHON_BIN`

Unset variables preserve the YAML value. A non-option first argument is also
treated as the config path. Additional CLI arguments may be appended to the
script and take precedence over environment-derived options.

The Python entry point can also be called directly from an installed package:

```bash
rdb-prior schema --config configs/refactor_v1.yaml --count 5
```

Stage 01 writes DOT without an extra Python dependency. To also render one PNG
per schema, install Graphviz so `dot` is on `PATH`, then run:

```bash
SCHEMA_GRAPH_FORMAT=png bash scripts/v1/01_schema.sh
```

An existing DOT artifact can also be rendered manually:

```bash
dot -Tpng outputs/v1_sample/schema/schema_graphs/sample_000000.dot \
  -o outputs/v1_sample/schema/schema_graphs/sample_000000.png
```

Generate instances for the schema manifest configured in the YAML:

```bash
bash scripts/v1/02_instance.sh
```

The most common stage-02 overrides are:

```bash
OUTPUT_DIR=outputs/v1_20k \
NUM_INSTANCES=20000 \
INSTANCE_JOBS=8 \
OVERWRITE=0 \
bash scripts/v1/02_instance.sh
```

`START_INDEX`, `SHARD_ID`, `NUM_SHARDS`, `PROGRESS_EVERY`, `CONFIG_PATH`,
`PYTHON_BIN`, and `VALIDATE_CONFIG_ONLY` are also supported. `INSTANCE_JOBS`
(or its `JOBS` alias) overrides `instance_generation.num_workers`; each worker
generates, validates, and atomically stores one database. Results are sorted
back into schema-manifest order before the parent process writes the manifest.
The direct entry point is `rdb-prior instance --jobs 8`.

Generate tasks for the configured instance manifest:

```bash
bash scripts/v1/03_task.sh
```

Control the exact requested task count per selected database:

```bash
OUTPUT_DIR=outputs/v1_20k \
NUM_DATABASES=20000 \
TASKS_PER_DATABASE=2 \
OVERWRITE=0 \
bash scripts/v1/03_task.sh
```

`03_task.sh` also supports `START_INDEX`, `SHARD_ID`, `NUM_SHARDS`,
`PROGRESS_EVERY`, `CONFIG_PATH`, `PYTHON_BIN`, and
`VALIDATE_CONFIG_ONLY`.

Export task artifacts for RDBPFN:

```bash
bash scripts/v1/04_rdbpfn_export.sh
```

The common stage-04 overrides are:

```bash
OUTPUT_DIR=outputs/v1_20k \
NUM_EXPORTS=40000 \
VALIDATION_FRACTION=0.2 \
COMPRESS=1 \
OVERWRITE=0 \
bash scripts/v1/04_rdbpfn_export.sh
```

`04_rdbpfn_export.sh` also supports `START_INDEX`, `SHARD_ID`, `NUM_SHARDS`,
`MIN_VALIDATION_ROWS`, `PROGRESS_EVERY`, `CONFIG_PATH`, `PYTHON_BIN`, and
`VALIDATE_CONFIG_ONLY`. The direct entry point is `rdb-prior rdbpfn-export`.

Optionally run RDBPFN DFS and package the processed classification tasks into
the H5 format consumed by its pretraining dataloader:

```bash
OUTPUT_DIR=outputs/v1_20k \
H5_EXPORT=1 \
RDBPFN_PREPROCESSING_DIR=../RDBPFN/data_preprocessing \
DFS_DEPTH=1 \
DFS_JOBS=8 \
H5_TOTAL_ROWS=600 \
H5_MAX_COLUMNS=60 \
OVERWRITE=0 \
bash scripts/v1/04_rdbpfn_export.sh
```

`OUTPUT_DIR` is the run root for every stage. Stage 04 therefore writes to
`<OUTPUT_DIR>/rdbpfn`; unless `H5_OUTPUT` is set, its H5 file is written there
as `rdbpfn_tasks.h5` (with a shard suffix for sharded runs). Set
`H5_RUN_DFS=0` to package an already existing
`<OUTPUT_DIR>/rdbpfn-processed` tree. Other H5 overrides are
`H5_SEED`; `DFS_DEPTH` accepts 1 or 2. The current RDBPFN H5 training contract
is classification-only, so regression tasks remain available as DBB datasets
but are reported as skipped by the H5 packer.

Run all four configured stages with:

```bash
OUTPUT_DIR=outputs/v1_20k bash scripts/v1/generate_v1.sh
```

The YAML equivalent is a single path setting:

```yaml
paths:
  output_root: outputs/v1_20k
```

The default stage directories are `schema/`, `instance/`, `task/`, and
`rdbpfn/` beneath that root. `SCHEMA_OUTPUT_DIR`, `INSTANCE_OUTPUT_DIR`,
`TASK_OUTPUT_DIR`, and `RDBPFN_OUTPUT_DIR` remain exact stage-level overrides.

## Logging and progress

All four commands write timestamped logs to stderr and keep the final JSON
summary on stdout. Interactive terminals show a progress bar automatically;
redirected jobs emit one progress log every `progress_every` completed items.

```bash
rdb-prior task \
  --config configs/refactor_v1.yaml \
  --log-level DEBUG \
  --log-file logs/task_v1.log \
  --progress
```

The shell scripts expose the same controls:

```bash
LOG_LEVEL=INFO \
LOG_FILE=logs/generate_v1.log \
PROGRESS_BAR=1 \
PROGRESS_WIDTH=32 \
bash scripts/v1/generate_v1.sh
```

Use `PROGRESS_BAR=0` or `--no-progress` for batch logs. Valid log levels are
`DEBUG`, `INFO`, `WARNING`, and `ERROR`. DEBUG records each committed artifact;
INFO records stage start, periodic progress, and completion.

## Configuration

The YAML loader rejects unknown fields, wrong scalar types, unsupported roles,
unknown motif names, overlapping feature-column ranges and inconsistent
value/weight lists.

- `seed`: deterministic root seed.
- `paths.output_root`: run root; stage directories and their manifest paths
  are derived automatically.
- `generation`: batch size, index range, artifact ID prefix, progress,
  overwrite behavior and project version.
- `schema_graph`: DOT creation, optional Graphviz rendering format and whether
  diagnostic column/role metadata is included in the graph.
- `schema`: table-count range/distribution, rank bound, extra-edge sampling and
  Blueprint ID prefix. Motif occurrence bounds and background attachment
  probability prevent every table from being forced by a structural motif.
- `motifs.weights`: enabled structural motifs and positive sampling weights.
- `physical_design`: PhysicalSchema ID prefix, feature-column defaults,
  table-count rules, optional role overrides, nullability probability and
  primary-key name candidates.
- `instance`: role-conditioned population bounds, shared latent dimension, FK
  affinity, optionality rates, global/role-specific SCM priors, missingness,
  noise, categorical cardinality and its softmax/Dirichlet sampling bounds, and
  event-time scales.
- `instance_generation`: schema selection, deterministic sharding, progress,
  overwrite behavior and project version.
- `task`: tasks per database, mechanism weights, support/query sizing,
  classification quality thresholds, future cutoff/horizon ranges,
  history-gated soft-label propensity weights, temporal aggregate
  window-regime mixing (short/long/repeated), and interaction-response beta
  weight / invert-probability ranges.
- `task_generation`: instance selection, deterministic sharding, progress,
  overwrite behavior and project version.
- `rdbpfn_export`: task selection, train/validation split, compressed NPZ,
  deterministic sharding, progress, overwrite behavior and project version.

Feature-column precedence is `role override -> table-count rule -> default`.
The only valid role override keys are `entity`, `event`, `lookup`, `bridge`
and `detail`.

`instance.role_scm.<role>` can override `scm_weights`,
`root_cause_weights`, `signal_scale_multiplier`, `noise_scale_multiplier`,
`activation_scale_multiplier`, and `output_scale_multiplier`. Missing fields
inherit the global instance prior; missing roles use global weights with unit
multipliers. For backward compatibility, an unconfigured Lookup role remains
exogenous with a standard-normal root cause.

Connectivity, FK DAG, self-loop prohibition, parallel-FK prohibition, rank
ordering, role-edge legality, Entity/Event presence and Bridge parent counts
are hard validity contracts. They are deliberately not configurable switches.
Task/Process/Temporal relations are also not schema motif configuration; they
belong to later Task/Process stages.

## Output

```text
OUTPUT_DIR/schema/
  manifest.json
  schemas/
    sample_000000.json
    sample_000001.json
    ...
  schema_graphs/
    sample_000000.dot
    sample_000001.dot
    ...
```

Each V2 schema artifact contains the runtime reproduction record, completed
logical blueprint, anonymous motif occurrences and bindings, physical schema,
logical-to-physical compilation trace, and validation report. The artifact can
be loaded with `rdb_prior.artifacts.load_schema_artifact` by a later stage.

Motif provenance and role metadata are generator-private. Physical table and
feature names remain anonymous; exporters must not expose generator metadata as
model features.
The manifest records graph paths under each entry's `graph_artifacts` field.
Rendered `png`, `svg`, or `pdf` files appear beside their DOT source when
`schema_graph.render_format` is enabled.

Stage 01 stops at `CompilationResult(PhysicalSchema, CompilationTrace)`.
The stage-02 output is:

```text
OUTPUT_DIR/instance/
  manifest.json
  instances/sample_000000/
    artifact.json
    instance_plan.json
    runtime.json
    validation.json
    tables/*.npz
```

Feature SCM families, root-cause families, and realized SCM scales are
conditioned on each table role. The default prior emphasizes Linear mechanisms
for Entity/Detail tables, CAM/MLP mechanisms for Event tables, and exogenous
features for Lookup tables. Lookup incoming selections use CPT, latent softmax,
or state-transition mechanisms. Event tables use parent-burst or time-lagged
temporal mechanisms. Bridge parents are sampled jointly with affinity. One
shared latent registry drives FK choice and feature contexts; FK rows are
therefore not sampled uniformly.

The production `refactor_v2.yaml` uses a realistic heavy-tailed instance
prior. Root Entity populations are bounded log-normal draws; Lookup sizes are
log-uniform; Event, Detail, Bridge, and child-Entity populations are
parent-conditioned log-normal multipliers. Missingness is zero-inflated with
a Beta-distributed non-zero tail, noise is log-uniform, and categorical
cardinality contains a small high-cardinality component. Categorical feature
columns are softmax-sampled per row from the signal logits plus a per-table
Dirichlet class prior (alpha controls class imbalance, signal strength the
causal link), so category frequencies are realistically non-uniform. Role-specific
feature widths and role-specific SCM priors prevent large schemas from forcing every
table to be unrealistically narrow or causally interchangeable. The bounds
remain explicit so large 20k-schema runs cannot create unbounded tables.

The instance artifact can be loaded with
`rdb_prior.artifacts.load_instance_artifact`.

The stage-03 output is:

```text
OUTPUT_DIR/task/
  manifest.json
  tasks/sample_000000/task_sample_000000_000/
    artifact.json
    task_plan.json
    task_data.npz
    runtime.json
    validation.json
```

V1 implements `relation_attribute` and `future_event_existence`. Relation
attribute tasks mask the physical target column and split Event rows in time
order. Future-event tasks label Entity rows from one Entity -> Event FK in a
sampled future window; every Event table is hidden after the common cutoff.
Rows downstream of hidden events are also hidden through the FK closure; use
`rdb_prior.task.view.build_task_view` to apply the canonical view. Invalid,
degenerate, leaking, or undersized tasks are rejected. By default the
pipeline requires exactly `tasks_per_database` valid tasks and fails instead
of silently reducing the count. Load artifacts with
`rdb_prior.task.artifacts.load_task_artifact`.

The `history_gated_future_inactive` / `history_gated_future_active` sub-modes
emit soft propensity targets: each entity with prior history draws
`Bernoulli(propensity)` where the propensity rises with pre-cutoff event
frequency and falls with the silence since the last event (so recently-active,
high-frequency entities are more likely "active"). The requested positive rate
only sets a logistic base-rate intercept — a soft constraint, not a hard
post-hoc match — and the exact labels are replayable from `plan.seed` plus the
stored `beta_freq` / `beta_sil` / `base_rate` / `label_retries` parameters.
`temporal_relational_aggregate` samples each task's aggregation window from a
mixture of SHORT, LONG and REPEATED (short + long nested at the same cutoff,
summed) regimes, storing `window` / `window_extended` / `window_regime`.

`interaction_response` predicts, for each Event row interpreted as one interaction
(`entity/user -> event -> item/context`, the item optional), whether that
interaction produced a response. A row is excluded (`-1`) unless its entity has at
least one prior interaction strictly before the row's own time; otherwise it draws
`Bernoulli(sigmoid(z))` with `z = logit(base_rate) + beta_u*U_e + beta_f*F_e -
beta_s*S_e + beta_p*P_e`, where `U_e` is a normalized entity feature, `F_e` /
`P_e` are log1p historical interaction counts of the entity and (optional) item,
and `S_e` is the normalized silence since the entity's last interaction. Each task
samples its beta weights and flips the label (`invert`, positive means ignore)
with `interaction_invert_probability`, so one mechanism covers both response/CTR
and ignore. Labels replay exactly from `plan.seed` plus the stored
`beta_u` / `beta_f` / `beta_s` / `beta_p` / `base_rate` / `invert` /
`label_retries` parameters.

The stage-04 output is directly loadable by RDBPFN:

```text
OUTPUT_DIR/rdbpfn/
  manifest.json
  task_sample_000000_000/
    metadata.yaml
    data/
      t_000_abcd.npz
      ...
    task_sample_000000_000/
      train.npz
      validation.npz
      test.npz
  rdbpfn_tasks.h5              # present when h5_enabled/H5_EXPORT is true
```

DFS writes relational features to the sibling directory
`OUTPUT_DIR/rdbpfn-processed/`. H5 rows keep train+validation (support) before
test (query), and `single_eval_pos` stores that boundary. The writer streams
one processed task at a time, so packaging does not retain the whole corpus in
memory. Per-dataset staging artifacts land in `OUTPUT_DIR/rdbpfn-tmp/` and are
removed once a DFS run succeeds; a failed run keeps them for debugging. While
DFS runs, its subprocess output is written to
`OUTPUT_DIR/rdbpfn_dfs_depth<N>.log` and a completed/total heartbeat is logged
every 60 seconds.

One directory is emitted per task because row visibility, cutoff time, and
masked targets are task-specific. The exporter applies the canonical
`TaskView`, translates logical row references to physical key values, removes
masked target columns, and converts time values to `datetime64[ns]`. Support
examples are deterministically split into DBB train/validation sets; query
examples become the test set. NPZ compression is enabled by default to limit
the extra on-disk copy.

## Sparse MLP router

The optional sparse route replaces DFS without changing stages 01--03. It
builds typed cell tokens, retains one output token per target column, creates a
task context from support rows and support labels only, scores at most 20
depth-2 schema paths with an MLP, executes the top three paths, and selects at
most eight observable source columns per path. Related rows are sampled with a
bounded fanout and aggregated independently for every path--source-column
pair. These relation-column tokens and the original target-column tokens feed
the jointly trained PFN-style transformer; no path latent slots are used.
Each PFN episode is capped at 600 support-plus-query rows; query subsampling is
label-blind.

The objective is:

```text
L = L_query-pred
  + lambda_route * L_route
  + lambda_cost * L_cost
  + lambda_sparse * L_sparse
  + lambda_diversity * L_diversity
```

`L_route` reads required/optional/distractor path labels from the synthetic
Task DSL. Generator-private table roles, motifs, SCM families and relation
strategies never enter model features. Query labels enter only
`L_query-pred`; schema/database IDs, rather than individual tasks, determine
the validation split.

Router training lazily loads task artifacts, caches recently used
schema/instance objects, prefetches CPU tensorization, pads heterogeneous tasks
into a GPU batch, and aggregates only the hard-selected `Top-K x C` relation
slots. The reference v2 configuration uses `batch_size: 8`, `num_workers: 8`,
`prefetch_factor: 2`, and BF16. Override them from the shell when tuning for a
specific GPU:

```bash
CUDA_VISIBLE_DEVICES=5 DEVICE=cuda \
ROUTER_BATCH_SIZE=16 NUM_WORKERS=8 PREFETCH_FACTOR=2 MIXED_PRECISION=bf16 \
bash scripts/v1/04b_router_train.sh configs/refactor_v2.yaml --progress
```

Use `MIXED_PRECISION=fp16` on CUDA devices without BF16 support. Batch size is
the main GPU-memory control; increase it gradually while watching peak memory.

Install the optional model dependency and run:

```bash
pip install -e '.[router]'
bash scripts/v1/04b_router_train.sh configs/refactor_v2.yaml
bash scripts/v1/05_routed_h5.sh configs/refactor_v2.yaml
bash scripts/v1/06_tfm_train.sh configs/refactor_v2.yaml
```

For a fresh output directory, the complete schema-to-routed-H5 sequence is:

```bash
bash scripts/v1/generate_v2.sh configs/refactor_v2.yaml
```

This route-native sequence does not invoke `04_rdbpfn_export.sh`; that script
is retained for the legacy RDBPFN/DFS export path.

`06_tfm_train.sh` launches `RDBPFN_routed` with one task per device. Its main
overrides are `RDBPFN_ROOT`, `ROUTED_H5_OUTPUT`, `NUM_PROCESSES`,
`MIXED_PRECISION`, `TFM_NUM_STEPS`, `TFM_NUM_EPOCHS`, `TFM_LR`,
`TFM_SAVE_EVERY_EVALS`, `TFM_FIND_UNUSED_PARAMETERS`,
`TFM_LOAD_CHECKPOINT`, and `TFM_SAVE_CHECKPOINT`. Additional arguments are
forwarded as Hydra overrides.

Stage 06 reads the routed H5 path from the selected TFM YAML by default.
Set `ROUTED_H5_OUTPUT` only when the YAML path should be overridden.

Checkpoints are written below `OUTPUT_DIR/router/checkpoints/`. The H5 file at
`OUTPUT_DIR/routed/routed_tasks.h5` stores per-task target tokens, selected
relation-column tokens and masks, support/query mask, labels, route/column
scores, selected indices, observable path descriptors and measured costs.
Because token widths vary by task, each task is an H5 group rather than one
globally padded DFS matrix.

### Real benchmark router evaluation

Real benchmark data must first be represented by the same leakage-safe task
artifact contract as synthetic data: a task manifest whose entries reference a
schema artifact, database-instance artifact and task artifact. Benchmark
train/validation rows are support rows and benchmark test rows are query rows.
The router sees support labels only.

RelBench entity tasks can be converted and evaluated through one shell entry.
Install the optional dependency once with `pip install -e '.[relbench,router]'`,
then run:

```bash
CUDA_VISIBLE_DEVICES=5 \
RELBENCH_DATASET=rel-amazon \
RELBENCH_TASK=user-churn \
MAX_ROWS_PER_TASK=600 QUERY_ROWS_PER_TASK=256 \
OVERWRITE=1 \
bash scripts/eval/relbench.sh convert
```

The generated router manifest is
`outputs/relbench/rel-amazon/user-churn/task/manifest.json`. The importer maps
all PK/FK values to contiguous row indices and creates a prediction-anchor row
for every RelBench train/validation/test example. Test rows are chunked so no
row is silently removed by the model's per-task row limit. Historical event
tables are filtered using each anchor row's prediction timestamp, rather than
one dataset-wide cutoff. `relbench_metadata.json` preserves original table and
column mappings, split sizes, query chunk ranges, and the exact mapping from
prediction `row_id` back to RelBench test order.

The importer currently supports scalar EntityTask targets: binary
classification, multiclass classification, and regression. Multilabel and
link/recommendation tasks are rejected because the current router task contract
does not represent their target shapes.

Use the same entry for the rest of the benchmark pipeline:

```bash
CUDA_VISIBLE_DEVICES=5 \
RELBENCH_DATASET=rel-amazon RELBENCH_TASK=user-churn \
ROUTER_CHECKPOINT=/absolute/path/to/router/checkpoints/best.pt \
DEVICE=cuda MIXED_PRECISION=bf16 OVERWRITE=1 \
bash scripts/eval/relbench.sh eval

# Export routed relation tokens for the downstream TFM stage.
CUDA_VISIBLE_DEVICES=5 \
RELBENCH_DATASET=rel-amazon RELBENCH_TASK=user-churn \
ROUTER_CHECKPOINT=/absolute/path/to/router/checkpoints/best.pt \
DEVICE=cuda OVERWRITE=1 \
bash scripts/eval/relbench.sh h5

# Consume routed.h5 with the stage-06 TFM checkpoint and run official scoring.
CUDA_VISIBLE_DEVICES=5 \
RELBENCH_DATASET=rel-amazon RELBENCH_TASK=user-churn \
TFM_CHECKPOINT=/absolute/path/to/checkpoints/RDBPFN_routed/model.pt \
DEVICE=cuda MIXED_PRECISION=bf16 OVERWRITE=1 \
bash scripts/eval/relbench.sh tfm
```

`bash scripts/eval/relbench.sh all` runs convert, router eval, and H5 export in
sequence. A complete `eval` also restores the original RelBench test order and
writes official `task.evaluate()` results to
`router_eval/relbench_metrics.json`; bounded `NUM_TASKS`/`START_INDEX` runs skip
that official score because their prediction set is incomplete. Set
`SUPPORT_ROWS`, `MAX_CLASSES`, `RELBENCH_OUTPUT`,
`ROUTER_EVAL_OUTPUT`, `ROUTED_H5_OUTPUT`, `TFM_CHECKPOINT`,
`TFM_MODEL_CONFIG`, `TFM_PREDICTIONS_OUTPUT`, `NUM_TASKS`, or `START_INDEX`
through environment variables when needed. For a cached/offline RelBench
dataset, set `DOWNLOAD=0`. The `tfm` action streams one independently shaped
binary routed task at a time, writes only query-row probabilities, and then
restores official RelBench test order for scoring.

Run the final end-to-end benchmark path with `pipeline`: it converts RelBench,
uses the Router checkpoint to export `routed.h5`, consumes that file with the
stage-06 TFM checkpoint, and runs official scoring. Routed regression and
multiclass groups are skipped because the current TFM contract is binary-only;
if no supported task remains, the pipeline reports `skipped` and exits cleanly.

Run every cached task for one RelBench dataset with
`scripts/eval/relbench_all.sh`. The wrapper discovers task directories under
`$RELBENCH_CACHE_DIR/<dataset>/tasks`, converts each task, and continues only
for binary classification. Regression, multiclass, link, recommendation, and
other unsupported targets are reported as skipped without stopping later
tasks. Set comma-separated `RELBENCH_TASKS` to run an explicit subset.

Evaluation parameters can be stored in a strict YAML file under
`configs/eval/`. Relative paths in these files are resolved against the
`02_syn_data` project root. Environment variables have higher priority than
YAML values, which makes one-off overrides possible without editing the file.

```bash
# One RelBench task: convert -> routed H5 -> TFM inference -> official score.
bash scripts/eval/relbench.sh pipeline configs/eval/relbench_f1.yaml

# Every cached task for the configured dataset.
bash scripts/eval/relbench_all.sh configs/eval/relbench_amazon.yaml

# Override one YAML value for a single run.
CUDA_VISIBLE_DEVICES=6 PROGRESS_EVERY=5 \
bash scripts/eval/relbench.sh pipeline configs/eval/relbench_f1.yaml
```

The supported YAML sections are `relbench`, `router`, `h5`, `tfm`, and
`runtime`. Unknown or misspelled fields fail immediately instead of silently
falling back to defaults. With `relbench.reuse_converted: true`, conversion is
skipped when the metadata plus schema, instance, and task manifests are all
present. Set it to `false` after changing conversion or query-chunk parameters
to force a rebuild. RelBench import progress covers each database table,
schema and instance writes, every query chunk, and final metadata.

Evaluate a trained checkpoint directly on those query rows:

```bash
CUDA_VISIBLE_DEVICES=5 \
BENCHMARK_TASK_MANIFEST=/absolute/path/to/benchmark/tasks/manifest.json \
ROUTER_CHECKPOINT=/absolute/path/to/router/checkpoints/best.pt \
ROUTER_EVAL_OUTPUT=outputs/benchmark/router_eval \
DEVICE=cuda MIXED_PRECISION=bf16 OVERWRITE=1 \
bash scripts/v1/04c_router_eval.sh
```

Use `NUM_TASKS` and `START_INDEX` for a bounded smoke test or shard. The output
directory contains:

```text
metrics.json       # per-task and aggregate classification/regression metrics
predictions.jsonl  # one query prediction per line with task_id and row_id
```

Classification metrics are accuracy, balanced accuracy, log loss and macro
one-vs-rest ROC-AUC when defined. Regression metrics are MAE, RMSE and R2.
Predictions include encoded and original classification values. To create the
routed-token H5 for a later consumer, use the same manifest and checkpoint:

```bash
CUDA_VISIBLE_DEVICES=5 \
TASK_MANIFEST=/absolute/path/to/benchmark/tasks/manifest.json \
ROUTER_CHECKPOINT=/absolute/path/to/router/checkpoints/best.pt \
ROUTED_H5_OUTPUT=outputs/benchmark/routed/benchmark.h5 \
DEVICE=cuda OVERWRITE=1 \
bash scripts/v1/05_routed_h5.sh configs/refactor_v2.yaml --progress
```

Routed H5 task groups include target-table `row_ids` and the JSON-encoded
`class_values` attribute so downstream benchmark predictions can be mapped
back to source rows and original class labels. Export loads task artifacts
lazily, keeping memory bounded for large benchmarks.

Run RDBPFN depth-1 DFS on every exported task dataset from the RDBPFN
`data_preprocessing` directory:

```bash
cd ../RDBPFN/data_preprocessing
bash benchmark_preprocess_depth1.sh \
  ../../02_syn_data/outputs/v1_sample/rdbpfn 8
```

Use `benchmark_preprocess_depth2.sh` in the same way for depth 2.

Business-process grammar, domain prototypes, role-aware compiler replacement,
database-design randomization and additional Task DSL mechanisms remain later
stages.
