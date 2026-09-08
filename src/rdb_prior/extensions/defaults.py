"""Default and no-op implementations of future pipeline extensions."""

from __future__ import annotations

from dataclasses import dataclass

from rdb_prior.compilation.compiler import (
    PhysicalCompilerConfig,
    PhysicalSchemaCompiler,
)
from rdb_prior.compilation.model import CompilationResult
from rdb_prior.extensions.interfaces import ExtensionBundle
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.blueprint import SchemaBlueprint
from rdb_prior.schema.semantics import SemanticSchemaPlan
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig


@dataclass(frozen=True, slots=True)
class AnonymousDefaultDomain:
    def sample(self, runtime: RuntimeContext) -> None:
        return None


@dataclass(frozen=True, slots=True)
class DefaultBlueprintProvider:
    sampler: BlueprintSampler

    def sample(
        self,
        sample_id: str | int,
        runtime: RuntimeContext,
        domain: object | None,
    ) -> SchemaBlueprint:
        return self.sampler.sample(sample_id, runtime)


@dataclass(frozen=True, slots=True)
class NoProcessGrammar:
    def instantiate(
        self,
        domain: object | None,
        blueprint: SchemaBlueprint,
        runtime: RuntimeContext,
    ) -> tuple[object, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class DeferredLegacyTask:
    def plan(
        self,
        domain: object | None,
        blueprint: SchemaBlueprint,
        processes: tuple[object, ...],
        runtime: RuntimeContext,
    ) -> None:
        return None


@dataclass(frozen=True, slots=True)
class FixedDatabaseDesign:
    def sample(
        self,
        blueprint: SchemaBlueprint,
        task_plan: object | None,
        runtime: RuntimeContext,
    ) -> None:
        return None


@dataclass(frozen=True, slots=True)
class IdentitySchemaCompiler:
    compiler: PhysicalSchemaCompiler

    def compile(
        self,
        blueprint: SchemaBlueprint,
        design: object | None,
        sample_id: str | int,
        runtime: RuntimeContext,
        *,
        semantic_schema: SemanticSchemaPlan | None = None,
    ) -> CompilationResult:
        return self.compiler.compile_result(
            blueprint,
            sample_id,
            runtime,
            semantic_schema=semantic_schema,
        )


def default_extension_bundle(
    sampler: BlueprintSamplerConfig,
    compiler: PhysicalCompilerConfig,
) -> ExtensionBundle:
    return ExtensionBundle(
        domain=AnonymousDefaultDomain(),
        blueprint=DefaultBlueprintProvider(BlueprintSampler(sampler)),
        process=NoProcessGrammar(),
        task=DeferredLegacyTask(),
        design=FixedDatabaseDesign(),
        compiler=IdentitySchemaCompiler(PhysicalSchemaCompiler(compiler)),
    )


__all__ = [
    "AnonymousDefaultDomain",
    "DefaultBlueprintProvider",
    "NoProcessGrammar",
    "DeferredLegacyTask",
    "FixedDatabaseDesign",
    "IdentitySchemaCompiler",
    "default_extension_bundle",
]
