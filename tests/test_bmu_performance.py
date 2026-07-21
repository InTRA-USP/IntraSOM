"""Correctness and performance tests for the optimized IntraSOM BMU search.

Run from the repository root after editable installation:

    python test_bmu_performance.py --mode quick
    python test_bmu_performance.py --mode full --repeats 5

To test a standalone source file instead of the installed package:

    python test_bmu_performance.py --module-file intrasom/intrasom.py

The script validates numerical equivalence against scikit-learn and reports
median timings for complete data, repeated/random missing patterns, projection,
rough/fine behavior, float32/float64, BMU1/BMU2, and multiple block sizes.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable


def _fmt_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.1f} µs"
    if value < 1.0:
        return f"{value * 1e3:.1f} ms"
    return f"{value:.3f} s"


class ProgressReporter:
    """Small terminal reporter with immediate, line-buffered progress output."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.started = time.perf_counter()
        self.current_section = None

    def line(self, message: str = "") -> None:
        if self.enabled:
            print(message, flush=True)

    def section(self, title: str) -> None:
        if not self.enabled:
            return
        if self.current_section is not None:
            print(flush=True)
        self.current_section = title
        print("=" * 88, flush=True)
        print(title, flush=True)
        print("=" * 88, flush=True)

    def start(self, label: str) -> float:
        self.line(f"  ▶ {label}")
        return time.perf_counter()

    def done(self, label: str, elapsed: float, detail: str = "") -> None:
        suffix = f" | {detail}" if detail else ""
        self.line(f"  ✓ {label}: {_fmt_seconds(elapsed)}{suffix}")

    def total(self) -> None:
        self.line(f"\nTempo total: {_fmt_seconds(time.perf_counter() - self.started)}")

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances


def load_som_class(module_file: str | None = None):
    if module_file is None:
        candidates = (
            "intrasom.intrasom",
            "intrasom",
        )
        errors = []
        for name in candidates:
            try:
                module = importlib.import_module(name)
                if hasattr(module, "SOM"):
                    return module.SOM
            except Exception as exc:  # pragma: no cover - diagnostic path
                errors.append(f"{name}: {exc}")
        raise ImportError(
            "Could not import SOM. Install IntraSOM in editable mode or pass "
            "--module-file. Attempts: " + "; ".join(errors)
        )

    path = Path(module_file).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    # Load the standalone module with minimal stubs for its relative imports.
    package_name = "_intrasom_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    sys.modules[package_name] = package

    codebook_stub = types.ModuleType(f"{package_name}.codebook")
    codebook_stub.Codebook = object
    sys.modules[codebook_stub.__name__] = codebook_stub

    objects_stub = types.ModuleType(f"{package_name}.object_functions")
    objects_stub.NeighborhoodFactory = object
    objects_stub.NormalizerFactory = object
    sys.modules[objects_stub.__name__] = objects_stub

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.intrasom",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SOM


def make_som(SOM, codebook: np.ndarray, *, missing: bool, actual_train: str,
             previous_epoch: bool = True):
    som = object.__new__(SOM)
    som.codebook = SimpleNamespace(
        matrix=np.asarray(codebook),
        nnodes=codebook.shape[0],
    )
    som.missing = missing
    som.actual_train = actual_train
    som.previous_epoch = previous_epoch
    return som


def create_missing_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    pattern: str,
    dtype,
) -> np.ndarray:
    data = rng.normal(size=(n_samples, n_features)).astype(dtype)

    if pattern == "repeated":
        masks = np.ones((4, n_features), dtype=bool)
        if n_features >= 2:
            masks[1, -1] = False
        if n_features >= 3:
            masks[2, -2:] = False
        masks[3, ::2] = False
        chosen = masks[np.arange(n_samples) % len(masks)]
        data[~chosen] = np.nan
    elif pattern == "random":
        missing_mask = rng.random(data.shape) < 0.30
        data[missing_mask] = np.nan
    elif pattern == "high_missing":
        missing_mask = rng.random(data.shape) < 0.70
        data[missing_mask] = np.nan
    else:
        raise ValueError(pattern)

    # Guarantee at least one observed feature per sample for BMU comparisons.
    empty = np.isnan(data).all(axis=1)
    if np.any(empty):
        data[empty, 0] = rng.normal(size=empty.sum()).astype(dtype)
    return data


def median_time(
    func: Callable[[], object],
    repeats: int,
    *,
    reporter: ProgressReporter | None = None,
    label: str = "benchmark",
) -> tuple[float, object]:
    if reporter is not None:
        reporter.line(f"    aquecimento: {label}")
    func()  # warm-up BLAS and allocation paths

    values = []
    result = None
    for repeat in range(1, repeats + 1):
        if reporter is not None:
            reporter.line(f"    repetição {repeat}/{repeats}: {label}")
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        values.append(elapsed)
        if reporter is not None:
            reporter.line(f"      concluída em {_fmt_seconds(elapsed)}")

    median = float(np.median(values))
    if reporter is not None:
        reporter.line(f"    mediana: {_fmt_seconds(median)}")
    return median, result


def assert_distance_equivalence(reference, result, dtype, label):
    rtol = 2e-4 if np.dtype(dtype) == np.float32 else 1e-8
    atol = 5e-3 if np.dtype(dtype) == np.float32 else 1e-7
    if not np.allclose(reference, result, rtol=rtol, atol=atol, equal_nan=True):
        difference = np.nanmax(np.abs(reference - result))
        raise AssertionError(f"{label}: distance mismatch; max abs diff={difference}")


def assert_bmu_equivalence(
    reference,
    result,
    label,
    *,
    reference_distance_matrix=None,
    rtol=2e-4,
    atol=5e-3,
):
    """Validate BMUs while allowing numerically equivalent near-ties.

    Exact neuron indices are required by default. When ``reference_distance_matrix``
    is supplied, an index mismatch is accepted only if the neuron selected by the
    tested implementation has a reference distance indistinguishable from the
    true minimum within the same numerical tolerance. This matters mainly for
    float32 projection tests, where BLAS and scikit-learn can round near-tied
    neurons in a different order without changing the actual minimum distance.
    """
    reference_idx = reference[0].astype(int)
    result_idx = result[0].astype(int)
    mismatch_mask = reference_idx != result_idx

    if np.any(mismatch_mask):
        mismatch = float(np.mean(mismatch_mask))

        if reference_distance_matrix is None:
            raise AssertionError(
                f"{label}: BMU mismatch fraction={mismatch:.6f}"
            )

        sample_idx = np.arange(result_idx.size)
        minimum_distance = np.nanmin(reference_distance_matrix, axis=0)
        selected_distance = reference_distance_matrix[result_idx, sample_idx]
        equivalent = np.isclose(
            selected_distance,
            minimum_distance,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        invalid = mismatch_mask & ~equivalent

        if np.any(invalid):
            excess = selected_distance[invalid] - minimum_distance[invalid]
            raise AssertionError(
                f"{label}: BMU mismatch fraction={mismatch:.6f}; "
                f"non-equivalent mismatch fraction={np.mean(invalid):.6f}; "
                f"maximum excess reference distance={np.nanmax(excess):.8g}"
            )

        print(
            f"      aviso: {np.sum(mismatch_mask)}/{result_idx.size} índices BMU "
            f"diferiram, mas todos eram empates numéricos equivalentes "
            f"(fração={mismatch:.6f}).",
            flush=True,
        )

    if not np.allclose(
        reference[1], result[1], rtol=rtol, atol=atol, equal_nan=True
    ):
        difference = np.nanmax(np.abs(reference[1] - result[1]))
        raise AssertionError(f"{label}: selected-distance mismatch={difference}")


@dataclass(frozen=True)
class ModeConfig:
    n_nodes: int
    n_samples: int
    n_features: int
    blocks: tuple[int, ...]


CONFIGS = {
    "quick": ModeConfig(256, 3000, 8, (128, 512, 2000)),
    "full": ModeConfig(2500, 20000, 8, (256, 2000, 10000)),
}


def run_suite(SOM, mode: str, repeats: int, seed: int, *, verbose: bool = True) -> pd.DataFrame:
    config = CONFIGS[mode]
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    report = ProgressReporter(verbose)
    report.section(
        f"BMU performance suite | mode={mode} | nodes={config.n_nodes} | "
        f"samples={config.n_samples} | features={config.n_features} | repeats={repeats}"
    )

    for dtype in (np.float32, np.float64):
        dtype_name = np.dtype(dtype).name
        report.section(f"Tipo numérico: {dtype_name}")
        codebook = rng.normal(
            size=(config.n_nodes, config.n_features)
        ).astype(dtype)
        som = make_som(
            SOM, codebook, missing=True, actual_train="Rough",
            previous_epoch=True,
        )

        report.line("\n[1/6] Kernels de distância com dados faltantes")
        for pattern in ("repeated", "random", "high_missing"):
            report.line(f"\n  Condição NaN: {pattern}")
            data = create_missing_data(
                rng, config.n_samples, config.n_features, pattern, dtype
            )

            ref_time, reference = median_time(
                lambda: nan_euclidean_distances(codebook, data), repeats,
                reporter=report, label=f"sklearn | {pattern} | {dtype_name}"
            )
            records.append({
                "category": "distance_kernel",
                "condition": pattern,
                "dtype": np.dtype(dtype).name,
                "method": "sklearn",
                "block_size": np.nan,
                "seconds": ref_time,
                "speedup_vs_sklearn": 1.0,
            })

            for strategy in ("vectorized", "grouped", "auto"):
                elapsed, result = median_time(
                    lambda s=strategy: som._nan_euclidean_distances_fast(
                        codebook, data, strategy=s
                    ),
                    repeats,
                    reporter=report,
                    label=f"{strategy} | {pattern} | {dtype_name}",
                )
                assert_distance_equivalence(
                    reference, result, dtype,
                    f"{pattern}/{np.dtype(dtype).name}/{strategy}",
                )
                records.append({
                    "category": "distance_kernel",
                    "condition": pattern,
                    "dtype": np.dtype(dtype).name,
                    "method": strategy,
                    "block_size": np.nan,
                    "seconds": elapsed,
                    "speedup_vs_sklearn": ref_time / elapsed,
                })

        report.line("\n[2/6] BMU com dados completos e diferentes blocos")
        # Complete-data BMU reference uses the historical partial squared
        # distance (the sample norm is omitted because it is neuron-invariant).
        complete = rng.normal(
            size=(config.n_samples, config.n_features)
        ).astype(dtype)
        som_complete = make_som(
            SOM, codebook, missing=False, actual_train="Rough"
        )
        y2 = np.einsum("ij,ij->i", codebook, codebook)
        full_dist = codebook @ complete.T
        full_dist *= -2.0
        full_dist += y2[:, None]
        idx = np.argmin(full_dist, axis=0)
        ref_complete = np.vstack((idx, full_dist[idx, np.arange(idx.size)]))

        for block in config.blocks:
            elapsed, result = median_time(
                lambda b=block: som_complete._find_bmu(
                    complete,
                    nth=1,
                    pace_size=b,
                    max_distance_memory_mb=256,
                ),
                repeats,
                reporter=report,
                label=f"complete | block={block} | {dtype_name}",
            )
            assert_bmu_equivalence(ref_complete, result, f"complete/block={block}")
            records.append({
                "category": "bmu",
                "condition": "complete",
                "dtype": np.dtype(dtype).name,
                "method": "optimized",
                "block_size": block,
                "seconds": elapsed,
                "speedup_vs_sklearn": np.nan,
            })

        report.line("\n[3/6] Rough training com NaN: estratégias e blocos")
        # Rough training with missing data: compare every strategy to sklearn.
        missing_data = create_missing_data(
            rng, config.n_samples, config.n_features, "repeated", dtype
        )
        ref_dist = nan_euclidean_distances(codebook, missing_data)
        ref_idx = np.argmin(ref_dist, axis=0)
        ref_missing = np.vstack(
            (ref_idx, ref_dist[ref_idx, np.arange(ref_idx.size)])
        )

        for strategy in ("sklearn", "vectorized", "grouped", "auto"):
            for block in config.blocks:
                elapsed, result = median_time(
                    lambda s=strategy, b=block: som._find_bmu(
                        missing_data,
                        nth=1,
                        pace_size=b,
                        max_distance_memory_mb=256,
                        nan_distance_strategy=s,
                    ),
                    repeats,
                    reporter=report,
                    label=f"rough | {strategy} | block={block} | {dtype_name}",
                )
                assert_bmu_equivalence(
                    ref_missing, result,
                    f"rough/{strategy}/block={block}",
                )
                records.append({
                    "category": "bmu",
                    "condition": "rough_missing_repeated",
                    "dtype": np.dtype(dtype).name,
                    "method": strategy,
                    "block_size": block,
                    "seconds": elapsed,
                    "speedup_vs_sklearn": np.nan,
                })

        report.line("\n[4/6] Projeção com menos variáveis e NaN")
        # Projection with fewer variables and NaN.
        projection_features = max(2, config.n_features - 3)
        projection = create_missing_data(
            rng, config.n_samples, projection_features, "random", dtype
        )
        projection_codebook = codebook[:, :projection_features]
        projection_ref_dist = nan_euclidean_distances(
            projection_codebook, projection
        )
        projection_idx = np.argmin(projection_ref_dist, axis=0)
        projection_ref = np.vstack((
            projection_idx,
            projection_ref_dist[projection_idx, np.arange(projection_idx.size)],
        ))
        elapsed, projection_result = median_time(
            lambda: som._find_bmu(
                projection,
                project=True,
                pace_size=config.blocks[1],
                nan_distance_strategy="auto",
            ),
            repeats,
            reporter=report,
            label=f"projection | auto | {dtype_name}",
        )
        assert_bmu_equivalence(
            projection_ref,
            projection_result,
            "projection",
            reference_distance_matrix=projection_ref_dist,
        )
        records.append({
            "category": "bmu",
            "condition": "projection_missing",
            "dtype": np.dtype(dtype).name,
            "method": "auto",
            "block_size": config.blocks[1],
            "seconds": elapsed,
            "speedup_vs_sklearn": np.nan,
        })

        report.line("\n[5/6] Fine training com imputação da época anterior")
        # Fine training with previous-epoch imputation follows complete distance.
        imputed = np.nan_to_num(missing_data, nan=0.0)
        som_fine = make_som(
            SOM, codebook, missing=True, actual_train="Fine",
            previous_epoch=True,
        )
        fine_dist = codebook @ imputed.T
        fine_dist *= -2.0
        fine_dist += y2[:, None]
        fine_idx = np.argmin(fine_dist, axis=0)
        fine_ref = np.vstack(
            (fine_idx, fine_dist[fine_idx, np.arange(fine_idx.size)])
        )
        elapsed, fine_result = median_time(
            lambda: som_fine._find_bmu(
                imputed, pace_size=config.blocks[1]
            ),
            repeats,
            reporter=report,
            label=f"fine previous_epoch | {dtype_name}",
        )
        assert_bmu_equivalence(fine_ref, fine_result, "fine/imputed")
        records.append({
            "category": "bmu",
            "condition": "fine_previous_epoch_imputed",
            "dtype": np.dtype(dtype).name,
            "method": "complete_path",
            "block_size": config.blocks[1],
            "seconds": elapsed,
            "speedup_vs_sklearn": np.nan,
        })

        report.line("\n[6/6] BMU1 e BMU2 em uma única passagem")
        # BMU1/BMU2 in one pass.
        top2_ref_idx = np.argpartition(ref_dist, kth=1, axis=0)[:2]
        top2_ref_values = np.take_along_axis(ref_dist, top2_ref_idx, axis=0)
        top2_order = np.argsort(top2_ref_values, axis=0)
        top2_ref_idx = np.take_along_axis(top2_ref_idx, top2_order, axis=0)
        top2_ref_values = np.take_along_axis(
            top2_ref_values, top2_order, axis=0
        )
        elapsed, top2_result = median_time(
            lambda: som._find_bmu_top2(
                missing_data,
                pace_size=config.blocks[1],
                nan_distance_strategy="auto",
            ),
            repeats,
            reporter=report,
            label=f"top2 | auto | {dtype_name}",
        )
        if not np.array_equal(top2_ref_idx, top2_result[0]):
            raise AssertionError("BMU1/BMU2 indices differ from reference")
        assert_distance_equivalence(
            top2_ref_values, top2_result[1], dtype, "top2 distances"
        )
        records.append({
            "category": "bmu",
            "condition": "top2_missing_one_pass",
            "dtype": np.dtype(dtype).name,
            "method": "auto",
            "block_size": config.blocks[1],
            "seconds": elapsed,
            "speedup_vs_sklearn": np.nan,
        })

    frame = pd.DataFrame.from_records(records)

    # Add within-condition speedups for BMU benchmark rows with sklearn entries.
    bmu_mask = frame["category"].eq("bmu")
    for keys, group in frame[bmu_mask].groupby(
        ["condition", "dtype", "block_size"], dropna=False
    ):
        ref = group.loc[group["method"].eq("sklearn"), "seconds"]
        if not ref.empty:
            frame.loc[group.index, "speedup_vs_sklearn"] = (
                float(ref.iloc[0]) / group["seconds"]
            )

    report.total()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=CONFIGS, default="quick")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--module-file", default=None)
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-test progress and print only the final table.",
    )
    parser.add_argument(
        "--csv", default="bmu_performance_results.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    print(f"Carregando SOM de: {args.module_file or 'instalação atual'}", flush=True)
    SOM = load_som_class(args.module_file)
    print(f"Classe carregada: {SOM.__module__}.{SOM.__name__}", flush=True)
    print(f"CSV de saída: {Path(args.csv).resolve()}", flush=True)
    frame = run_suite(
        SOM, args.mode, args.repeats, args.seed, verbose=not args.quiet
    )
    frame.to_csv(args.csv, index=False)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)
    summary = frame.sort_values(
        ["category", "condition", "dtype", "block_size", "seconds"]
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nALL CORRECTNESS TESTS PASSED")
    print(f"Results saved to: {Path(args.csv).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
