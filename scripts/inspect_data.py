#!/usr/bin/env python
"""Inspect the Frangieh et al. 2021 Perturb-CITE-seq dataset.

Downloads the dataset into data/ on the first run only, then prints a summary
of its contents and writes the same text to data/inspection_summary.txt.

Perturbation targets are derived from the `sgRNA` column rather than from
`perturbation_name`. In pertpy's copy, `perturbation_name` has lost the digits
1, 2 and 3 (JAK1 and JAK2 both read "JAK"; CD274 reads "CD74"). Section 3 tests
for this directly and reports both versions.

Usage: python scripts/inspect_data.py
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pertpy as pt
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_FILE = DATA_DIR / "frangieh_2021.h5ad"
SUMMARY_FILE = DATA_DIR / "inspection_summary.txt"

PAPER_CELL_COUNT = 218_331
EXPECTED_PROTEIN_COUNT = 20
LOW_CELL_THRESHOLD = 50

GENES_OF_INTEREST = [
    "B2M",
    "HLA-A",
    "HLA-B",
    "HLA-C",
    "CD274",
    "JAK1",
    "JAK2",
    "IFNGR1",
    "IFNGR2",
    "STAT1",
    "CD58",
]
# Non-targeting guide classes, named as they appear in `sgRNA`.
CONTROL_CLASSES = {"NO_SITE", "ONE_NON-GENE_SITE"}
PROTEINS_TO_FLAG = {
    "MHC-I (HLA-A)": ["HLA-A"],
    "MHC-I (HLA-B)": ["HLA-B"],
    "MHC-I (HLA-C)": ["HLA-C"],
    "PD-L1 (CD274)": ["CD274", "PD-L1"],
    "CD58": ["CD58"],
}


class Report:
    """Collects lines of output, echoing each one to stdout."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.flags: list[str] = []

    def __call__(self, text: object = "") -> None:
        text = str(text)
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self()
        self("=" * 78)
        self(title)
        self("=" * 78)

    def flag(self, message: str) -> None:
        self(f"[FLAG] {message}")
        self.flags.append(message)

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        temporary.replace(path)


def load_dataset(report: Report):
    sc.settings.datasetdir = DATA_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"scanpy.settings.datasetdir = {sc.settings.datasetdir}")
    action = "using existing file" if DATA_FILE.exists() else "downloading (about 1 GB)"
    print(f"{DATA_FILE}: {action}")
    # pertpy skips the download when the file exists and downloads via a
    # temporary .part file, so an interrupted run cannot leave a partial .h5ad.
    adata = pt.data.frangieh_2021()
    report(f"Data file: {DATA_FILE} ({DATA_FILE.stat().st_size / 1e9:.2f} GB)")
    return adata


def report_structure(adata, report: Report) -> None:
    report.section("1. DATASET STRUCTURE")
    report(f"Cells: {adata.n_obs:,}")
    report(f"Genes: {adata.n_vars:,}")
    if adata.n_obs != PAPER_CELL_COUNT:
        report.flag(
            f"{adata.n_obs:,} cells here vs {PAPER_CELL_COUNT:,} reported in the "
            f"paper ({adata.n_obs / PAPER_CELL_COUNT:.0%}); this is a subset."
        )
    report()
    report(adata)

    report()
    report("obs columns:")
    obs_table = pd.DataFrame(
        {
            "dtype": adata.obs.dtypes.astype(str),
            "n_unique": adata.obs.nunique(dropna=False),
            "n_missing": adata.obs.isna().sum(),
        }
    )
    report(obs_table.to_string())

    report()
    report(f"var columns: {list(adata.var.columns)}")
    report(f"var index name: {adata.var.index.name}")
    report(f"First var_names: {list(adata.var_names[:5])}")
    report(f"Duplicated var_names: {int(adata.var_names.duplicated().sum())}")

    report()
    report("obsm:")
    for key in adata.obsm:
        value = adata.obsm[key]
        report(f"  {key}: shape={value.shape}, dtype={getattr(value, 'dtype', 'mixed')}")

    report()
    report(f"layers: {list(adata.layers.keys()) or 'none'}")
    report(f"raw: {'present' if adata.raw is not None else 'none'}")
    report(f"uns keys: {list(adata.uns.keys())}")

    report()
    sample = adata.X[: min(1000, adata.n_obs)]
    sample = sample.toarray() if hasattr(sample, "toarray") else np.asarray(sample)
    totals = np.expm1(sample).sum(axis=1)
    report(f"X: {type(adata.X).__name__}, dtype={adata.X.dtype}")
    report(
        f"X value range in first {sample.shape[0]} cells: "
        f"{sample.min():.3f} to {sample.max():.3f}"
    )
    report(
        "expm1(X) per-cell totals (min/median/max): "
        f"{totals.min():,.0f} / {np.median(totals):,.0f} / {totals.max():,.0f}"
    )
    is_integer = np.allclose(sample, np.round(sample))
    if not is_integer and not adata.layers and adata.raw is None:
        report.flag("X is normalized and the file holds no raw counts (no layers, no raw).")


def find_condition_column(obs: pd.DataFrame) -> str:
    """Find the column whose values name a control and an IFN-gamma arm."""
    for column in obs.columns:
        if obs[column].nunique() > 20:
            continue
        values = [str(value).lower() for value in obs[column].dropna().unique()]
        if any("control" in value for value in values) and any("ifn" in value for value in values):
            return column
    raise SystemExit(f"No condition column found among obs columns: {list(obs.columns)}")


def report_conditions(adata, report: Report) -> str:
    report.section("2. CONDITIONS")
    column = find_condition_column(adata.obs)
    counts = adata.obs[column].value_counts(dropna=False)
    report(f"Condition column: {column!r}")
    report(f"Unique values: {[str(value) for value in counts.index]}")
    report()
    report(counts.rename("cells").to_string())
    if len(counts) != 3:
        report.flag(f"Expected 3 conditions, found {len(counts)}.")
    return column


def describe_counts(counts: pd.Series, report: Report, label: str) -> None:
    n_low = int((counts < LOW_CELL_THRESHOLD).sum())
    report(
        f"{label}: n={len(counts):,}; cells min/median/max = "
        f"{counts.min():,} / {counts.median():,.1f} / {counts.max():,}; "
        f"<{LOW_CELL_THRESHOLD} cells: {n_low}"
    )


def report_perturbations(adata, report: Report) -> pd.Series:
    report.section("3. PERTURBATIONS")
    for column in ("sgRNA", "perturbation_name"):
        if column not in adata.obs:
            raise SystemExit(f"Expected obs column {column!r} is missing.")

    guides = adata.obs["sgRNA"].astype(str)
    curated = adata.obs["perturbation_name"].astype(str)
    if not guides.str.fullmatch(r".+_\d+").all():
        raise SystemExit("Some sgRNA values do not match the <target>_<number> format.")
    targets = guides.str.replace(r"_\d+$", "", regex=True).rename("target")
    is_control = targets.isin(CONTROL_CLASSES)

    report("Target = sgRNA with its trailing _<number> removed.")
    report()
    describe_counts(targets[~is_control].value_counts(), report, "Target genes")
    describe_counts(guides[~is_control].value_counts(), report, "Targeting sgRNAs")
    describe_counts(guides[is_control].value_counts(), report, "Control sgRNAs")
    describe_counts(curated.value_counts(), report, "As-labelled perturbation_name")
    report()
    report(f"Control cells: {int(is_control.sum()):,}")
    report(targets[is_control].value_counts().rename("cells").to_string())

    guides_per_target = guides[~is_control].groupby(targets[~is_control]).nunique()
    report()
    report(f"sgRNAs per target gene: {guides_per_target.value_counts().sort_index().to_dict()}")

    gene_counts = targets[~is_control].value_counts()
    low = gene_counts[gene_counts < LOW_CELL_THRESHOLD].sort_values()
    report()
    report(f"Target genes with <{LOW_CELL_THRESHOLD} cells: {low.to_dict()}")

    report()
    report("Integrity of perturbation_name, compared with sgRNA-derived targets:")
    expected = targets.where(~is_control)
    mismatched = expected.notna() & (expected != curated)
    report(f"  Targeting cells whose label differs: {int(mismatched.sum()):,} of {int((~is_control).sum()):,}")

    pairs = pd.DataFrame({"curated": curated[~is_control], "target": targets[~is_control]})
    per_label = pairs.groupby("curated")["target"].unique()
    merged = per_label[per_label.map(len) > 1]
    report(f"  Labels that merge more than one gene: {len(merged)}")
    for label, genes in merged.items():
        report(f"    {label}: {sorted(genes)}")
    if mismatched.any():
        stripped = targets[~is_control].map(lambda name: re.sub("[123]", "", name))
        share = (stripped == curated[~is_control]).mean()
        report(f"  Labels equal to the target with digits 1-3 removed: {share:.1%}")
        report.flag(
            "perturbation_name is corrupted for "
            f"{int(mismatched.sum()):,} cells, and {len(merged)} labels merge distinct "
            "genes. Use sgRNA-derived targets."
        )
    return targets


def normalize_protein_name(name: str) -> str:
    return re.sub(r"\s*protein$", "", str(name), flags=re.IGNORECASE).replace("_", "-").upper()


def report_proteins(adata, report: Report) -> None:
    report.section("4. SURFACE PROTEINS")
    keys = [key for key in adata.obsm if "protein" in key.lower()]
    names = adata.uns.get("protein_names")
    if not keys or names is None:
        report.flag("No surface protein matrix or names were found in this file.")
        return

    matrix = adata.obsm[keys[0]]
    names = [str(name) for name in names]
    report(f"Protein matrix: obsm[{keys[0]!r}], shape={matrix.shape}, stored dtype={matrix.dtype}")
    report(f"Protein names: uns['protein_names'], n={len(names)}")
    if matrix.shape[1] != len(names):
        report.flag(f"{matrix.shape[1]} protein columns but {len(names)} names.")
        return
    if len(names) != EXPECTED_PROTEIN_COUNT:
        report.flag(f"Expected {EXPECTED_PROTEIN_COUNT} proteins, found {len(names)}.")
    if matrix.dtype == object:
        report.flag(
            f"obsm[{keys[0]!r}] is stored as strings; "
            "convert with .astype(float) before use."
        )

    values = pd.DataFrame(matrix.astype(float), columns=names)
    report(f"Missing values after conversion to float: {int(values.isna().sum().sum())}")
    report()
    report(values.agg(["min", "median", "mean", "max"]).T.round(3).to_string())

    normalized = {normalize_protein_name(name): name for name in names}
    report()
    report("Requested markers:")
    for label, aliases in PROTEINS_TO_FLAG.items():
        matches = [normalized[alias] for alias in aliases if alias in normalized]
        report(f"  {label}: {'PRESENT as ' + repr(matches[0]) if matches else 'NOT PRESENT'}")
    if not {"HLA-B", "HLA-C"} <= normalized.keys():
        report.flag(
            "No protein is labelled HLA-B or HLA-C; MHC-I protein is represented "
            "only by 'HLA_A protein'."
        )


def report_genes(adata, targets: pd.Series, condition_column: str, report: Report) -> None:
    report.section("5. GENES OF INTEREST")
    curated = adata.obs["perturbation_name"].astype(str)
    guides = adata.obs["sgRNA"].astype(str)
    condition = adata.obs[condition_column].astype(str)
    condition_names = sorted(condition.unique())

    rows = []
    for gene in GENES_OF_INTEREST:
        is_target = targets == gene
        labels = sorted(curated[is_target].unique())
        row = {
            "gene": gene,
            "in_var_names": gene in adata.var_names,
            "is_target": bool(is_target.any()),
            "n_sgRNAs": guides[is_target].nunique(),
            "cells": int(is_target.sum()),
        }
        for name in condition_names:
            row[f"cells_{name}"] = int((is_target & (condition == name)).sum())
        row["literal_in_perturbation_name"] = gene in set(curated)
        row["curated_label"] = ",".join(labels)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("gene")
    report(table.to_string())
    for gene in table.index[~table["in_var_names"]]:
        report.flag(f"{gene} is not in var_names.")
    for gene in table.index[~table["is_target"]]:
        report.flag(f"{gene} is not a perturbation target.")
    wrong = table.index[table["is_target"] & ~table["literal_in_perturbation_name"]]
    if len(wrong):
        report.flag(f"Not findable by name in perturbation_name: {list(wrong)}")


def main() -> None:
    report = Report()
    report("TMEsim data inspection: Frangieh et al. 2021 Perturb-CITE-seq")
    versions = {name: importlib.metadata.version(name) for name in ("pertpy", "scanpy", "anndata")}
    report(f"Python {sys.version.split()[0]}; {versions}")

    adata = load_dataset(report)
    report_structure(adata, report)
    condition_column = report_conditions(adata, report)
    targets = report_perturbations(adata, report)
    report_proteins(adata, report)
    report_genes(adata, targets, condition_column, report)

    report.section("FLAGS")
    for message in report.flags:
        report(f"- {message}")
    if not report.flags:
        report("None.")

    report.save(SUMMARY_FILE)
    print(f"\nWrote {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
