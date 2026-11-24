# Loaders for the Frangieh et al. 2021 data. All paths resolve from the repo root,
# so scripts and notebooks work from any working directory.
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCP = DATA / "scp"
SUPP_TABLES = DATA / "paper" / "frangieh_2021_supp_tables.xlsx"

CONTROL_TARGETS = {"NO_SITE", "ONE_NON-GENE_SITE"}  # non-targeting guide families


def guide_target(sgrna: str) -> str:
    # "CD274_2" -> "CD274". pertpy's perturbation_name column strips digits, so derive from the guide.
    return re.sub(r"_\d+$", "", sgrna)


def load_adata():
    # pertpy processed AnnData: 126,966 single-guide cells x 23,712 genes, X = ln(1+CPM).
    # Downloaded once (~1 GB) to data/frangieh_2021.h5ad.
    import pertpy as pt
    import scanpy as sc

    sc.settings.datasetdir = DATA
    return pt.data.frangieh_2021()


def load_metadata() -> pd.DataFrame:
    # SCP RNA_metadata.csv: one row per cell (218,331), indexed by cell name.
    # Columns: condition, MOI (guides per cell), sgRNA (empty unless MOI == 1), UMI_count.
    # Row 2 of the file is an SCP type annotation ("TYPE,group,..."), not a cell.
    return pd.read_csv(SCP / "RNA_metadata.csv", skiprows=[1], index_col="NAME")


def load_supp_table(sheet: str) -> pd.DataFrame:
    # One sheet of the paper's supplementary workbook, e.g. "2j".
    # Each sheet starts with title rows; the header row is the first whose first cell is a column name.
    raw = pd.read_excel(SUPP_TABLES, sheet_name=f"Supplementary Table {sheet}", header=None)
    header = raw.index[raw[0].isin(["id", "sgrna", "Guide Name"])][0]
    df = raw.iloc[header + 1 :].reset_index(drop=True)
    df.columns = raw.iloc[header]
    return df.dropna(axis=1, how="all").infer_objects()
