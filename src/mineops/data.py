"""Data sources, preparation and loading.

Two public sources:
* the iron ore flotation plant dataset on OpenML (dataset 43311, CC0);
* the United States Mine Safety and Health Administration (MSHA) Open Government Data, filtered to gold ore mines.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OPENML_URL = "https://data.openml.org/datasets/0004/43311/dataset_43311.pq"
MSHA_BASE = "https://arlweb.msha.gov/OpenGovernmentData/DataSets/"

# name -> (url, file name, SHA-256 of the copy used for the published results)
# MSHA refreshes its files every Friday, so a fresh download will usually differ from the snapshot below.
SOURCES = {
    "flotation": (OPENML_URL, "dataset_43311.pq", "2f0675981a04e2747cfce0d5c16ea56d3f1d14817fcfcd7a96c469a020246f31"),
    "mines": (MSHA_BASE + "Mines.zip", "Mines.zip", "17464dfd7d5ac1128582a38754a298aa5ec3b99d6dd6baad2d238c777283ad04"),
    "prod_yearly": (MSHA_BASE + "MinesProdYearly.zip", "MinesProdYearly.zip", "7fab22f337f183bed6084c91864ce23dc75bd197adadc736a013fa1b3967e666"),
    "accidents": (MSHA_BASE + "Accidents.zip", "Accidents.zip", "03bda6aa8ab3c12e54db60cbf0c206f58c407ef72330f7bed3130f2702273791"),
    "inspections": (MSHA_BASE + "Inspections.zip", "Inspections.zip", "c44f2e2328d06b67b6b78cfcbdce1eb5d0e88983ec4190bf678ffbb4c2569e21"),
    "violations": (MSHA_BASE + "Violations.zip", "Violations.zip", "2a3f531ce46baf9115a4934bca151bfe6465e4482304cb7001980e9ea8f78c8b"),
}
SNAPSHOT_DATE = "2026-10-27"   # day the MSHA files above were downloaded
GOLD_SIC = "104100"            # MSHA primary SIC code for gold ore mines

# ---------------------------------------------------------------- flotation plant
FLOTATION_COLUMNS = {
    "%_Iron_Feed": "iron_feed",
    "%_Silica_Feed": "silica_feed",
    "Starch_Flow": "starch_flow",
    "Amina_Flow": "amina_flow",
    "Ore_Pulp_Flow": "pulp_flow",
    "Ore_Pulp_pH": "pulp_ph",
    "Ore_Pulp_Density": "pulp_density",
    **{f"Flotation_Column_0{i}_Air_Flow": f"air_flow_{i}" for i in range(1, 8)},
    **{f"Flotation_Column_0{i}_Level": f"level_{i}" for i in range(1, 8)},
    "%_Iron_Concentrate": "iron_conc",
    "%_Silica_Concentrate": "silica_conc",
}
FEED = ["iron_feed", "silica_feed"]
REAGENTS = ["starch_flow", "amina_flow"]
PULP = ["pulp_flow", "pulp_ph", "pulp_density"]
AIR = [f"air_flow_{i}" for i in range(1, 8)]
LEVELS = [f"level_{i}" for i in range(1, 8)]
PROCESS = REAGENTS + PULP + AIR + LEVELS          # online process sensors (20-second readings)
LAB = ["iron_conc", "silica_conc"]                 # hourly laboratory assays of the concentrate

LABELS = {
    "iron_feed": "Iron in feed (%)", "silica_feed": "Silica in feed (%)",
    "starch_flow": "Starch flow", "amina_flow": "Amine flow", "pulp_flow": "Ore pulp flow",
    "pulp_ph": "Ore pulp pH", "pulp_density": "Ore pulp density",
    **{f"air_flow_{i}": f"Column {i} air flow" for i in range(1, 8)},
    **{f"level_{i}": f"Column {i} level" for i in range(1, 8)},
    "iron_conc": "Iron in concentrate (%)", "silica_conc": "Silica in concentrate (%)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def flotation_to_numeric(raw: pd.DataFrame) -> pd.DataFrame:
    """The OpenML copy stores every value as text with a decimal comma. Returns numbers with short column names."""
    out = pd.DataFrame({"time": pd.to_datetime(raw["date"])})
    for src, dst in FLOTATION_COLUMNS.items():
        out[dst] = pd.to_numeric(raw[src].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return out


def prepare_flotation(raw_dir: Path, out_dir: Path) -> Path:
    raw = pd.read_parquet(raw_dir / SOURCES["flotation"][1])
    df = flotation_to_numeric(raw)
    out = out_dir / "flotation.parquet"
    df.to_parquet(out, index=False)
    return out


def load_flotation(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / "flotation.parquet")


# ---------------------------------------------------------------- MSHA gold mines
VIOLATION_COLUMNS = ["EVENT_NO", "VIOLATION_NO", "VIOLATOR_TYPE_CD", "MINE_ID", "MINE_TYPE", "CONTRACTOR_ID",
                     "VIOLATION_ISSUE_DT", "VIOLATION_OCCUR_DT", "CAL_YR", "SIG_SUB", "PART_SECTION", "CIT_ORD_SAFE",
                     "LIKELIHOOD", "INJ_ILLNESS", "NO_AFFECTED", "NEGLIGENCE", "PRIMARY_OR_MILL", "PROPOSED_PENALTY",
                     "AMOUNT_PAID", "CONTESTED_IND"]


def _read_msha(zip_path: Path, mine_ids: set | None = None, usecols=None, chunksize: int = 400_000) -> pd.DataFrame:
    """MSHA files are pipe-delimited, Latin-1 text. The violations file unzips to about 1.4 GB, so it is streamed."""
    parts = []
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            reader = pd.read_csv(f, sep="|", encoding="latin-1", dtype=str, usecols=usecols,
                                 chunksize=chunksize, low_memory=False, on_bad_lines="warn")
            for chunk in reader:
                chunk.columns = [c.strip() for c in chunk.columns]
                if mine_ids is not None:
                    chunk = chunk[chunk["MINE_ID"].isin(mine_ids)]
                parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def prepare_gold(raw_dir: Path, out_dir: Path) -> dict:
    """Extract every record for mines whose primary SIC is gold ore."""
    gold = out_dir / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    mines = _read_msha(raw_dir / SOURCES["mines"][1])
    mines = mines[mines["PRIMARY_SIC_CD"].str.strip() == GOLD_SIC].copy()
    ids = set(mines["MINE_ID"])
    counts = {"mines": len(mines)}
    mines.to_parquet(gold / "mines.parquet", index=False)
    for key, cols in [("prod_yearly", None), ("accidents", None), ("inspections", None), ("violations", VIOLATION_COLUMNS)]:
        df = _read_msha(raw_dir / SOURCES[key][1], ids, usecols=cols)
        df.to_parquet(gold / f"{key}.parquet", index=False)
        counts[key] = len(df)
    return counts


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_gold(data_dir: Path) -> dict:
    """Gold mine tables with numeric and date fields parsed."""
    g = Path(data_dir) / "gold"
    mines = pd.read_parquet(g / "mines.parquet")
    prod = pd.read_parquet(g / "prod_yearly.parquet")
    prod["year"] = num(prod["CALENDAR_YR"]).astype("Int64")
    prod["hours"] = num(prod["ANNUAL_HRS"]).fillna(0.0)
    prod["employees"] = num(prod["AVG_ANNUAL_EMPL"]).fillna(0.0)
    prod["subunit_cd"] = prod["SUBUNIT_CD"].str.strip()

    acc = pd.read_parquet(g / "accidents.parquet")
    acc["year"] = num(acc["CAL_YR"]).astype("Int64")
    acc["date"] = pd.to_datetime(acc["ACCIDENT_DT"], format="%m/%d/%Y", errors="coerce")
    acc["degree"] = num(acc["DEGREE_INJURY_CD"])
    acc["contractor"] = acc["CONTRACTOR_ID"].notna() & (acc["CONTRACTOR_ID"].str.strip() != "")
    for c in ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "DAYS_LOST", "DAYS_RESTRICT", "NO_INJURIES"]:
        acc[c.lower()] = num(acc[c])
    acc["subunit_cd"] = acc["SUBUNIT_CD"].str.strip()

    ins = pd.read_parquet(g / "inspections.parquet")
    ins["begin"] = pd.to_datetime(ins["INSPECTION_BEGIN_DT"], format="%m/%d/%Y", errors="coerce")
    ins["year"] = ins["begin"].dt.year.astype("Int64")
    ins["insp_hours"] = num(ins["SUM(TOTAL_INSP_HOURS)"]).fillna(0.0)
    ins["onsite_hours"] = num(ins["SUM(TOTAL_ON_SITE_HOURS)"]).fillna(0.0)

    vio = pd.read_parquet(g / "violations.parquet")
    vio["year"] = num(vio["CAL_YR"]).astype("Int64")
    vio["ss"] = vio["SIG_SUB"].str.strip().eq("Y")
    vio["order"] = vio["CIT_ORD_SAFE"].str.strip().eq("Order")
    vio["high_negligence"] = vio["NEGLIGENCE"].str.strip().isin(["HighNegligence", "Reckless"])
    vio["penalty"] = num(vio["PROPOSED_PENALTY"]).fillna(0.0)
    vio["contractor"] = vio["VIOLATOR_TYPE_CD"].str.strip().eq("Contractor")
    return {"mines": mines, "prod": prod, "accidents": acc, "inspections": ins, "violations": vio}


def finite(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
