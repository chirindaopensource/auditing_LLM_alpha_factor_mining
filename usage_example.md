## **Usage Example**

Here is the granular, step-by-step guide to executing the end-to-end pipeline for **"Auditing LLM Alpha-Factor Mining: Metric Provenance and Full-Cohort Reanalysis"**. This example shows how to load the study configuration from `config.yaml`, synthetically generate every input artifact the study consumes, and execute the full pipeline with the top-level orchestrator, `run_study` (Task 14).

**Assumptions.**

- **Single notebook.** Every callable defined in this project (Tasks 1 to 16, including `run_study`, `STUDY_CONFIG`, `STAGE_ORDER` and the Task 7 parser) lives in the one Jupyter notebook, `auditing_LLM_alpha_factor_mining_draft.ipynb`, and has been executed in the current kernel. There is no folder of `.py` modules; nothing below is imported from a package. The cells below are run in that same notebook.
- **Working directory.** `config.yaml` and `study_revised_inputs_20261111_143538.md` sit in the working directory (the repository root). Task 8 reads the second file at run time to look up the paper's comparison values, and halts with `DATA_ABSENT` without it. The example writes the 80 synthetic data files under `data/raw/` in the working directory.
- **Synthetic data.** The study's real inputs (the BaoStock snapshot and the preserved project artifacts) are not redistributable, so every frame below is generated. The figures this run prints are therefore properties of the synthetic corpus, not reproductions of the paper's numbers; the paper's numbers stay comparison values in the configuration.
- **Runtime.** On the full synthetic corpus (40 stocks, 727 sessions, 1,047 expressions) `run_study` took **92 minutes** on a desktop CPU. Daily scoring (about 47 minutes) and the v2 re-evaluation (about 38 minutes) dominate.

The inputs of `run_study` are:

| Parameter | Type | Content (this example) |
|---|---|---|
| `files` | `pd.DataFrame` | `FILES`: the 80-row acquisition manifest (40 stocks × 2 views) with its 12 header `attrs` |
| `unadjusted` | `pd.DataFrame` | `UNADJUSTED_BARS`: 29,080 nominal bars, the feature substrate |
| `adjusted` | `pd.DataFrame` | `ADJUSTED_BARS`: 29,080 provider-adjusted bars, the label substrate only |
| `preserved_rows` | `pd.DataFrame` | `PRESERVED_INPUT_ROWS`: 3,114 preserved expressions with IS metrics |
| `native_oos_checkpoint` | `pd.DataFrame` | 2,016 checkpoint rows, 368 with negative OOS Sharpe |
| `outcome_filtered_catalog` | `pd.DataFrame` | 1,837 catalog rows, all with positive OOS Sharpe |
| `run_report_totals` | `pd.DataFrame` | The three Table 1 rows (P, S-A, S-B) |
| `run_report_rounds` | `pd.DataFrame` | 60 rows: 20 rounds per run |
| `collector_a` | `pd.DataFrame` | 67 kept-candidate Sharpes of the pipeline |
| `collector_b` | `pd.DataFrame` | 40 positive per-round maxima of the swarm |
| `config` | `dict` | `STUDY_CONFIG`, read from `config.yaml` |
| `stages` | `Sequence[str]` | All thirteen stages, `list(STAGE_ORDER)` |
| `halt_on_failure` | `bool` | `True` (production mode: stop at the first halt) |

### **Step 1: Loading the Configuration (`config.yaml`)**

The configuration is loaded first because every generator below reads its pins from it (the 727-session calendar, the 3,114-row source, the IS thresholds, the retention counts and each frame's column schema), and because `FILES.attrs["config_self_hash"]` must be the SHA-256 of the exact configuration passed to `run_study`.

**Methodology:**
1.  **Parsing:** `yaml.safe_load` reads the file. The YAML keeps the integer year keys and the one set-valued closure (`!!set`) exactly as in the notebook, so the loaded dictionary equals the notebook's literal leaf for leaf.
2.  **Provenance tags:** the notebook assigns `STUDY_CONFIG["_provenance_tags"]` after the literal. `config.yaml` carries those tags only as comments, so the loader re-attaches the dictionary; Task 1 hashes the whole configuration, tags included.

```python
import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_study_configuration(filepath: str = "config.yaml") -> Dict[str, Any]:
    """
    Load ``config.yaml`` into the ``config`` dictionary used by ``run_study``.

    Purpose:
        Read the study parameters (sections P00 to P45, the raw input
        schemas and the verification targets) from the YAML file in
        the working directory.

    Inputs:
        filepath (str): Path of the YAML file. Default ``"config.yaml"``.

    Processes:
        1. Parse the file with ``yaml.safe_load``; YAML keeps the
           integer year keys and the ``!!set`` closure as Python ints
           and a set, exactly as in the notebook.
        2. Unwrap the top-level ``STUDY_CONFIG`` key.
        3. Re-attach the parallel ``_provenance_tags`` dictionary. It
           is data in the notebook (assigned after the literal), the
           YAML carries it only as comments, and Task 1 hashes the
           whole configuration including it.

    Outputs:
        Dict[str, Any]: The configuration ``run_study`` expects.

    Raises:
        TypeError: If ``filepath`` is not a string.
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not hold a ``STUDY_CONFIG`` mapping.
    """
    # Validate the argument type.
    if not isinstance(filepath, str):
        raise TypeError(
            f"filepath must be a str, got {type(filepath).__name__}"
        )
    # Refuse a missing file instead of continuing with an empty config.
    if not Path(filepath).is_file():
        raise FileNotFoundError(f"{filepath} not found in {Path.cwd()}")
    # Parse the YAML document safely.
    with open(filepath, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    # The document must be a mapping holding STUDY_CONFIG.
    if not isinstance(document, dict) or "STUDY_CONFIG" not in document:
        raise ValueError(f"{filepath} does not hold a STUDY_CONFIG mapping")
    # Unwrap the configuration dictionary.
    loaded: Dict[str, Any] = document["STUDY_CONFIG"]
    # Re-attach the notebook's provenance-tag dictionary (data, not comments).
    loaded["_provenance_tags"] = copy.deepcopy(
        STUDY_CONFIG["_provenance_tags"]
    )
    # Report what was loaded.
    print(f"Loaded {len(loaded)} configuration sections from {filepath}")
    # Return the configuration.
    return loaded


# Load the configuration once; every later step reads its pins from it.
config: Dict[str, Any] = load_study_configuration("config.yaml")
```

### **Step 2: Synthetic Market Snapshot (`FILES`, `UNADJUSTED_BARS`, `ADJUSTED_BARS`)**

The snapshot mirrors Section 6: "the same ordered 40-stock convenience universe over 727 trading dates in 2023--2025", acquired as "unadjusted and provider-adjusted daily views through BaoStock 0.9.3", with "Two non-trading stock-day cells" masked.

**Methodology:**
1.  **Calendar:** weekdays of 2023 to 2025 minus the Shanghai and Shenzhen exchange closures give 242 + 242 + 243 = 727 sessions, checked against `P11.calendar_year_row_counts`.
2.  **Prices:** nominal closes follow a one-factor geometric Brownian motion; open, high and low bracket the close; volume is an integral share count and `amount` a nominal traded value. Features use only these nominal fields.
3.  **Adjustment:** each stock has one ex-dividend date; the adjusted view is the nominal view times a piecewise-constant factor, so $P_{i,t}^{\mathrm{adj}} / P_{i,t}^{\mathrm{nom}}$ is constant on each segment (the Task 4 constancy test).
4.  **Masking:** the two masked cells carry vendor status 0, `tradable == False` and null (never zero) prices and counts.
5.  **Thin spell:** on sessions 400 to 406 (late August 2024) 35 of the 40 stocks close unchanged, as in a thin, limit-locked market. This gives the nine-cell check of Task 12 the cells its rules require: a date with exactly five eligible stocks (rule r1), a three-value signal (r2) and tied signal and label vectors (r5 to r8). Without such cells the `dual_check` stage halts.
6.  **Custody:** every stock-view slice is written to `data/raw/<sha256>`, its content address, which is the `file_uri` that Task 1 derives and Task 3 rehashes.

```python
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Fix the generator so that every run produces byte-identical files.
RNG: np.random.Generator = np.random.default_rng(20261113)

# Weekday exchange closures of the Shanghai and Shenzhen exchanges.
SSE_CLOSURES: Tuple[str, ...] = (
    "2023-01-02",
    "2023-01-23",
    "2023-01-24",
    "2023-01-25",
    "2023-01-26",
    "2023-01-27",
    "2023-04-05",
    "2023-05-01",
    "2023-05-02",
    "2023-05-03",
    "2023-06-22",
    "2023-06-23",
    "2023-09-29",
    "2023-10-02",
    "2023-10-03",
    "2023-10-04",
    "2023-10-05",
    "2023-10-06",
    "2024-01-01",
    "2024-02-09",
    "2024-02-12",
    "2024-02-13",
    "2024-02-14",
    "2024-02-15",
    "2024-02-16",
    "2024-04-04",
    "2024-04-05",
    "2024-05-01",
    "2024-05-02",
    "2024-05-03",
    "2024-06-10",
    "2024-09-16",
    "2024-09-17",
    "2024-10-01",
    "2024-10-02",
    "2024-10-03",
    "2024-10-04",
    "2024-10-07",
    "2025-01-01",
    "2025-01-28",
    "2025-01-29",
    "2025-01-30",
    "2025-01-31",
    "2025-02-03",
    "2025-02-04",
    "2025-04-04",
    "2025-05-01",
    "2025-05-02",
    "2025-05-05",
    "2025-06-02",
    "2025-10-01",
    "2025-10-02",
    "2025-10-03",
    "2025-10-06",
    "2025-10-07",
    "2025-10-08",
)

# A thin, limit-locked spell: 35 of the 40 stocks close unchanged on
# sessions 400-406 (mid-2024). It gives the nine-cell check of Task 12 its
# small-cross-section and tie witnesses (rules r1, r2 and r5-r8).
FLAT_WINDOW: Tuple[int, int] = (399, 406)
FLAT_STOCK_COUNT: int = 35

# The two non-trading stock-day cells that the study masks (P12, P34).
MASKED_CELLS: Tuple[Tuple[str, str], ...] = (
    ("sh.600519", "2024-03-14"),
    ("sz.000858", "2025-06-18"),
)


def build_trading_calendar(config: Dict[str, Any]) -> pd.DatetimeIndex:
    """
    Build the 727-session A-share trading calendar for 2023 to 2025.

    Purpose:
        Reproduce the paper's "727 trading dates in 2023--2025" as a
        dense session axis: weekdays minus the exchange closures.

    Inputs:
        config (Dict[str, Any]): The study configuration; supplies
            ``P11_ACQUISITION_PARAMETERS.calendar_year_row_counts``
            and ``trading_date_count``.

    Processes:
        1. Enumerate the weekdays of 2023-01-01 to 2025-12-31.
        2. Remove the weekday exchange closures.
        3. Check the per-year and total session counts against the
           configuration.

    Outputs:
        pd.DatetimeIndex: 727 naive, strictly increasing sessions.

    Raises:
        KeyError: If the acquisition section is missing.
        ValueError: If a session count disagrees with the pins.
    """
    # Read the acquisition section that pins the calendar.
    p11: Dict[str, Any] = config["P11_ACQUISITION_PARAMETERS"]
    # Enumerate every weekday in the three study years.
    weekdays: pd.DatetimeIndex = pd.bdate_range("2023-01-01", "2025-12-31")
    # Remove the weekday exchange closures.
    sessions: pd.DatetimeIndex = weekdays.difference(
        pd.DatetimeIndex(SSE_CLOSURES)
    )
    # Check each year's session count against its pin (242, 242, 243).
    for year, pinned in p11["calendar_year_row_counts"].items():
        # Count this year's sessions.
        observed: int = int((sessions.year == int(year)).sum())
        # Halt on any disagreement rather than adjusting the calendar.
        if observed != int(pinned):
            raise ValueError(f"{year}: {observed} sessions, pinned {pinned}")
    # Check the total against the pinned 727 sessions.
    if len(sessions) != int(p11["trading_date_count"]):
        raise ValueError(
            f"{len(sessions)} sessions, pinned {p11['trading_date_count']}"
        )
    # Return the dense session axis.
    return sessions


def build_universe(size: int = 40) -> List[str]:
    """
    Build the ordered 40-stock convenience universe.

    Purpose:
        The paper does not disclose its 40 codes, so a documented
        convenience sample of BaoStock-style codes is used; every
        downstream count is therefore a reproduction, not a replica.

    Inputs:
        size (int): Universe size. Default is 40 (the pinned value).

    Processes:
        1. Take 20 Shanghai ("sh.") and 20 Shenzhen ("sz.") codes.
        2. Keep the fixed order: it is the ``stock_rank`` order.

    Outputs:
        List[str]: ``size`` exchange-prefixed six-digit codes.

    Raises:
        ValueError: If ``size`` is not 40.
    """
    # Only the pinned universe size is supported.
    if size != 40:
        raise ValueError("the study pins a 40-stock universe")
    # Twenty large Shanghai main-board codes, including the masked one.
    shanghai: List[str] = [
        "sh.600000",
        "sh.600009",
        "sh.600016",
        "sh.600028",
        "sh.600030",
        "sh.600036",
        "sh.600048",
        "sh.600050",
        "sh.600104",
        "sh.600196",
        "sh.600276",
        "sh.600309",
        "sh.600519",
        "sh.600585",
        "sh.600690",
        "sh.600887",
        "sh.600900",
        "sh.601088",
        "sh.601166",
        "sh.601318",
    ]
    # Twenty large Shenzhen main-board codes, including the masked one.
    shenzhen: List[str] = [
        "sz.000001",
        "sz.000002",
        "sz.000063",
        "sz.000069",
        "sz.000100",
        "sz.000157",
        "sz.000166",
        "sz.000333",
        "sz.000338",
        "sz.000425",
        "sz.000538",
        "sz.000568",
        "sz.000625",
        "sz.000651",
        "sz.000725",
        "sz.000768",
        "sz.000776",
        "sz.000858",
        "sz.000895",
        "sz.000938",
    ]
    # Return the fixed ordering.
    return shanghai + shenzhen


def simulate_bars(
    stock_codes: List[str], sessions: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate the unadjusted and provider-adjusted daily bars.

    Purpose:
        Produce the two BaoStock views of the same 40 x 727 grid, with
        nominal prices for features and adjusted prices for labels.

    Inputs:
        stock_codes (List[str]): The ordered universe.
        sessions (pd.DatetimeIndex): The trading calendar.

    Processes:
        1. Simulate nominal closes as geometric Brownian motion with a
           cross-sectional market factor, plus open, high and low.
        2. Simulate integral share volume and nominal traded amount.
        3. Give each stock a piecewise-constant adjustment factor that
           jumps at one ex-dividend date, so that
           P_adj / P_nom is constant on each segment.
        4. Mark the two masked cells as non-trading: tradestatus 0,
           tradable False, prices and quantities null (never zero).

    Outputs:
        Tuple[pd.DataFrame, pd.DataFrame]: ``(unadjusted, adjusted)``
            flat frames, one row per stock-session, rows ordered by
            stock rank then date. ``byte_hash`` is filled later.

    Raises:
        ValueError: If the inputs are empty.
    """
    # Refuse an empty grid.
    if not stock_codes or len(sessions) == 0:
        raise ValueError("empty universe or calendar")
    # Number of sessions per stock.
    n_dates: int = len(sessions)
    # Daily market factor: mean 0.03 percent, volatility 1.2 percent.
    market: np.ndarray = RNG.normal(0.0003, 0.012, n_dates)
    # Collect the per-stock frames of each view.
    unadj_parts: List[pd.DataFrame] = []
    adj_parts: List[pd.DataFrame] = []
    # The masked coordinates as timestamps, for fast lookup.
    masked = {(c, pd.Timestamp(d)) for c, d in MASKED_CELLS}
    # Simulate each stock in rank order.
    for code in stock_codes:
        # Stock-specific beta and idiosyncratic volatility.
        beta: float = RNG.uniform(0.7, 1.3)
        # Idiosyncratic daily volatility between 1 and 2.5 percent.
        sigma: float = RNG.uniform(0.010, 0.025)
        # Log return: r = beta * m + e.
        log_ret: np.ndarray = beta * market + RNG.normal(0.0, sigma, n_dates)
        # Starting price between 5 and 150 RMB.
        start: float = RNG.uniform(5.0, 150.0)
        # Nominal close: P_t = P_0 * exp(sum of log returns).
        close: np.ndarray = np.round(start * np.exp(np.cumsum(log_ret)), 2)
        # Open: the close moved by an overnight shock.
        open_: np.ndarray = np.round(
            close * np.exp(RNG.normal(0, 0.004, n_dates)), 2
        )
        # High: at least the larger of open and close.
        high: np.ndarray = np.round(
            np.maximum(open_, close) * (1 + RNG.uniform(0, 0.02, n_dates)), 2
        )
        # Low: at most the smaller of open and close.
        low: np.ndarray = np.round(
            np.minimum(open_, close) * (1 - RNG.uniform(0, 0.02, n_dates)), 2
        )
        # Share volume: integral lognormal counts.
        volume: np.ndarray = np.round(np.exp(RNG.normal(16.0, 0.5, n_dates)))
        # Nominal traded amount: volume times a typical price.
        amount: np.ndarray = np.round(volume * (high + low + close) / 3.0, 2)
        # Flatten the close over the spell for the first 35 stocks.
        if stock_codes.index(code) < FLAT_STOCK_COUNT:
            # Hold the close at its value on the session before the spell.
            close[FLAT_WINDOW[0] : FLAT_WINDOW[1] + 1] = close[FLAT_WINDOW[0]]
            # A flat session trades at one price: open = high = low = close.
            for bar in (open_, high, low):
                bar[FLAT_WINDOW[0] : FLAT_WINDOW[1] + 1] = close[
                    FLAT_WINDOW[0]
                ]
            # Recompute the traded amount for the flattened sessions.
            amount = np.round(volume * (high + low + close) / 3.0, 2)
        # One ex-dividend date per stock, before the spell, so that the
        # labels on the spell are exact.
        jump: int = int(RNG.integers(60, FLAT_WINDOW[0] - 20))
        # Cumulative factor: 1.0 before, a step up from the event on.
        factor: np.ndarray = np.where(
            np.arange(n_dates) < jump, 1.0, RNG.uniform(1.01, 1.06)
        )
        # Segment ordinal of each session (1 before the event, 2 after).
        segment: np.ndarray = np.where(np.arange(n_dates) < jump, 1, 2)
        # Vendor status code: 1 means trading.
        status: np.ndarray = np.ones(n_dates, dtype=int)
        # Identify this stock's masked sessions.
        is_masked: np.ndarray = np.array(
            [(code, d) in masked for d in sessions]
        )
        # A non-trading day carries vendor status 0.
        status[is_masked] = 0
        # Build the nominal view.
        unadj = pd.DataFrame(
            {
                "stock_code": code,
                "trade_date": sessions,
                "adjustment": "unadjusted",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
                "quantity": volume,
                "tradable": status == 1,
                "tradestatus_raw": status,
            }
        )
        # Build the adjusted view: nominal prices times the factor.
        adj = pd.DataFrame(
            {
                "stock_code": code,
                "trade_date": sessions,
                "adjustment": "provider_adjusted",
                "open_adj": open_ * factor,
                "high_adj": high * factor,
                "low_adj": low * factor,
                "close_adj": close * factor,
                "adj_factor": factor,
                "adj_factor_segment_id": segment,
                "quantity": volume,
                "tradable": status == 1,
                "tradestatus_raw": status,
            }
        )
        # Null (never zero) the measures of a masked cell in both views.
        for frame, cols in (
            (
                unadj,
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "quantity",
                ],
            ),
            (
                adj,
                ["open_adj", "high_adj", "low_adj", "close_adj", "quantity"],
            ),
        ):
            # Write the missing-value marker into each measure.
            frame.loc[is_masked, cols] = np.nan
        # Keep the stock's frames.
        unadj_parts.append(unadj)
        adj_parts.append(adj)
    # Concatenate the stocks in rank order.
    unadjusted: pd.DataFrame = pd.concat(unadj_parts, ignore_index=True)
    adjusted: pd.DataFrame = pd.concat(adj_parts, ignore_index=True)
    # Return both views.
    return unadjusted, adjusted


def cast_bar_dtypes(frame: pd.DataFrame, view: str) -> pd.DataFrame:
    """
    Cast one bar view to the dtypes of ``raw_input_schemas``.

    Purpose:
        Enforce the pinned schema: string identities, naive dates,
        float64 prices, nullable Int64 counts and nullable booleans.

    Inputs:
        frame (pd.DataFrame): A flat view from ``simulate_bars``.
        view (str): ``"UNADJUSTED_BARS"`` or ``"ADJUSTED_BARS"``.

    Processes:
        1. Cast every column to its pinned dtype.
        2. Index the frame on ``(stock_code, trade_date)``.

    Outputs:
        pd.DataFrame: The typed view on its compound index.

    Raises:
        ValueError: If ``view`` is unknown.
    """
    # Integer-valued columns of each view.
    ints: Dict[str, List[str]] = {
        "UNADJUSTED_BARS": ["volume", "quantity", "tradestatus_raw"],
        "ADJUSTED_BARS": [
            "adj_factor_segment_id",
            "quantity",
            "tradestatus_raw",
        ],
    }
    # Refuse an unknown view name.
    if view not in ints:
        raise ValueError(f"unknown view {view!r}")
    # Work on a copy.
    out: pd.DataFrame = frame.copy()
    # Identities stay strings, never categories (P22).
    for col in ("stock_code", "adjustment", "byte_hash"):
        # Cast to the pandas string dtype.
        out[col] = out[col].astype("string")
    # Trading dates are naive datetime64[ns].
    out["trade_date"] = pd.to_datetime(out["trade_date"]).astype(
        "datetime64[ns]"
    )
    # Counts are nullable Int64: a masked cell is <NA>, never 0.
    for col in ints[view]:
        # Cast the integral float column to nullable Int64.
        out[col] = out[col].astype("Int64")
    # The trading flag is a nullable boolean.
    out["tradable"] = out["tradable"].astype("boolean")
    # Index on the compound primary key.
    return out.set_index(["stock_code", "trade_date"])


def write_files_and_manifest(
    unadjusted: pd.DataFrame,
    adjusted: pd.DataFrame,
    config: Dict[str, Any],
    raw_dir: Path = Path("data") / "raw",
) -> pd.DataFrame:
    """
    Write the 80 per-stock files and build the ``FILES`` manifest.

    Purpose:
        Make the snapshot witnessed: every file exists on disk under
        its content address ``data/raw/<sha256>``, which is the
        ``file_uri`` that Task 1 derives and Task 3 rehashes.

    Inputs:
        unadjusted (pd.DataFrame): Typed nominal view.
        adjusted (pd.DataFrame): Typed adjusted view.
        config (Dict[str, Any]): The study configuration (hashed into
            ``attrs["config_self_hash"]``).
        raw_dir (Path): Content-addressed store. Default
            ``data/raw`` beneath the working directory.

    Processes:
        1. Serialise each stock-view slice to CSV bytes and hash them.
        2. Store the bytes at ``raw_dir/<sha256>``.
        3. Fill ``byte_hash`` back into the bar rows.
        4. Build one manifest row per file and the twelve header
           ``attrs``.

    Outputs:
        pd.DataFrame: ``FILES``, 80 rows indexed on ``file_id``. The
            two views are updated in place with their ``byte_hash``.

    Raises:
        OSError: If a file cannot be written.
    """
    # Create the content-addressed store.
    raw_dir.mkdir(parents=True, exist_ok=True)
    # The acquisition batch started at 09:00 Shanghai on 2026-03-20.
    batch_start = pd.Timestamp("2026-03-20 09:00:00", tz="Asia/Shanghai")
    # Collect one manifest row per file.
    rows: List[Dict[str, Any]] = []
    # Ordered universe, read from the index.
    codes: List[str] = list(
        dict.fromkeys(unadjusted.index.get_level_values(0))
    )
    # Walk the two views, then the stocks in rank order.
    for view_name, frame in (
        ("unadjusted", unadjusted),
        ("provider_adjusted", adjusted),
    ):
        # One file per stock in this view.
        for rank, code in enumerate(codes, start=1):
            # The stock's slice, without the not-yet-known hash column.
            part: pd.DataFrame = frame.loc[code].drop(columns=["byte_hash"])
            # Serialise deterministically (UTF-8, LF line endings).
            payload: bytes = part.to_csv(lineterminator="\n").encode("utf-8")
            # SHA-256 of the exact bytes stored.
            digest: str = hashlib.sha256(payload).hexdigest()
            # Store the file at its content address.
            (raw_dir / digest).write_bytes(payload)
            # Denormalise the digest onto the stock's bar rows.
            frame.loc[code, "byte_hash"] = digest
            # Schema fingerprint: hash of the column names and dtypes.
            fingerprint: str = hashlib.sha256(
                str([(c, str(t)) for c, t in part.dtypes.items()]).encode()
            ).hexdigest()
            # Dates observed in the file.
            dates = part.index
            # Record the manifest row.
            rows.append(
                {
                    "file_id": f"{view_name[:3]}_{code.replace('.', '_')}",
                    "stock_code": code,
                    "stock_rank": rank,
                    "adjustment": view_name,
                    "row_count": len(part),
                    "byte_hash": digest,
                    "content_length_bytes": len(payload),
                    "schema_fingerprint": fingerprint,
                    "date_min": dates.min(),
                    "date_max": dates.max(),
                    "requested_stock_code": code,
                    "requested_date_min": dates.min(),
                    "requested_date_max": dates.max(),
                    "requested_adjustment": view_name,
                    "response_truncated": False,
                    "hash_ok": pd.NA,
                    "date_id_ok": pd.NA,
                    "stock_id_ok": pd.NA,
                    "adjustment_id_ok": pd.NA,
                    "quantity_agreement_ok": pd.NA,
                    "status_agreement_ok": pd.NA,
                    "verified": pd.NA,
                    "retrieved": batch_start
                    + pd.Timedelta(seconds=7 * len(rows)),
                    "redistribution_flag": False,
                }
            )
    # Assemble the manifest.
    files: pd.DataFrame = pd.DataFrame(rows)
    # Cast every column to its pinned dtype.
    for col, dtype in config["raw_input_schemas"]["FILES"]["columns"].items():
        # Apply the declared dtype.
        files[col] = files[col].astype(dtype)
    # Index on the surrogate key.
    files = files.set_index("file_id")
    # The twelve header attributes of the acquisition.
    files.attrs = {
        "provider": "BaoStock",
        "provider_version": "0.9.3",
        "universe_label": "40-stock convenience universe",
        "universe_order_fixed": True,
        "universe_disclosed": False,
        "date_count": 727,
        "file_count": 80,
        "masked_cell_count": 2,
        "retrieval_batch_timestamp": batch_start,
        "retrieval_coherence_window": "<= 1 trading day",
        # SHA-256 of the canonical JSON of the whole configuration.
        "config_self_hash": _sha256_hex(_canonical_json_bytes(config)),
        "redistribution_grant": False,
    }
    # Return the manifest.
    return files
```

### **Step 3: Synthetic Historical Artifacts (Sections 4 to 6)**

The seven remaining frames reproduce the structures the paper audits, with the paper's stated counts built in so every closure can be exercised:

- `PRESERVED_INPUT_ROWS`: 3,114 rows. 1,047 distinct expressions pass the stated IS screen ($\text{Sharpe} \ge 2$, $\text{Fitness} \ge 0$, $\text{Turnover} \le 500$), 520 respelled duplicates collapse under canonical-AST deduplication, and 1,547 rows fail the screen. The 1,047 include ordinary expressions, degenerate signals on the `tradable` flag, 25 expressions naming operators outside the public set, and 57 conditionals `if(delta(tradable, k), 1, x)` whose panel condition with a scalar true branch is exactly the v1 defect.
- `NATIVE_OOS_CHECKPOINT` and `OUTCOME_FILTERED_CATALOG`: 2,016 and 1,837 rows with the Section 5 partition, $2016 = 368 + 1648$ and $1837 = 189 + 1648$, no sign disagreement on the intersection, and 1,037 catalog rows passing the stated IS rule.
- `RUN_REPORT_TOTALS`, `RUN_REPORT_ROUNDS`, `COLLECTOR_A` and `COLLECTOR_B`: Table 1 and its per-round counters (sums close exactly; six S-A rounds return more outcomes than they generate), 67 kept Sharpes with mean 2.227 and maximum 2.449, and 20 S-A round maxima with mean 2.433 and maximum 4.206.

```python
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Windows (in trading rows) drawn by the expression generator.
WINDOWS: Tuple[int, ...] = (1, 2, 3, 5, 10, 20)
# The three nominal measure terminals (features never see _adj columns).
MEASURES: Tuple[str, ...] = ("close", "volume", "amount")


def random_expression(rng: np.random.Generator, depth: int = 3) -> str:
    """
    Draw one random expression of the public price/volume DSL.

    Purpose:
        Mimic the mined candidates: functional compositions of the
        eight public operators over the nominal terminals.

    Inputs:
        rng (np.random.Generator): The random source.
        depth (int): Maximum remaining tree depth. Default 3.

    Processes:
        1. At depth 0, or with probability 0.3, return a terminal.
        2. Otherwise draw a time-series, binary or negation node and
           recurse into its operands.

    Outputs:
        str: Expression text, e.g.
            ``"div(delta(close, 5), ts_mean(volume, 20))"``.

    Raises:
        ValueError: If ``depth`` is negative.
    """
    # A negative depth is a caller error.
    if depth < 0:
        raise ValueError("depth must be non-negative")
    # Leaves: a nominal terminal.
    if depth == 0 or rng.random() < 0.3:
        return str(rng.choice(MEASURES))
    # Draw the node kind.
    kind: float = rng.random()
    # Time-series node: op(x, n).
    if kind < 0.55:
        # Pick the operator and its window.
        op = str(rng.choice(("ts_mean", "ts_rank", "delta", "lag")))
        # The operand subtree and the window.
        arg, window = random_expression(rng, depth - 1), int(
            rng.choice(WINDOWS)
        )
        # Assemble op(x, n).
        return f"{op}({arg}, {window})"
    # Binary arithmetic node: div(x, y) or sub(x, y).
    if kind < 0.9:
        op = str(rng.choice(("div", "sub")))
        # The two operand subtrees.
        left, right = random_expression(rng, depth - 1), random_expression(
            rng, depth - 1
        )
        # Assemble op(x, y).
        return f"{op}({left}, {right})"
    # Negation node, written with the leading minus the parser desugars.
    return f"-{random_expression(rng, depth - 1)}"


def canonical_id(text: str) -> str:
    """
    Return the canonical-AST identity of an expression (Task 7).

    Purpose:
        Deduplicate on the same key the pipeline freezes on.

    Inputs:
        text (str): Expression text.

    Processes:
        1. Parse with ``parse_expression_task_7``.
        2. Hash the canonical AST with ``expression_id_task_7``.

    Outputs:
        str: 32-character lowercase hexadecimal identity.

    Raises:
        DslParseError: If the text does not parse.
    """
    # Parse, then hash the canonical tree.
    return expression_id_task_7(parse_expression_task_7(text))


def build_cohort_texts(rng: np.random.Generator) -> Dict[str, List[str]]:
    """
    Build the 1,047 distinct expressions that the IS screen will admit.

    Purpose:
        Give the frozen cohort the record types the paper accounts
        for, so every ledger arm is exercised.

    Inputs:
        rng (np.random.Generator): The random source.

    Processes:
        1. 867 ordinary expressions (790 full-coverage, 77 with a long
           lag that leaves them partial coverage).
        2. 98 degenerate expressions on the ``tradable`` flag, whose
           cross-sections never reach three distinct values.
        3. 25 expressions naming an operator outside the public set.
        4. 57 conditionals ``if(delta(tradable, k), 1, x)``: a panel
           condition with a scalar true branch, which the v1 evaluator
           cannot evaluate; 56 have a varying false branch and one is
           constant.

    Outputs:
        Dict[str, List[str]]: Texts keyed by record type.

    Raises:
        RuntimeError: If the generator cannot reach a pinned count.
    """
    # Track the identities already used so all 1,047 are distinct.
    seen: set = set()
    # The texts of each record type.
    out: Dict[str, List[str]] = {
        "ordinary": [],
        "degenerate": [],
        "unsupported": [],
        "conditional": [],
    }

    def add(kind: str, text: str) -> bool:
        """Keep ``text`` under ``kind`` if its identity is new; say if kept."""
        # Compute the canonical identity.
        ident = canonical_id(text)
        # Skip a duplicate identity.
        if ident in seen:
            return False
        # Record the identity and keep the text.
        seen.add(ident)
        out[kind].append(text)
        return True

    # The r1 witness: delta(close, 1) / delta(close, 1) is 1, or NaN (0/0) on a
    # flat session, so on the spell only the five moving stocks are eligible.
    add(
        "ordinary",
        "div(delta(volume, 1), div(delta(close, 1), delta(close, 1)))",
    )
    # 790 ordinary, full-coverage expressions (windows sum well below 243
    # rows).
    while len(out["ordinary"]) < 790:
        add("ordinary", random_expression(rng))
    # 77 ordinary expressions whose 250-300 row lag starts after 2024-01-02.
    for lag in range(250, 327):
        add("ordinary", f"lag({random_expression(rng, 2)}, {lag})")
    # 98 degenerate signals: lag and mean of the tradable flag.
    for k in range(1, 50):
        add("degenerate", f"lag(tradable, {k})")
        add("degenerate", f"ts_mean(tradable, {k})")
    # 25 expressions using operators outside the public registry.
    for k, op in enumerate(("ts_std", "ts_corr", "log", "abs", "ts_max")):
        for w in (5, 10, 20, 40, 60):
            # ts_corr takes two series; the others take one.
            arg = "close, volume" if op == "ts_corr" else str(MEASURES[k % 3])
            add(
                "unsupported",
                (
                    f"{op}({arg}, {w})"
                    if op not in ("log", "abs")
                    else f"{op}(ts_mean({arg}, {w}))"
                ),
            )
    # 56 conditionals with a varying false branch.
    attempts = 0
    while len(out["conditional"]) < 56:
        attempts += 1
        add(
            "conditional",
            f"if(delta(tradable, {1 + attempts % 5}), 1, "
            f"{random_expression(rng, 2)})",
        )
    # One conditional whose false branch is constant.
    add("conditional", "if(delta(tradable, 1), 1, 0)")
    # Check every pinned count before returning.
    for kind, n in (
        ("ordinary", 867),
        ("degenerate", 98),
        ("unsupported", 25),
        ("conditional", 57),
    ):
        if len(out[kind]) != n:
            raise RuntimeError(f"{kind}: {len(out[kind])} != {n}")
    # Return the texts by type.
    return out


def respell(text: str, rng: np.random.Generator) -> str:
    """
    Rewrite an expression with the same canonical identity.

    Purpose:
        Produce the duplicate rows that canonical-AST deduplication
        must collapse (whitespace and operator-name case).

    Inputs:
        text (str): Expression text.
        rng (np.random.Generator): The random source.

    Processes:
        1. Remove spaces or add extra ones.
        2. Upper-case the operator names half of the time.

    Outputs:
        str: A different spelling of the same expression.

    Raises:
        None.
    """
    # Squeeze or widen the spacing.
    out = (
        text.replace(" ", "")
        if rng.random() < 0.5
        else text.replace(",", " ,  ")
    )
    # Upper-case operator names (the parser lower-cases them).
    if rng.random() < 0.5:
        for op in ("ts_mean", "ts_rank", "delta", "lag", "div", "sub", "if"):
            out = out.replace(f"{op}(", f"{op.upper()}(")
    # Return the respelling.
    return out


def build_preserved_input_rows(
    rng: np.random.Generator, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Build ``PRESERVED_INPUT_ROWS``: the 3,114 rows the cohort is frozen from.

    Purpose:
        Reproduce the frozen-cohort source of Section 6: 3,114 rows,
        of which the IS screen plus deduplication keep 1,047.

    Inputs:
        rng (np.random.Generator): The random source.
        config (Dict[str, Any]): Supplies ``P06`` (3,114 rows, the IS
            thresholds) and the ``P27`` column schema.

    Processes:
        1. 1,047 distinct cohort expressions with passing IS metrics.
        2. 520 respelled duplicates of cohort rows, also passing.
        3. 1,547 rows that fail the IS screen (low Sharpe, negative
           fitness, high turnover or a missing metric).
        4. Shuffle, number ``row_id`` 1..3114, parse every row with
           the Task 7 parser for ``expression_id``,
           ``operators_used`` and ``parse_status``; OOS fields are
           null before the freeze.

    Outputs:
        Tuple[pd.DataFrame, Dict[str, List[str]]]: The frame, and the
            cohort texts by record type.

    Raises:
        ValueError: If the row count differs from ``P06``.
    """
    # Pins of the admission rule.
    p06: Dict[str, Any] = config["P06_COHORT_ADMISSION_PARAMETERS"]
    # The 1,047 cohort expressions.
    cohort = build_cohort_texts(rng)
    # Flatten them in type order.
    cohort_texts: List[str] = [t for kind in cohort for t in cohort[kind]]
    # Respell 520 cohort rows as duplicates.
    dup_texts: List[str] = [
        respell(cohort_texts[i], rng)
        for i in rng.choice(len(cohort_texts), 520, replace=False)
    ]
    # Failing rows: fresh expressions whose identities are not in the cohort.
    cohort_ids = {canonical_id(t) for t in cohort_texts}
    fail_texts: List[str] = []
    while len(fail_texts) < int(p06["source_row_count"]) - len(
        cohort_texts
    ) - len(dup_texts):
        text = random_expression(rng, 3)
        if canonical_id(text) not in cohort_ids:
            fail_texts.append(text)

    # IS metrics that pass the stated screen: Sharpe >= 2, Fitness >= 0,
    # Turnover <= 500.
    def passing(n: int) -> Dict[str, np.ndarray]:
        """Draw ``n`` IS metric triples that satisfy the stated screen."""
        # Sharpe in [2, 3.2], fitness in [0, 1.5], turnover in [80, 500].
        return {
            "is_sharpe": rng.uniform(2.0, 3.2, n),
            "is_fitness": rng.uniform(0.0, 1.5, n),
            "is_turnover": rng.uniform(80.0, 500.0, n),
        }

    # IS metrics that fail at least one condition (or are missing).
    def failing(n: int) -> Dict[str, np.ndarray]:
        """Draw ``n`` IS metric triples that each fail one condition."""
        # Start from passing metrics.
        m = passing(n)
        # Pick which condition each row fails: 0 Sharpe, 1 fitness,
        # 2 turnover, 3 a missing Sharpe.
        reason = rng.integers(0, 4, n)
        # Sharpe below 2.
        m["is_sharpe"] = np.where(
            reason == 0, rng.uniform(0.5, 1.99, n), m["is_sharpe"]
        )
        # Negative fitness.
        m["is_fitness"] = np.where(
            reason == 1, rng.uniform(-1.0, -0.01, n), m["is_fitness"]
        )
        # Turnover above 500.
        m["is_turnover"] = np.where(
            reason == 2, rng.uniform(500.01, 900.0, n), m["is_turnover"]
        )
        # A missing Sharpe: NaN fails every comparator of the screen.
        m["is_sharpe"] = np.where(reason == 3, np.nan, m["is_sharpe"])
        # Return the failing metrics.
        return m

    # Assemble the three blocks.
    blocks = []
    for texts, metrics in (
        (cohort_texts, passing(len(cohort_texts))),
        (dup_texts, passing(len(dup_texts))),
        (fail_texts, failing(len(fail_texts))),
    ):
        blocks.append(pd.DataFrame({"expression_text": texts, **metrics}))
    # Shuffle the rows: the preserved order carries no information.
    rows = (
        pd.concat(blocks, ignore_index=True)
        .sample(frac=1.0, random_state=7)
        .reset_index(drop=True)
    )
    # Halt if the pinned row count is not met.
    if len(rows) != int(p06["source_row_count"]):
        raise ValueError(f"{len(rows)} rows, pinned {p06['source_row_count']}")
    # Parse every row once with the Task 7 parser.
    parsed = parse_records_task_7(rows[["expression_text"]])
    # Assemble the P27 columns.
    n = len(rows)
    frame = pd.DataFrame(
        {
            "row_id": np.arange(1, n + 1),
            "expression_id": parsed["expression_id"],
            "expression_text": rows["expression_text"],
            "category_label": rng.choice(
                ["momentum", "reversal", "volume", "liquidity"], n
            ),
            "origin_run": rng.choice(["P", "S-A", "S-B"], n),
            "is_sharpe": rows["is_sharpe"],
            "is_fitness": rows["is_fitness"],
            "is_turnover": rows["is_turnover"],
            "is_ic": rng.normal(0.03, 0.01, n),
            "is_complexity": rng.integers(2, 12, n),
            "is_pool_score": rng.uniform(0, 1, n),
            "is_correlation": rng.uniform(0, 0.7, n),
            "oos_sharpe": np.nan,
            "oos_computed_at": pd.NaT,
            "operators_used": parsed["operators_used"],
            "parse_status": parsed["parse_status"],
        }
    )
    # Cast to the P27 dtypes.
    for col, spec in config["P27_SCHEMA_PRESERVED_INPUT_ROWS"][
        "column_schema"
    ].items():
        frame[col] = frame[col].astype(spec["dtype"])
    # Task 8 deduplicates on an explicit key: the canonical-AST identity.
    frame["expression_id_canonical"] = frame["expression_id"]
    # Return the frame and the cohort texts.
    return frame, cohort


def build_retention_artifacts(
    rng: np.random.Generator, preserved: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build ``NATIVE_OOS_CHECKPOINT`` and ``OUTCOME_FILTERED_CATALOG``.

    Purpose:
        Reproduce the Section 5 retention structure: a 2,016-row
        checkpoint with 368 negatives and a 1,837-row catalog that
        kept only positive OOS outcomes.

    Inputs:
        rng (np.random.Generator): The random source.
        preserved (pd.DataFrame): ``PRESERVED_INPUT_ROWS`` (source of
            identities and texts).
        config (Dict[str, Any]): Supplies the ``P10`` counts.

    Processes:
        1. Split the distinct preserved identities into the
           intersection (1,648, including 1,037 that pass the stated
           IS rule), checkpoint-only (368) and catalog-only (189).
        2. Give the checkpoint-only rows negative OOS Sharpe and all
           others positive: no sign disagreement on the intersection.

    Outputs:
        Tuple[pd.DataFrame, pd.DataFrame]: ``(checkpoint, catalog)``.

    Raises:
        ValueError: If there are too few distinct identities.
    """
    # Retention pins.
    p10: Dict[str, Any] = config["P10_RETENTION_AUDIT_PARAMETERS"]
    # Distinct identities with their first text and IS verdict.
    first = preserved.dropna(subset=["expression_id"]).drop_duplicates(
        "expression_id"
    )
    # The stated IS rule on each identity.
    passes = (
        (first["is_sharpe"] >= 2)
        & (first["is_fitness"] >= 0)
        & (first["is_turnover"] <= 500)
    )
    # Passing and failing pools.
    pass_ids = first[passes.fillna(False).astype(bool)]
    fail_ids = first[~passes.fillna(False).astype(bool)]
    # Sizes of the three partition cells.
    n_int, n_ck, n_cat = (
        p10["intersection_count"],
        p10["checkpoint_only_count"],
        p10["catalog_only_count"],
    )
    n_pass = p10["catalog_rows_passing_stated_is_rule"]
    # Make sure the pools are large enough.
    if (
        len(pass_ids) < n_pass
        or len(fail_ids) < (n_int - n_pass) + n_ck + n_cat
    ):
        raise ValueError(
            "not enough distinct identities for the retention partition"
        )
    # Intersection: 1,037 passing plus 611 failing identities.
    inter = pd.concat(
        [pass_ids.iloc[:n_pass], fail_ids.iloc[: n_int - n_pass]]
    )
    # Checkpoint-only: the next 368 failing identities.
    ck_only = fail_ids.iloc[n_int - n_pass : n_int - n_pass + n_ck]
    # Catalog-only: the next 189 failing identities.
    cat_only = fail_ids.iloc[
        n_int - n_pass + n_ck : n_int - n_pass + n_ck + n_cat
    ]
    # Checkpoint rows: intersection (positive) then checkpoint-only (negative).
    ck = pd.concat([inter, ck_only], ignore_index=True)
    oos = np.concatenate(
        [
            rng.uniform(0.05, 2.5, len(inter)),
            -rng.uniform(0.05, 2.0, len(ck_only)),
        ]
    )
    checkpoint = pd.DataFrame(
        {
            "expression_id": ck["expression_id"],
            "expression_text": ck["expression_text"],
            "checkpoint_rank": np.arange(1, len(ck) + 1),
            "oos_sharpe_native": oos,
            "oos_sign": np.where(oos > 0, "positive", "negative"),
            "is_sharpe_native": ck["is_sharpe"],
            "is_fitness_native": ck["is_fitness"],
            "is_turnover_native": ck["is_turnover"],
            "retention_status": np.where(
                oos > 0, "retained_in_catalog", "dropped_before_write"
            ),
        }
    )
    # Catalog rows: intersection plus catalog-only, all strictly positive OOS.
    cat = pd.concat([inter, cat_only], ignore_index=True)
    catalog = pd.DataFrame(
        {
            "expression_id": cat["expression_id"],
            "expression_text": cat["expression_text"],
            "oos_sharpe_native": np.concatenate(
                [oos[: len(inter)], rng.uniform(0.05, 2.5, len(cat_only))]
            ),
            "written": True,
            "passes_stated_is_rule": np.arange(len(cat)) < n_pass,
            "member_of_checkpoint": np.arange(len(cat)) < len(inter),
        }
    )
    # Cast both frames to their pinned dtypes.
    for frame, key in (
        (checkpoint, "P28_SCHEMA_NATIVE_OOS_CHECKPOINT"),
        (catalog, "P29_SCHEMA_OUTCOME_FILTERED_CATALOG"),
    ):
        for col, spec in config[key]["column_schema"].items():
            frame[col] = frame[col].astype(spec["dtype"])
    # Return both artifacts.
    return checkpoint, catalog


def split_total(
    rng: np.random.Generator, total: int, parts: int = 20
) -> np.ndarray:
    """
    Split an integer total into ``parts`` non-negative integers.

    Purpose:
        Build per-round counters whose sums close exactly to the
        reported totals (P24 sum closures).

    Inputs:
        rng (np.random.Generator): The random source.
        total (int): The reported total.
        parts (int): Number of rounds. Default 20.

    Processes:
        1. Draw a multinomial split with equal round probabilities.

    Outputs:
        np.ndarray: ``parts`` integers summing to ``total``.

    Raises:
        ValueError: If ``total`` is negative.
    """
    # A negative total is impossible for a counter.
    if total < 0:
        raise ValueError("total must be non-negative")
    # Multinomial split: the sum is exact by construction.
    return rng.multinomial(total, np.full(parts, 1.0 / parts))


def split_returned(
    rng: np.random.Generator,
    generated: np.ndarray,
    total: int,
    n_anomalies: int,
) -> np.ndarray:
    """
    Split returned outcomes so ``n_anomalies`` rounds exceed generation.

    Purpose:
        Reproduce the Section 4 observation that six S-A rounds have
        more outcome records than newly generated proposals.

    Inputs:
        rng (np.random.Generator): The random source.
        generated (np.ndarray): Per-round generated proposals.
        total (int): Reported returned-outcome total.
        n_anomalies (int): Rounds whose returns exceed generation.

    Processes:
        1. Start from returns of about 85 percent of generation.
        2. Lift ``n_anomalies`` rounds above their generation.
        3. Move the remaining difference onto the other rounds without
           letting any of them exceed its generation.

    Outputs:
        np.ndarray: Per-round returns summing to ``total``.

    Raises:
        ValueError: If the total cannot be reached under the constraint.
    """
    # Integer generation counts.
    gen = generated.astype(int)
    # Rounds chosen to carry the anomaly.
    anomalous = np.zeros(len(gen), dtype=bool)
    anomalous[rng.choice(len(gen), n_anomalies, replace=False)] = True
    # Baseline: 85 percent of generation, at least one more where anomalous.
    ret = np.where(
        anomalous, gen + rng.integers(1, 6, len(gen)), np.floor(0.85 * gen)
    ).astype(int)
    # Distribute the shortfall (or excess) over the ordinary rounds.
    diff = total - int(ret.sum())
    # Walk the ordinary rounds until the total closes.
    ordinary = np.flatnonzero(~anomalous)
    i = 0
    while diff != 0:
        # Index of the round to adjust.
        j = ordinary[i % len(ordinary)]
        # Add one where room remains below generation; remove one otherwise.
        if diff > 0 and ret[j] < gen[j]:
            ret[j] += 1
            diff -= 1
        elif diff < 0 and ret[j] > 0:
            ret[j] -= 1
            diff += 1
        i += 1
        # Give up if the constraint makes the total unreachable.
        if i > 100000:
            raise ValueError("cannot close the returned-outcome total")
    # Return the per-round returns.
    return ret


def build_run_reports(
    rng: np.random.Generator, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, ...]:
    """
    Build the run reports and the two Sharpe collectors (Section 4).

    Purpose:
        Reproduce Table 1 and the two collector populations: the
        pipeline's kept-candidate vector and the swarm's positive
        per-round maxima.

    Inputs:
        rng (np.random.Generator): The random source.
        config (Dict[str, Any]): Supplies ``P23`` (the totals rows),
            ``P24`` sum closures and ``P09`` mean targets.

    Processes:
        1. ``RUN_REPORT_TOTALS``: the three Table 1 rows.
        2. ``RUN_REPORT_ROUNDS``: 20 rounds per run whose sums close
           to the totals; six S-A rounds return more outcomes than
           they generate.
        3. ``COLLECTOR_A``: 67 kept Sharpes in [2.0, 2.449] with mean
           2.227; ``COLLECTOR_B``: 20 S-A round maxima with mean 2.433
           and maximum 4.206, plus 20 S-B maxima.

    Outputs:
        Tuple[pd.DataFrame, ...]: ``(totals, rounds, collector_a,
            collector_b)``.

    Raises:
        ValueError: If a sum closure fails.
    """
    # Table 1 rows, as pinned in P23.
    totals = pd.DataFrame(
        config["P23_SCHEMA_RUN_REPORT_TOTALS"]["example_rows"]
    )
    # Cast to the P23 dtypes.
    for col, spec in config["P23_SCHEMA_RUN_REPORT_TOTALS"][
        "column_schema"
    ].items():
        totals[col] = totals[col].astype(spec["dtype"])
    # Per-round counters for each run.
    parts = []
    for run_id, keeps, tested, ok, gen, ret in (
        ("P", 67, 289, 234, None, None),
        ("S-A", 20, None, None, 887, 816),
        ("S-B", 11, None, None, 876, 461),
    ):
        rounds = pd.DataFrame(
            {
                "run_id": run_id,
                "round_index": np.arange(1, 21),
                "category_label": rng.choice(
                    ["momentum", "reversal", "volume", "liquidity"], 20
                ),
                "keep_decisions": split_total(rng, keeps),
            }
        )
        # Pipeline counters exist only for P; swarm counters only for S-A/S-B.
        rounds["tested_candidates"] = (
            split_total(rng, tested) if tested else pd.NA
        )
        rounds["successful_outcomes"] = split_total(rng, ok) if ok else pd.NA
        rounds["generated_proposals"] = split_total(rng, gen) if gen else pd.NA
        rounds["returned_outcomes"] = (
            split_returned(
                rng,
                rounds["generated_proposals"].to_numpy(),
                ret,
                6 if run_id == "S-A" else 0,
            )
            if ret
            else pd.NA
        )
        rounds["reviewer_next_step_source"] = (
            rng.choice(["llm", "rule_fallback"], 20, p=[0.8, 0.2])
            if run_id == "P"
            else pd.NA
        )
        rounds["fallback_triggered"] = (
            rounds["reviewer_next_step_source"].eq("rule_fallback")
            if run_id == "P"
            else pd.NA
        )
        parts.append(rounds)
    rounds = pd.concat(parts, ignore_index=True)
    # Cast to the P24 dtypes.
    for col, spec in config["P24_SCHEMA_RUN_REPORT_ROUNDS"][
        "column_schema"
    ].items():
        rounds[col] = rounds[col].astype(spec["dtype"])
    # Check the pinned sum closures.
    for key, pinned in config["P24_SCHEMA_RUN_REPORT_ROUNDS"][
        "sum_closures_required"
    ].items():
        # Key layout: sum_<column>_<run>, the run written P, S_A or S_B.
        match = re.fullmatch(r"sum_(.+)_(P|S_A|S_B)", key)
        # Map the run token back to its run_id.
        col, run = match.group(1), match.group(2).replace("_", "-")
        # Halt if the rounds do not sum to the reported total.
        if int(rounds.loc[rounds["run_id"] == run, col].sum()) != int(pinned):
            raise ValueError(f"sum closure {key} fails")
    # Collector A: 67 kept-candidate Sharpes, max 2.449, mean 2.227.
    a = np.sort(rng.uniform(2.0, 2.449, 67))
    a[-1] = 2.449
    a[:-1] += (2.227 * 67 - a.sum()) / 66
    collector_a = pd.DataFrame(
        {
            "run_id": "P",
            "round_index": rng.integers(1, 21, 67),
            "decision_rank": np.arange(1, 68),
            "expression_id": [f"p{i:04d}" for i in range(67)],
            "sharpe": np.round(a, 3),
        }
    )
    # Collector B: S-A round maxima (mean 2.433, max 4.206) and S-B maxima.
    b = rng.uniform(1.5, 3.0, 20)
    b[0] = 4.206
    b[1:] += (2.433 * 20 - b.sum()) / 19
    collector_b = pd.DataFrame(
        {
            "run_id": ["S-A"] * 20 + ["S-B"] * 20,
            "round_index": np.tile(np.arange(1, 21), 2),
            "max_sharpe_positive": np.round(
                np.concatenate([b, rng.uniform(0.6, 2.8, 20)]), 3
            ),
            "max_is_from_kept": rng.random(40) < 0.4,
        }
    )
    # Cast both collectors to their pinned dtypes.
    for frame, key in (
        (collector_a, "P25_SCHEMA_COLLECTOR_A_PIPELINE"),
        (collector_b, "P26_SCHEMA_COLLECTOR_B_SWARM"),
    ):
        for col, spec in config[key]["column_schema"].items():
            frame[col] = frame[col].astype(spec["dtype"])
    # Return the four Section 4 artifacts.
    return totals, rounds, collector_a, collector_b
```

### **Step 4: Assembling the Ten Inputs**

**Methodology:**
1.  **Generate** the snapshot, then the historical artifacts, each with its own seeded generator.
2.  **Shape** the frames as `run_study` reads them. The bar views keep `stock_code` and `trade_date` both as the compound index (Task 4) and as columns (the calendar built at the validate stage), and `FILES` carries `file_id` as a column; `reset_index` keeps `attrs`.

```python
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Assemble the ten input artifacts.
# ------------------------------------------------------------------
# Trading calendar and universe from the configuration pins.
sessions = build_trading_calendar(config)
stock_codes = build_universe(
    int(config["P11_ACQUISITION_PARAMETERS"]["universe_size"])
)
# Simulate both views; byte_hash is filled once the files are written.
unadj_flat, adj_flat = simulate_bars(stock_codes, sessions)
unadj_flat["byte_hash"] = ""
adj_flat["byte_hash"] = ""
# Cast to the pinned dtypes on the compound index.
UNADJUSTED_BARS = cast_bar_dtypes(unadj_flat, "UNADJUSTED_BARS")
ADJUSTED_BARS = cast_bar_dtypes(adj_flat, "ADJUSTED_BARS")
# Write the 80 files under data/raw/<sha256> and build the manifest.
FILES = write_files_and_manifest(UNADJUSTED_BARS, ADJUSTED_BARS, config)
# run_study reads the keys both as the MultiIndex (Task 4) and as columns
# (the calendar built at the validate stage), so the views carry both.
UNADJUSTED_BARS = UNADJUSTED_BARS.reset_index().set_index(
    ["stock_code", "trade_date"], drop=False
)
ADJUSTED_BARS = ADJUSTED_BARS.reset_index().set_index(
    ["stock_code", "trade_date"], drop=False
)
# The manifest also travels with file_id as a column; attrs survive
# reset_index.
FILES = FILES.reset_index()
# One random source for the historical artifacts.
rng = np.random.default_rng(20261021)
# The 3,114 preserved rows and the cohort's record types.
PRESERVED_INPUT_ROWS, cohort_texts = build_preserved_input_rows(rng, config)
# The Section 5 retention artifacts.
NATIVE_OOS_CHECKPOINT, OUTCOME_FILTERED_CATALOG = build_retention_artifacts(
    rng, PRESERVED_INPUT_ROWS, config
)
# The Section 4 run reports and collectors.
RUN_REPORT_TOTALS, RUN_REPORT_ROUNDS, COLLECTOR_A, COLLECTOR_B = (
    build_run_reports(rng, config)
)
# Print the shape of every artifact.
for _name in (
    "FILES",
    "UNADJUSTED_BARS",
    "ADJUSTED_BARS",
    "PRESERVED_INPUT_ROWS",
    "NATIVE_OOS_CHECKPOINT",
    "OUTCOME_FILTERED_CATALOG",
    "RUN_REPORT_TOTALS",
    "RUN_REPORT_ROUNDS",
    "COLLECTOR_A",
    "COLLECTOR_B",
):
    print(f"{_name:<26}{globals()[_name].shape}")
```

```text
Loaded 50 configuration sections from config.yaml
FILES                     (80, 24)
UNADJUSTED_BARS           (29080, 13)
ADJUSTED_BARS             (29080, 13)
PRESERVED_INPUT_ROWS      (3114, 17)
NATIVE_OOS_CHECKPOINT     (2016, 9)
OUTCOME_FILTERED_CATALOG  (1837, 6)
RUN_REPORT_TOTALS         (3, 13)
RUN_REPORT_ROUNDS         (60, 10)
COLLECTOR_A               (67, 5)
COLLECTOR_B               (40, 4)
```

### **Step 5: Executing the Pipeline (`run_study`)**

With the ten frames and `config` in memory, `run_study` checks that the requested stages are closed under `STAGE_DEPENDENCIES`, then dispatches all thirteen in the pinned order: `validate`, `cleanse`, `custody`, `identity`, `calendar`, `labels`, `dsl`, `freeze`, `score`, `aggregate`, `reference`, `dual_check`, `correction`. With `halt_on_failure=True` it stops at the first halt and stamps the later stages `PIPELINE_HALTED_AT_<stage>`.

**Methodology:**
1.  **Function call:** pass the ten frames, `config`, `list(STAGE_ORDER)` and `halt_on_failure=True`.
2.  **Output handling:** the frozen `StudyResult` carries `stages_halt`, `admissibility_report`, `cohort_hash`, the two ledgers, `correction_partition`, both daily registers, `reference_means`, `dual_check_result`, `block_verdicts` and the claim register.

```python
def summarise_study_result(result: Any) -> None:
    """
    Print the parts of a ``StudyResult`` a reader checks first.

    Purpose:
        Show, in order, whether every stage ran, the run mode and its
        admissibility, the frozen cohort, both ledgers, the v1 to v2
        correction partition, the reference means, the nine-cell
        dual check and any closure that did not hold.

    Inputs:
        result (Any): The frozen ``StudyResult`` returned by
            ``run_study``.

    Processes:
        1. Print one line per stage: ``None`` means it passed.
        2. Print the ``run_mode`` block of the admissibility report.
        3. Print the cohort hash, ledgers and correction partition.
        4. Print the reference means and the dual-check rows.
        5. List the P42/P44 block verdicts that are False.

    Outputs:
        None: Everything is printed.

    Raises:
        AttributeError: If ``result`` is not a ``StudyResult``.
    """
    # Refuse anything that is not a study result.
    if not hasattr(result, "stages_halt"):
        raise AttributeError("expected the StudyResult returned by run_study")
    # 1. Stage verdicts in the pinned order.
    print("Stage verdicts (None = passed):")
    for stage, halt in result.stages_halt.items():
        # One stage per line.
        print(f"  {stage:<11} {halt}")
    # 2. The run mode and whether the run is admissible.
    run_mode = result.admissibility_report["run_mode"]
    print(
        f"\nrun_mode={run_mode['run_mode']}  "
        f"admissible={run_mode['admissible']}  "
        f"n_stages_halted={run_mode['n_stages_halted']}"
    )
    # 3. The frozen cohort and both three-state ledgers.
    print(f"cohort_hash      = {result.cohort_hash}")
    print(
        f"ledger_primary   = {result.ledger_primary}  "
        "(scorable, insufficient, unsupported)"
    )
    print(f"ledger_reverse   = {result.ledger_reverse}")
    print(f"correction       = {dict(result.correction_partition)}")
    # Rows in the two daily registers.
    print(
        f"v1 / v2 register = {len(result.score_register_v1):,} / "
        f"{len(result.score_register_v2):,} rows"
    )
    # 4. The six pre-registered references.
    print("\nReference means:")
    print(result.reference_means.round(4).to_string())
    # The nine-cell dual-implementation check.
    print("\nDual check (primary vs alternative rho):")
    print(result.dual_check_result.to_string(max_colwidth=14))
    # 5. Closures and invariants that did not hold.
    failed = [name for name, ok in result.block_verdicts.items() if not ok]
    print(
        f"\nBlock verdicts: {len(result.block_verdicts) - len(failed)} of "
        f"{len(result.block_verdicts)} hold"
    )
    for name in failed:
        # Each failed closure with its stated reason.
        reason = (
            result.admissibility_report["closures"]
            .get("reasons", {})
            .get(name.split(".", 1)[1], "")
        )
        print(f"  {name}: {reason}")


# Time the end-to-end run.
started = time.time()
# Execute every stage in the pinned order and stop at the first halt.
study_result = run_study(
    files=FILES,
    unadjusted=UNADJUSTED_BARS,
    adjusted=ADJUSTED_BARS,
    preserved_rows=PRESERVED_INPUT_ROWS,
    native_oos_checkpoint=NATIVE_OOS_CHECKPOINT,
    outcome_filtered_catalog=OUTCOME_FILTERED_CATALOG,
    run_report_totals=RUN_REPORT_TOTALS,
    run_report_rounds=RUN_REPORT_ROUNDS,
    collector_a=COLLECTOR_A,
    collector_b=COLLECTOR_B,
    config=config,
    stages=list(STAGE_ORDER),
    halt_on_failure=True,
)
# Report the wall-clock time.
print(f"run_study finished in {(time.time() - started) / 60:.1f} minutes\n")
# Print the summary.
summarise_study_result(study_result)
```

Output of this run:

```text
run_study finished in 92.1 minutes

Stage verdicts (None = passed):
  validate    None
  cleanse     None
  custody     None
  identity    None
  calendar    None
  labels      None
  dsl         None
  freeze      None
  score       None
  aggregate   None
  reference   None
  dual_check  None
  correction  None

run_mode=PRODUCTION  admissible=True  n_stages_halted=0
cohort_hash      = 65f75a9613eb430559eab0cf0d090d667b406176d605e6397115847898912bfc
ledger_primary   = (770, 252, 25)  (scorable, insufficient, unsupported)
ledger_reverse   = (770, 277, 0)
correction       = {'n_total': 1047, 'n_scorable': 923, 'n_insufficient': 99, 'n_unsupported': 25}
v1 / v2 register = 502,560 / 502,560 rows

Reference means:
              mean_rank_ic  n_valid_dates
reference_id                             
r1                  0.0139            480
r2                 -0.0019            480
r3                  0.0106            480
r4                  0.0039            480
r5                  0.0073            480
r6                  0.0254            480

Dual check (primary vs alternative rho):
   expression_id trade_date  primary_rho  alternative_rho         delta  pass_at_tolerance
0  21f74644fe... 2024-09-02     0.111803       0.111803    0.000000e+00           True    
1  0042aded08... 2024-01-24     0.063532       0.063532    0.000000e+00           True    
2  6d2f3461a7... 2024-08-28     1.000000       1.000000    2.220446e-16           True    
3  0342a9640b... 2025-03-17     0.000000       0.000000    0.000000e+00           True    
4  00354f64d1... 2024-01-09    -0.169333      -0.169333    0.000000e+00           True    
5  00354f64d1... 2024-01-15    -0.290633      -0.290633    0.000000e+00           True    
6  00354f64d1... 2024-08-26     0.022202       0.022202    3.469447e-18           True    
7  00354f64d1... 2024-08-27     0.001306       0.001306    2.168404e-19           True    

Block verdicts: 32 of 32 hold
```

Reading the output:

> **Remediation note (R1–R6).** The outputs recorded below come from the notebook *before* the remediation. Since then, `correction_partition` is computed from this run's own v1/v2 registers: the fixture route has been removed, and Step 6's recount is what the correction stage itself reports. The v1 ledger carries an explicit `evaluation_error` outcome. The extended DSL now supports the 25 synthetic "unsupported" expressions. Pass `freeze_attestation` to `run_study`, or the pre-outcome immutability gate is reported as not established and the run is not admissible. See the remediation log in the notebook and the README.

- **Every stage passed** and every P42/P44 closure holds, so the run reports `PRODUCTION` and `admissible=True`.
- **The ledger** under the primary precedence is 770 scorable, 252 insufficient and 25 unsupported, with $770 + 252 + 25 = 1047$: no record is dropped and none is assigned a zero score. The 57 conditionals that v1 cannot evaluate sit in the insufficient arm. Under the reversed precedence the unsupported records move to insufficient.
- **The references** each have 480 valid dates: the label-available sessions of 2024 to 2025, $485 - 5 = 480$.
- **The dual check** recomputed eight cells with the independent tie-ranking and Pearson routine; every difference is at most $2.2 \times 10^{-16}$. Rule r9 (a masked stock with finite raw factor and label values) has no witness here, because a masked cell carries null prices, so the requirement becomes eight of eight.
- **`correction_partition` is not computed from this run's data.** `run_study` re-scores the whole cohort under v2 (the `score_register_v2` register), but the partition it reports comes from Task 13, which runs on its declared, deterministic reconstruction fixture and therefore always returns the paper's 923 / 99 / 25. Step 6 derives the actual v2 ledger.

### **Step 6: The v2 Ledger of This Run**

**Methodology:**
1.  **Recount** admissible dates per expression in each register.
2.  **Classify** each expression: unsupported if it was never evaluated, scorable with at least 20 admissible dates, insufficient otherwise.

```python
def register_ledger(
    register: pd.DataFrame, min_valid_dates: int = 20
) -> Dict[str, int]:
    """
    Derive the three-state ledger directly from a daily-scores register.

    Purpose:
        Recompute the ledger of one evaluator version from its own
        register, so the v2 ledger of *this* run can be read beside
        ``correction_partition``.

    Inputs:
        register (pd.DataFrame): ``score_register_v1`` or
            ``score_register_v2``: one row per (expression, date) with
            ``admissible`` and ``reject_reason``.
        min_valid_dates (int): The scorability threshold
            (``P03.min_valid_dates_for_scorability``). Default 20.

    Processes:
        1. Count admissible dates per expression: |T_e|.
        2. ``unsupported``: the expression was never evaluated
           (``reject_reason == "parse_failed"`` on its rows).
        3. ``scorable`` when |T_e| >= 20, otherwise ``insufficient``.

    Outputs:
        Dict[str, int]: ``n_total``, ``n_scorable``,
            ``n_insufficient``, ``n_unsupported``.

    Raises:
        KeyError: If the register lacks the required columns.
    """
    # Work on flat columns whether or not expression_id is an index level.
    flat: pd.DataFrame = (
        register.reset_index()
        if "expression_id" not in register.columns
        else register
    )
    # Check the required columns.
    for col in ("expression_id", "admissible", "reject_reason"):
        if col not in flat.columns:
            raise KeyError(f"register lacks {col!r}")
    # Strict-True admissibility per row (missing counts as not admissible).
    admissible = (
        flat["admissible"]
        .astype("string")
        .str.lower()
        .eq("true")
        .fillna(False)
    )
    # |T_e|: admissible dates per expression.
    n_valid = admissible.groupby(flat["expression_id"]).sum()
    # Expressions the evaluator could not parse or evaluate.
    unsupported = (
        flat["reject_reason"]
        .eq("parse_failed")
        .fillna(False)
        .groupby(flat["expression_id"])
        .any()
    )
    # Scorable: supported and at least min_valid_dates admissible dates.
    scorable = (~unsupported) & (n_valid >= min_valid_dates)
    # Return the three counts and their total.
    return {
        "n_total": int(len(n_valid)),
        "n_scorable": int(scorable.sum()),
        "n_insufficient": int(((~unsupported) & ~scorable).sum()),
        "n_unsupported": int(unsupported.sum()),
    }


# The ledger of each evaluator version, recomputed from this run's registers.
print("v1 ledger (this run):", register_ledger(study_result.score_register_v1))
print("v2 ledger (this run):", register_ledger(study_result.score_register_v2))
# What the correction stage reports (Task 13's declared reconstruction
# fixture).
print("correction_partition:", dict(study_result.correction_partition))
```

Output (recomputed from the two registers this run wrote):

```text
v1 ledger (this run): {'n_total': 1047, 'n_scorable': 770, 'n_insufficient': 195, 'n_unsupported': 82}
v2 ledger (this run): {'n_total': 1047, 'n_scorable': 822, 'n_insufficient': 200, 'n_unsupported': 25}
correction_partition: {'n_total': 1047, 'n_scorable': 923, 'n_insufficient': 99, 'n_unsupported': 25}
```

The v2 correction recovers 52 of the 57 conditionals to scorable ($770 + 52 = 822$); the other five stay insufficient because their false branch is itself degenerate. In this recount the v1 "unsupported" figure of 82 is the 25 genuinely unsupported expressions plus the 57 that v1 could not evaluate; the pipeline's own v1 ledger reads support from the parse table instead, which is why it reports those 57 as insufficient. Both v2 counts close: $822 + 200 + 25 = 1047$.

### **Summary of the Execution Flow**

1.  **Configuration:** `config.yaml` is read into `config`, which equals the notebook's `STUDY_CONFIG` leaf for leaf once the provenance tags are re-attached.
2.  **Validation and cleansing (Tasks 1 and 2):** the manifest, both bar views and the configuration are checked against their schemas and pins; dtypes are cast without filling, sorting or re-indexing.
3.  **Custody and identity (Tasks 3 and 4):** all 80 files are rehashed from `data/raw`; the two views are aligned and split into nominal features and adjusted labels.
4.  **Calendar, masking and labels (Tasks 5 and 6):** the 727-session calendar is built, the two cells are masked, and Eq. (1) gives $y_{i,t} = P_{i,t+5}^{\mathrm{adj}} / P_{i,t}^{\mathrm{adj}} - 1$ on 480 label-available dates.
5.  **Evaluator and freeze (Tasks 7 and 8):** expressions are parsed with the public DSL; the cohort of 1,047 is screened, deduplicated and frozen with its SHA-256 before any scoring.
6.  **Scoring and aggregation (Tasks 9 and 10):** daily average-tie rank-ICs are computed under the joint date gate, then aggregated into the three-state ledger with every closure checked.
7.  **References, dual check and correction (Tasks 11 to 13):** six pre-registered references are scored with the same instrument; nine (here eight) fixed cells are recomputed independently; the cohort is re-scored under v2.
8.  **Result:** `run_study` freezes everything into one `StudyResult`, whose admissibility report states the run mode, the stage verdicts and the closures. The v2 ledger of the run itself is read from `score_register_v2`.
