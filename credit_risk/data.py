"""Load the Home Credit files and build one row per applicant."""
import re

import pandas as pd

from . import config

BUREAU_CATEGORICAL = ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"]
COUNT_PREFIXES = tuple(f"BUREAU_{c}_" for c in BUREAU_CATEGORICAL)


def load_csv(path):
    print(f"  reading {path.name}")
    return pd.read_csv(path)


def aggregate_bureau(bureau):
    """Summarise each applicant's bureau records (0 to 116 of them) into a single row.

    Numeric columns get their mean, max and min; money columns (AMT_*) also get a total.
    Text columns become counts, e.g. BUREAU_CREDIT_ACTIVE_Active = number of active loans.
    """
    grouped = bureau.groupby(config.ID_COL)

    numeric_cols = [c for c in bureau.select_dtypes("number").columns if c not in (config.ID_COL, "SK_ID_BUREAU")]
    aggregations = {c: ["mean", "max", "min"] + (["sum"] if c.startswith("AMT_") else []) for c in numeric_cols}
    numeric = grouped.agg(aggregations)
    numeric.columns = [f"BUREAU_{col}_{stat.upper()}" for col, stat in numeric.columns]

    dummies = pd.get_dummies(bureau[BUREAU_CATEGORICAL], prefix=[f"BUREAU_{c}" for c in BUREAU_CATEGORICAL])
    counts = dummies.groupby(bureau[config.ID_COL]).sum()
    counts.columns = [re.sub(r"\W+", "_", c) for c in counts.columns]  # "Consumer credit" -> "Consumer_credit"

    summary = numeric.join(counts)
    summary.insert(0, "BUREAU_LOAN_COUNT", grouped.size())
    return summary, list(counts.columns)


def build_applicant_table(application, bureau):
    """Left-join the bureau summary onto the applications, keeping exactly one row per applicant."""
    summary, count_cols = aggregate_bureau(bureau)
    table = application.join(summary, on=config.ID_COL, how="left")

    # An applicant with no bureau records has zero bureau loans, not an unknown number of them.
    count_cols = ["BUREAU_LOAN_COUNT", *count_cols]
    table[count_cols] = table[count_cols].fillna(0).astype(int)

    assert len(table) == len(application), "the join must not duplicate applicants"
    assert table[config.ID_COL].is_unique
    return table


def to_model_input(table, input_columns):
    """Put the columns in the order a saved model expects.

    Loan-count columns only exist for loan types present in the data, so a small batch of new applicants
    may lack some; a missing count means zero such loans. Any other missing column is an error.
    """
    missing = [c for c in input_columns if c not in table.columns and not c.startswith(COUNT_PREFIXES)]
    if missing:
        raise KeyError(f"input is missing columns the model needs: {missing}")
    return table.reindex(columns=input_columns, fill_value=0)
