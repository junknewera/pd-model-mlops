import logging
import sys

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

TARGET_RAW = "default payment next month"
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_COLUMNS = (
    ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_COLS
    + BILL_COLS
    + PAY_AMT_COLS
    + [TARGET_RAW]
)


def build_suite() -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="credit_raw")
    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchSet(column_set=RAW_COLUMNS, exact_match=True)
    )
    suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=1))
    for col in RAW_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInTypeList(column=col, type_list=["int64"])
        )
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="ID"))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="LIMIT_BAL", min_value=1, max_value=10_000_000
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="AGE", min_value=18, max_value=100)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="SEX", value_set=[1, 2])
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="EDUCATION", value_set=list(range(7)))
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="MARRIAGE", value_set=[0, 1, 2, 3])
    )
    for col in PAY_COLS:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(column=col, min_value=-2, max_value=9)
        )
    for col in PAY_AMT_COLS:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(column=col, min_value=0)
        )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column=TARGET_RAW, value_set=[0, 1])
    )
    # в исходных данных дефолтов 22%, сильно другая доля - повод разобраться
    suite.add_expectation(
        gxe.ExpectColumnMeanToBeBetween(column=TARGET_RAW, min_value=0.1, max_value=0.4)
    )
    return suite


def validate(df: pd.DataFrame):
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    batch_def = (
        context.data_sources.add_pandas("pandas")
        .add_dataframe_asset("credit")
        .add_batch_definition_whole_dataframe("batch")
    )
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    return batch.validate(build_suite())


def failed_expectations(result) -> list[str]:
    return [
        f"{r.expectation_config.type} {r.expectation_config.kwargs.get('column', '')}"
        for r in result.results
        if not r.success
    ]


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/credit.csv"
    result = validate(pd.read_csv(path))
    stats = result.statistics
    print(
        f"{path}: {stats['successful_expectations']}/"
        f"{stats['evaluated_expectations']} expectations passed"
    )
    if not result.success:
        for name in failed_expectations(result):
            print("FAILED:", name)
        sys.exit(1)
