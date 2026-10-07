import pandas as pd
import pytest

from src.data.download import RAW_PATH
from src.data.validation import TARGET_RAW, failed_expectations, validate


def test_valid_data_passes(raw_clients):
    assert validate(raw_clients).success


@pytest.mark.parametrize(
    "column, value",
    [
        ("AGE", 150),
        ("SEX", 3),
        ("EDUCATION", 9),
        ("LIMIT_BAL", -1),
        ("PAY_0", 12),
        ("PAY_AMT1", -5),
        (TARGET_RAW, 2),
    ],
)
def test_out_of_range_fails(raw_clients, column, value):
    raw_clients.loc[0, column] = value
    assert not validate(raw_clients).success


def test_null_fails(raw_clients):
    raw_clients["AGE"] = raw_clients["AGE"].astype(float)
    raw_clients.loc[0, "AGE"] = None
    result = validate(raw_clients)
    assert "expect_column_values_to_not_be_null AGE" in failed_expectations(result)


def test_wrong_type_fails(raw_clients):
    raw_clients["LIMIT_BAL"] = raw_clients["LIMIT_BAL"].astype(str)
    assert not validate(raw_clients).success


def test_missing_column_fails(raw_clients):
    assert not validate(raw_clients.drop(columns="BILL_AMT3")).success


def test_duplicate_id_fails(raw_clients):
    raw_clients.loc[1, "ID"] = raw_clients.loc[0, "ID"]
    assert not validate(raw_clients).success


@pytest.mark.skipif(not RAW_PATH.exists(), reason="нет данных, нужен dvc repro")
def test_raw_data_is_valid():
    result = validate(pd.read_csv(RAW_PATH))
    assert result.success, failed_expectations(result)
