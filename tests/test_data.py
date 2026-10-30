import pytest

from drybean_mlops import config, data


def test_valid_data_passes(df):
    assert len(data.validate(df)) == len(df)


def test_missing_column_fails(df):
    with pytest.raises(ValueError, match="Faltan columnas"):
        data.validate(df.drop(columns=["Area"]))


def test_nulls_fail(df):
    bad = df.copy(); bad.loc[0, "Area"] = None
    with pytest.raises(ValueError, match="nulos"):
        data.validate(bad)


def test_out_of_range_fails(df):
    bad = df.copy(); bad.loc[0, "Extent"] = 5
    with pytest.raises(ValueError, match="fuera del rango"):
        data.validate(bad)


def test_unknown_class_fails(df):
    bad = df.copy(); bad.loc[0, config.TARGET] = "FRIJOL_X"
    with pytest.raises(ValueError, match="Clases desconocidas"):
        data.validate(bad)


def test_normalize_columns():
    import pandas as pd
    out = data.normalize_columns(pd.DataFrame(columns=["aspectration", "Roundness", "class"]))
    assert list(out.columns) == ["AspectRation", "roundness", "Class"]
