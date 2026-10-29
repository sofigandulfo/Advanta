import numpy as np
import pandas as pd
import pytest

from agroscore.features import features_curva
from agroscore.timeseries import filtrar_validas, interpolar_diaria, max_hueco


def test_filtrar_validas_descarta_nubes_y_promedia_mismo_dia():
    obs = pd.DataFrame({
        "fecha": ["2023-01-01", "2023-01-01", "2023-01-06", "2023-01-11"],
        "ndvi": [0.5, 0.7, 0.1, np.nan],
        "frac_valida": [0.9, 0.8, 0.3, 0.95],
    })
    v = filtrar_validas(obs, 0.6)
    assert len(v) == 1
    assert v["ndvi"].iloc[0] == pytest.approx(0.6)


def test_interpolacion_lineal_entre_observaciones():
    s = interpolar_diaria(["2023-01-01", "2023-01-11"], [0.2, 0.7], "2023-01-01", "2023-01-11",
                          max_hueco_dias=30, savgol_ventana=3, savgol_orden=1)
    assert s.loc["2023-01-06", "interp"] == pytest.approx(0.45)
    assert not s["hueco_largo"].any()


def test_no_extrapola_fuera_de_las_observaciones():
    s = interpolar_diaria(["2023-01-05", "2023-01-10"], [0.3, 0.5], "2023-01-01", "2023-01-15")
    assert s.loc[:"2023-01-04", "interp"].isna().all()
    assert s.loc["2023-01-11":, "interp"].isna().all()


def test_marca_huecos_largos():
    s = interpolar_diaria(["2023-01-01", "2023-02-15", "2023-02-20"], [0.3, 0.6, 0.6],
                          "2023-01-01", "2023-02-20", max_hueco_dias=30)
    assert s.loc["2023-01-20", "hueco_largo"]
    assert not s.loc["2023-02-17", "hueco_largo"]
    assert s.loc["2023-01-20", "interp"] == pytest.approx(0.3 + 0.3 * 19 / 45)


def test_max_hueco_incluye_bordes():
    assert max_hueco(["2023-01-10", "2023-01-15"], "2023-01-01", "2023-01-31") == 16


def test_features_curva_detecta_pico():
    dias = pd.date_range("2023-01-01", periods=121, freq="D")
    t = np.arange(121)
    curva = 0.2 + 0.6 * np.exp(-((t - 60) / 20) ** 2)
    f = features_curva(pd.DataFrame({"suave": curva}, index=dias), "2023-01-01", ndvi_base=0.2)
    assert f["dias_al_pico"] == 60
    assert f["ndvi_pico"] == pytest.approx(0.8)
    assert f["auc_ndvi"] == pytest.approx(0.6 * 20 * np.sqrt(np.pi), rel=0.01)
    assert f["caida_pre_pico"] == pytest.approx(0.0)
    assert f["caida_post_pico_30d"] == pytest.approx(0.6 * (1 - np.exp(-(30 / 20) ** 2)))
