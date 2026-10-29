import numpy as np
import pandas as pd
import pytest

from agroscore import model


def _datos(campanias=("2019/20", "2020/21", "2021/22"), n=30, semilla=0):
    rng = np.random.default_rng(semilla)
    filas = []
    for c in campanias:
        for i in range(n):
            fila = {v: rng.normal() for v in model.VARIABLES_NUMERICAS}
            fila |= {"lote_id": f"L{i}", "campania": c, "cultivo": ["soja", "maiz"][i % 2]}
            fila["rinde_kg_ha"] = 3000 + 800 * fila["ndvi_pico"] + rng.normal(0, 100)
            filas.append(fila)
    return pd.DataFrame(filas)


def test_pliegues_nunca_mezclan_campanias():
    df = _datos()
    vistas = []
    for campania, train, test in model.pliegues_por_campania(df):
        assert set(df.iloc[test]["campania"]) == {campania}
        assert campania not in set(df.iloc[train]["campania"])
        vistas.append(campania)
    assert sorted(vistas) == sorted(df["campania"].unique())


def test_una_sola_campania_no_se_puede_validar():
    with pytest.raises(ValueError):
        list(model.pliegues_por_campania(_datos(campanias=("2022/23",))))


def test_entrenar_supera_al_baseline_cuando_hay_senal():
    paquete = model.entrenar(_datos())
    g = paquete["metricas_globales"]
    assert g["ridge"]["mae_kg_ha"] < g["baseline"]["mae_kg_ha"]
    assert paquete["predicciones_validacion"]["rinde_validacion_kg_ha"].notna().all()
    assert set(paquete["rmse_por_cultivo"]) == {"soja", "maiz"}


def test_metricas_basicas():
    m = model.metricas([100, 200], [110, 190])
    assert m["mae_kg_ha"] == pytest.approx(10)
    assert m["sesgo_kg_ha"] == pytest.approx(0)
