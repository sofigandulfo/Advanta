import numpy as np
import pytest

from agroscore.score import ParametrosScore, calcular_score, categoria_riesgo

P = ParametrosScore(k_incertidumbre=1.0, cobertura_score_0=0.8, cobertura_score_1000=1.6,
                    umbral_riesgo_alto=1.0, umbral_riesgo_bajo=1.3)


def test_desglose_del_score():
    # 3500 - 500 = 3000 kg/ha -> 3 t * 300 USD/t = 900 USD/ha -> 900/600 = 1.5
    r = calcular_score(3500, 500, 300, 600, P)
    assert r["rinde_pesimista_kg_ha"] == pytest.approx(3000)
    assert r["ingreso_usd_ha"] == pytest.approx(900)
    assert r["cobertura"] == pytest.approx(1.5)
    assert r["score"] == round(1000 * 0.7 / 0.8)
    assert r["riesgo"] == "bajo"
    assert "900" in r["explicacion"]


def test_score_recortado_a_0_y_1000():
    assert calcular_score(1000, 0, 300, 600, P)["score"] == 0
    assert calcular_score(10000, 0, 300, 600, P)["score"] == 1000


def test_rinde_pesimista_no_negativo():
    r = calcular_score(500, 2000, 300, 600, P)
    assert r["rinde_pesimista_kg_ha"] == 0
    assert r["riesgo"] == "alto"


def test_mas_incertidumbre_baja_el_score():
    assert calcular_score(3500, 800, 300, 600, P)["score"] < calcular_score(3500, 200, 300, 600, P)["score"]


@pytest.mark.parametrize("cobertura,esperado", [(0.99, "alto"), (1.0, "medio"), (1.29, "medio"), (1.3, "bajo")])
def test_limites_de_categorias(cobertura, esperado):
    assert categoria_riesgo(cobertura, P) == esperado


def test_sin_rinde_estimado():
    r = calcular_score(np.nan, 500, 300, 600, P)
    assert r["riesgo"] == "sin dato"
    assert np.isnan(r["score"])


@pytest.mark.parametrize("costo", [0, -10, None])
def test_costo_invalido(costo):
    with pytest.raises(ValueError):
        calcular_score(3000, 500, 300, costo, P)


def test_parametros_inconsistentes():
    with pytest.raises(ValueError):
        ParametrosScore(cobertura_score_0=1.6, cobertura_score_1000=0.8)
