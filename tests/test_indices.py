import numpy as np
import pytest

from agroscore.indices import evi, ndvi, ndwi, reflectancia


def test_ndvi_valores_conocidos():
    assert ndvi(0.5, 0.1) == pytest.approx(0.4 / 0.6)
    assert ndvi(0.3, 0.3) == pytest.approx(0.0)
    np.testing.assert_allclose(ndvi([0.4, 0.1], [0.1, 0.4]), [0.6, -0.6])


def test_ndvi_division_por_cero_es_nan():
    assert np.isnan(ndvi(0.0, 0.0))


def test_ndvi_rango():
    rng = np.random.default_rng(0)
    nir, rojo = rng.uniform(0, 1, 1000), rng.uniform(0, 1, 1000)
    v = ndvi(nir, rojo)
    assert np.all((v >= -1) & (v <= 1))


def test_evi_valor_conocido():
    # G*(NIR-R)/(NIR + 6R - 7.5B + 1) = 2.5*0.4/(0.5+0.6-0.375+1)
    assert evi(0.5, 0.1, 0.05) == pytest.approx(2.5 * 0.4 / 1.725)


def test_ndwi_gao():
    assert ndwi(0.4, 0.2) == pytest.approx(0.2 / 0.6)


def test_reflectancia_aplica_offset_desde_baseline_04():
    assert reflectancia(2000, "03.01") == pytest.approx(0.2)
    assert reflectancia(2000, "04.00") == pytest.approx(0.1)
    assert reflectancia(2000, None) == pytest.approx(0.2)


def test_reflectancia_cero_es_sin_dato():
    assert np.isnan(reflectancia(0, "05.09"))
