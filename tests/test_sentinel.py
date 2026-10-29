"""Resumen por lote de Sentinel-2 con imágenes falsas en memoria (sin red)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

odc_stac = pytest.importorskip("odc.stac")
from odc.geo.geobox import GeoBox  # noqa: E402
from odc.geo.geom import Geometry  # noqa: E402
from odc.geo.xr import xr_zeros  # noqa: E402
from shapely.geometry import mapping  # noqa: E402

from agroscore.config import cargar_config  # noqa: E402
from agroscore.lotes import rectangulo_metros  # noqa: E402
from agroscore.sentinel import ProveedorPlanetary  # noqa: E402


def test_resumir_aplica_mascara_scl_y_offset(monkeypatch):
    geom = rectangulo_metros(-33.9, -60.6, 200, 200)
    geobox = GeoBox.from_geopolygon(Geometry(mapping(geom), crs="EPSG:4326").to_crs("EPSG:32720"), resolution=10)
    fechas = pd.Index(pd.to_datetime(["2023-01-05", "2023-01-10"]), name="time")
    base = xr_zeros(geobox, dtype="uint16")

    def banda(valor):
        return xr.concat([base + np.uint16(valor)] * 2, dim=fechas)

    scl = banda(4)
    scl[1] = np.uint16(9)  # segunda fecha: toda nube
    ds = xr.Dataset({"B02": banda(1500), "B03": banda(1800), "B04": banda(1500), "B08": banda(5000),
                     "B11": banda(3000), "SCL": scl})
    monkeypatch.setattr(odc_stac, "load", lambda *a, **k: ds)

    items = [SimpleNamespace(datetime=datetime(2023, 1, d, 14, tzinfo=timezone.utc),
                             properties={"s2:processing_baseline": "05.09"}) for d in (5, 10)]
    obs = ProveedorPlanetary(cargar_config())._resumir(items, geom)

    assert obs.loc[0, "frac_valida"] == pytest.approx(1.0)
    # Con offset -1000: NIR 0.4, rojo 0.05 -> NDVI 0.35/0.45
    assert obs.loc[0, "ndvi"] == pytest.approx(0.35 / 0.45)
    assert obs.loc[1, "frac_valida"] == 0
    assert np.isnan(obs.loc[1, "ndvi"])
