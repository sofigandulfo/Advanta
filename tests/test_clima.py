import numpy as np
import pandas as pd

from agroscore import clima


def test_respuesta_nasa_power_a_dataframe():
    # Estructura de properties.parameter de la API diaria por punto de NASA POWER.
    parametros = {
        "PRECTOTCORR": {"20230101": 12.5, "20230102": 0.0},
        "T2M_MAX": {"20230101": 31.2, "20230102": -999},
        "T2M_MIN": {"20230101": 18.0, "20230102": 17.1},
        "ALLSKY_SFC_SW_DWN": {"20230101": 25.3, "20230102": 27.0},
    }
    df = clima._a_dataframe(parametros)
    assert list(df.columns) == clima.COLUMNAS
    assert df["fecha"].iloc[0] == pd.Timestamp("2023-01-01")
    assert df["lluvia_mm"].iloc[0] == 12.5
    assert np.isnan(df["tmax"].iloc[1])  # -999 = faltante


def test_nasa_power_usa_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(clima.cache, "DIR_CACHE", tmp_path)
    llamadas = []

    def consulta_falsa(self, lat, lon, inicio, fin):
        llamadas.append((lat, lon))
        return {"PRECTOTCORR": {"20230101": 1.0}, "T2M_MAX": {"20230101": 30.0},
                "T2M_MIN": {"20230101": 15.0}, "ALLSKY_SFC_SW_DWN": {"20230101": 20.0}}

    monkeypatch.setattr(clima.ProveedorNasaPower, "_consultar", consulta_falsa)
    prov = clima.ProveedorNasaPower(pausa_s=0)
    fila = {"lat": -33.891, "lon": -60.572}
    ini, fin = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")
    prov.clima(fila, ini, fin)
    prov.clima(fila, ini, fin)
    assert len(llamadas) == 1
