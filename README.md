# agroscore · MVP de scoring de riesgo agropecuario con imágenes satelitales

Prototipo para **validar si un modelo basado en Sentinel-2 y clima predice bien los rindes de
campañas pasadas** y cómo se traduciría ese rinde en un score de riesgo crediticio.

> ⚠️ **El score es una herramienta de apoyo al análisis, no una decisión de crédito.**
> Leé [LIMITACIONES.md](LIMITACIONES.md) antes de usar cualquier resultado.

```
CSV de lotes ─► Sentinel-2 L2A (máscara de nubes SCL) ─► serie NDVI/EVI/NDWI ─► interpolación + suavizado
                NASA POWER (lluvia, temperatura, radiación) ───────────────────┐        │
                                                                               ▼        ▼
                                                          variables por lote (pico, AUC, pendientes, clima)
                                                                               │
                                     modelo de rinde (validación por campaña) ◄┘
                                                                               │
                         rinde estimado ─► ingreso ─► cobertura vs. costo/deuda ─► score 0-1000 + riesgo
```

## Inicio rápido (sin conexión, con el dataset sintético)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard,dev]"

agroscore demo                    # procesa, entrena, valida y calcula scores del ejemplo
streamlit run app/dashboard.py    # abre el dashboard en http://localhost:8501
pytest                            # tests
```

`agroscore demo` usa los archivos de `data/ejemplo/` y escribe sus resultados en `outputs/` y `models/`.
La corrida tarda unos segundos y muestra algo así:

```
Validación por campaña (leave-one-campaign-out), 226 lotes:
  baseline MAE   1473 kg/ha | RMSE   1851 | R²  0.55 | MAPE 48.4 %
  ridge    MAE    806 kg/ha | RMSE   1006 | R²  0.87 | MAPE 28.1 %
  gboost   MAE    598 kg/ha | RMSE    872 | R²  0.90 | MAPE 17.7 %
```

**Estos números no dicen nada sobre la calidad real del modelo**: el dataset sintético se genera
con las mismas variables que el modelo observa. Sirve para probar que el circuito funciona.
Ver [METRICAS.md](METRICAS.md).

Para regenerar el ejemplo: `agroscore generar-ejemplo --n-lotes 60 --semilla 42`.

## Uso con datos reales

### 1. Preparar el CSV de lotes

Guardalo en `data/private/` (**ignorada por git**: los datos de productores nunca se suben al repo).
Hay una plantilla en [`data/ejemplo/plantilla_lotes.csv`](data/ejemplo/plantilla_lotes.csv).

| Columna | Obligatoria | Descripción |
|---|---|---|
| `lote_id` | sí | Identificador del lote (el mismo en todas las campañas) |
| `cultivo` | sí | `soja` o `maiz` (para otros, agregar parámetros en `config/config.yaml`) |
| `campania` | sí | Formato `AAAA/AA`, ej. `2022/23` |
| `geometria` | una de las dos | Polígono GeoJSON en EPSG:4326 (Polygon, MultiPolygon o Feature) |
| `lat`, `lon`, `buffer_m` | una de las dos | Punto; se arma un cuadrado de lado 2×`buffer_m` (100 m por defecto) |
| `fecha_siembra` | no | `AAAA-MM-DD`. Si falta se usa una ventana fija por cultivo |
| `rinde_kg_ha` | no | Rinde real. Necesario para entrenar y validar |
| `costo_usd_ha` | no | Costo o deuda del lote. Si falta, se usa el de referencia del cultivo |

### 2. Descargar y procesar

Necesita los extras satelitales y acceso a internet:

```bash
pip install -e ".[satelital]"
agroscore procesar --lotes data/private/lotes_aliado.csv \
    --fuente-img planetary --fuente-clima nasa_power --salida outputs/aliado
```

Las descargas quedan en caché en `data/cache/` (ignorada por git). La clave de caché incluye geometría,
fechas y umbrales: si se repite la corrida con los mismos parámetros, no se consulta nada de nuevo.
Si un lote falla (por ejemplo, sin imágenes), queda registrado en la columna `error` y el resto sigue.

> **Código de descarga no probado contra los servicios reales.** El MVP se desarrolló en un entorno
> sin acceso a Planetary Computer ni a NASA POWER. La lógica de nubes, índices y series sí está
> testeada. Antes de confiar en la descarga, probala con 2 o 3 lotes conocidos y revisá las curvas en
> el dashboard.

### 3. Entrenar, validar y scorear

```bash
agroscore entrenar --features outputs/aliado/features.csv --modelo models/aliado.joblib --salida outputs/aliado
agroscore scorear  --features outputs/aliado/features.csv --modelo models/aliado.joblib --salida outputs/aliado
```

Para reentrenar con datos nuevos alcanza con volver a correr `procesar` y `entrenar`: el modelo se
guarda junto con sus métricas, las campañas usadas y la fecha de entrenamiento.

El dashboard lee `outputs/`. Para ver otra carpeta:
`AGROSCORE_SALIDA=outputs/aliado streamlit run app/dashboard.py`.

## Metodología

### Nubes (Sentinel-2 L2A)
1. **Por escena:** se descartan las escenas con `eo:cloud_cover` mayor a `max_nubosidad_escena` (60 %).
2. **Por píxel:** se usa la banda SCL (Scene Classification Layer). Son válidas las clases 4
   (vegetación), 5 (suelo desnudo) y 6 (agua). Se descartan sombras de nube (3), nubes de probabilidad
   media y alta (8, 9), cirros (10), nieve (11), saturados (1), áreas oscuras (2) y sin clasificar (7).
3. **Por fecha:** se usa solo si al menos el 60 % de los píxeles del lote son válidos
   (`min_fraccion_valida`). El valor del lote es la **mediana** de los píxeles válidos.
4. **Radiometría:** desde el processing baseline 04.00 (enero 2022) se aplica el offset de −1000 DN.

### Índices
- **NDVI** = (B08 − B04) / (B08 + B04)
- **EVI** = 2,5 × (B08 − B04) / (B08 + 6·B04 − 7,5·B02 + 1). Satura menos que el NDVI en canopeos densos, como el maíz.
- **NDWI (Gao)** = (B08 − B11) / (B08 + B11). Mide el agua del canopeo, no el agua libre.

El modelo usa hoy solo variables de NDVI. EVI y NDWI se calculan y guardan para usarlos más adelante.

### Interpolación de huecos (documentada en `src/agroscore/timeseries.py`)
1. Serie diaria con **interpolación lineal** entre observaciones válidas. No se extrapola fuera de la
   primera y la última observación.
2. Los huecos de más de `max_hueco_dias` (30) se interpolan igual pero se **marcan**. Cada lote informa
   `n_obs_validas`, `max_hueco_dias` y `frac_dias_hueco_largo`, que el modelo usa y el dashboard muestra.
3. **Suavizado Savitzky-Golay** (ventana de 21 días, orden 2), que conserva la altura del pico mejor que una media móvil.

### Variables por lote
| Variable | Definición |
|---|---|
| `ndvi_pico` | Máximo de la curva suavizada |
| `dias_al_pico` | Días desde el inicio de la campaña (siembra − 15) hasta el pico |
| `auc_ndvi` | Área bajo la curva por encima de NDVI 0,2 (NDVI·día). Aproxima la biomasa acumulada |
| `pendiente_crecimiento` | Máxima pendiente en 10 días antes del pico (NDVI/día) |
| `caida_pre_pico` | Mayor caída respecto del máximo previo antes del pico (estrés durante el crecimiento) |
| `caida_post_pico_30d` | NDVI del pico − NDVI 30 días después (senescencia anticipada) |
| `lluvia_total_mm`, `lluvia_critico_mm` | Lluvia de la campaña y del período crítico (pico −20 / +30 días) |
| `tmax_media`, `dias_calor_critico` | Temperatura máxima media y días con más de 35 °C en el período crítico |
| `radiacion_total_mj`, `gdd` | Radiación acumulada y grados-día (base 10 °C) |

### Modelo y validación
- Se comparan tres modelos: un **baseline** (rinde medio del cultivo), **Ridge** (regresión regularizada)
  y **HistGradientBoosting**. Se elige el de menor MAE fuera de muestra.
- **Validación leave-one-campaign-out:** se entrena con todas las campañas menos una y se evalúa en la
  excluida. Nunca hay división aleatoria, porque los lotes de una misma campaña comparten clima.
- Se reportan MAE, RMSE, R², MAPE y sesgo, en total, por cultivo y por campaña.

### Score
```
rinde_pesimista = rinde_estimado − k × RMSE_validación_del_cultivo   (k = 1)
ingreso [USD/ha] = rinde_pesimista [t/ha] × precio de referencia [USD/t]
cobertura        = ingreso / costo_o_deuda [USD/ha]
score            = 0 si cobertura ≤ 0,8 · 1000 si cobertura ≥ 1,6 · lineal en el medio
riesgo           = alto (< 1,0) · medio (1,0 a 1,3) · bajo (≥ 1,3)
```
Cada lote trae una explicación paso a paso en `scores.csv` y en el dashboard. Todos los parámetros
están en `config/config.yaml`. Precio, costo y k también se pueden ajustar en vivo en el dashboard.

Para los lotes usados en el entrenamiento, `scorear` usa la **predicción fuera de muestra** (la de
cuando su campaña quedó excluida), así el backtest no es optimista. `scores.csv` incluye además
`riesgo_real`, el riesgo calculado con el rinde verdadero, para medir si el score ordena bien.

## Servicios de datos: registro, límites y términos

No se usa Google Earth Engine. Lo que sigue se relevó en octubre de 2026 a partir de documentación
pública y búsquedas. **Hay que verificarlo en la fuente oficial antes de cualquier uso comercial.**

| Servicio | Registro | Límites | Uso comercial / restricciones |
|---|---|---|---|
| **Microsoft Planetary Computer** (STAC, Sentinel-2 L2A) | No hace falta para leer. Los tokens SAS se piden en forma anónima; con una cuenta (`PC_SDK_SUBSCRIPTION_KEY`) los límites son más amplios | Límites de descarga sin números publicados; más holgados desde Azure West Europe | Términos en estado *"Preview"*: **sin garantía de disponibilidad ni soporte**. Los tokens son personales y no se pueden compartir con terceros. **No se encontró una cláusula explícita sobre uso comercial**: revisar [planetarycomputer.microsoft.com/terms](https://planetarycomputer.microsoft.com/terms) |
| **Datos Copernicus Sentinel-2** | — | — | Libres y abiertos, uso comercial incluido. Exigen atribución: *"Contains modified Copernicus Sentinel data [año]"* |
| **NASA POWER** | No hace falta clave | Sin límite fijo publicado; el equipo monitorea el uso y puede restringirlo. El cliente espera 1 s entre consultas y reintenta | Datos gratuitos. No se encontró una página de términos ni una forma de cita: verificar en [power.larc.nasa.gov](https://power.larc.nasa.gov) y citar la fuente en informes |
| **Copernicus Data Space** (alternativa, no implementada) | Obligatorio (cuenta gratuita y claves S3) | Cuotas según el tipo de cuenta | Según su documentación, la cuota gratuita admite uso comercial individual; los volúmenes grandes tienen condiciones comerciales. Ver el capítulo 3 de sus T&C |

El mapa del dashboard usa teselas de OpenStreetMap (atribución obligatoria; su política de uso no
permite tráfico intensivo).

## Estructura

```
config/config.yaml        umbrales de nubes, ventanas por cultivo, precios, costos y parámetros del score
data/ejemplo/             dataset sintético y plantilla de CSV (versionados)
data/private/             datos reales (ignorados por git)
data/cache/               descargas cacheadas (ignoradas)
src/agroscore/
  lotes.py                lectura y validación del CSV, geometrías, ventana de campaña
  sentinel.py             proveedores de imágenes: archivo, Planetary Computer, Copernicus (stub)
  indices.py              NDVI, EVI, NDWI, reflectancia con offset por baseline
  timeseries.py           filtro de nubes, interpolación, suavizado
  clima.py                NASA POWER y clima desde archivo
  features.py             variables de la curva y del clima
  pipeline.py             orquestación por lote
  model.py                baseline, Ridge, gradient boosting y validación por campaña
  score.py                score explicable
  synthetic.py            generador del dataset sintético
  cache.py                caché en disco
  cli.py                  comandos agroscore
app/dashboard.py          dashboard de Streamlit
tests/                    índices, series, score, validación y pipeline de punta a punta
```

## Privacidad
- `data/private/`, `data/cache/`, `outputs/` y `models/` están en `.gitignore`.
- Antes de cada commit, revisá `git status` y confirmá que no aparezca ningún archivo con datos de productores.
- Las claves (Planetary Computer, Copernicus) van en variables de entorno, nunca en el repo.
