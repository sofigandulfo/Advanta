# Qué métricas esperar y cómo validarlas con datos reales

## Métricas que reporta el sistema

| Métrica | Qué mide | Dónde |
|---|---|---|
| **MAE** (kg/ha) | Error medio absoluto. La más fácil de comunicar: "en promedio le erramos por X kg/ha" | consola, `metricas.json`, dashboard |
| **RMSE** (kg/ha) | Igual que el MAE pero castiga más los errores grandes. **Lo usa el score** como margen de incertidumbre | ídem |
| **R²** | Qué proporción de la variabilidad del rinde explica el modelo (1 = perfecto, 0 = igual que el promedio) | ídem |
| **MAPE** (%) | Error porcentual medio. Útil para comparar soja y maíz | ídem |
| **Sesgo** (kg/ha) | Promedio de (estimado − real). Positivo significa que el modelo sobreestima | `metricas_por_campania.csv` |
| **Riesgo estimado vs. real** | Tabla cruzada entre el riesgo del score y el que daría el rinde verdadero | consola de `scorear`, `scores.csv` |

Todas las métricas son **fuera de muestra con validación por campaña**: cada lote se evalúa con un
modelo que no vio su campaña.

## Qué esperar

### Con el dataset sintético
MAE de unos 600 kg/ha y R² cercano a 0,9 con gradient boosting. Estos valores **no son una
expectativa realista**: el rinde sintético se generó con las mismas variables que el modelo observa.

### Con datos reales (orientativo)
A nivel de lote, con NDVI de Sentinel-2 y clima de baja resolución, es razonable esperar en un
primer modelo:

| Cultivo | MAE razonable | R² por cultivo | Señal de alarma |
|---|---|---|---|
| Soja | 350 a 600 kg/ha (12 a 20 %) | 0,4 a 0,7 | MAE > 700 kg/ha o R² < 0,3 |
| Maíz | 900 a 1.500 kg/ha (12 a 20 %) | 0,4 a 0,7 | MAE > 1.800 kg/ha o R² < 0,3 |

Son rangos aproximados para fijar expectativas, no compromisos. Dependen mucho de la calidad de los
rindes informados, de la cantidad de campañas y de la diversidad de zonas.

**No mires el R² global que mezcla soja y maíz**: sale alto solo porque el maíz rinde más que la
soja. Las métricas que importan son las de cada cultivo.

## Cómo validar con datos reales del aliado

1. **Reunir al menos 3 campañas** (idealmente 4 o 5, con al menos un año seco), con rinde real
   por lote y polígono. Si puede ser, que tengan fecha de siembra.
2. **Reservar la última campaña completa** como prueba final: entrenar con las anteriores
   (`agroscore entrenar` con un CSV filtrado) y evaluar en esa campaña una sola vez, al final.
3. **Comparar contra el baseline.** El modelo tiene que superar con claridad al "promedio del cultivo"
   en todas las campañas. Si no lo hace en alguna, investigar esa campaña (¿nubes? ¿un año extremo?).
4. **Revisar el sesgo por campaña.** Un sesgo positivo grande en el año seco significa que el modelo
   no aprendió a detectar sequías; en un sistema de crédito es el error más caro.
5. **Separar por zona y por cultivo.** Agregar la columna de zona al CSV y calcular el MAE por zona a
   partir de `predicciones_validacion.csv`. Una zona con error muy distinto indica un sesgo regional.
6. **Revisar los lotes con más error** en el dashboard: casi siempre aparecen problemas de datos
   (polígono equivocado, cultivo de segunda, rinde mal cargado, huecos de nubes).
7. **Validar el score, no solo el rinde.** En la tabla de riesgo estimado contra real, lo más
   importante es la celda **real "alto" / estimado "bajo"**, que tiene que quedar cerca de cero.
   El margen `k × RMSE` hace al score conservador a propósito: vas a ver muchos lotes con riesgo real
   "bajo" clasificados como "medio". Ajustar `k` y los umbrales con el aliado según cuánto pesa cada
   tipo de error.
8. **Comparar con la experiencia del aliado.** Si existen, contrastar las categorías de riesgo con
   lotes que efectivamente tuvieron problemas de pago (no con el rinde solamente).

## Criterio sugerido para pasar del MVP a un piloto
- MAE por cultivo dentro de los rangos de arriba en la campaña reservada.
- Supera al baseline en todas las campañas.
- Sesgo en el año seco por debajo del 15 % del rinde medio.
- Ningún caso, o casos aislados y explicables, con riesgo real "alto" estimado como "bajo".
