# Limitaciones del modelo

**El score es una herramienta de apoyo al análisis, no una decisión de crédito.** Siempre tiene que
complementarse con información del productor, el historial, las garantías y el criterio del analista.

## Nubes y disponibilidad de imágenes
- En años húmedos, o en el período crítico de verano, puede haber semanas sin imágenes válidas. La
  curva se interpola en esos tramos y **el pico real puede quedar fuera de la serie**. El NDVI del pico
  queda entonces subestimado y el rinde estimado tiende a ser menor.
- La máscara SCL no es perfecta: los bordes de nubes, las nubes finas y las sombras a veces se
  clasifican como válidos. La mediana por lote y el umbral de fracción válida mitigan el problema,
  pero no lo eliminan.
- Cada lote informa `n_obs_validas`, `max_hueco_dias` y `frac_dias_hueco_largo`. **Un lote con huecos
  largos tiene una estimación menos confiable**, aunque el score no lo penaliza en forma explícita
  (solo lo hace a través del error medio del modelo).

## Falta de datos
- **El modelo necesita rindes reales de varias campañas.** Con menos de 3 campañas, la validación por
  campaña es muy inestable. Con menos de unos 30 lotes por cultivo y campaña, las métricas por
  cultivo tienen mucho ruido.
- Los rindes informados por productores o acopios pueden tener errores (rinde promedio del
  establecimiento en lugar del lote, lotes mal georreferenciados, cultivos de segunda informados como
  de primera).
- Sin fecha de siembra se usa una ventana fija por cultivo. Esto mezcla siembras tempranas y tardías
  y puede desplazar las variables de "días al pico".
- NASA POWER tiene una resolución de unos 50 km: no capta tormentas locales, granizo ni anegamientos
  puntuales.

## Sesgos por zona, cultivo y año
- **Sesgo por zona:** el modelo aprende las relaciones NDVI-rinde de las zonas que vio. Aplicarlo a
  otras regiones (NOA, NEA, sudoeste bonaerense) sin datos de esas zonas puede dar errores grandes.
  Conviene reportar las métricas por zona antes de usarlo en una región nueva.
- **Sesgo por cultivo:** soja de primera, soja de segunda, maíz temprano y maíz tardío tienen curvas
  distintas. El MVP solo distingue `soja`/`maiz`. Los cultivos de segunda, o lotes con dos cultivos en
  la misma campaña, rompen los supuestos de la ventana de campaña.
- **Años extremos:** si el entrenamiento no incluye una sequía como la de 2022/23, el modelo tiende a
  **sobreestimar** el rinde en una campaña así. Esto se ve en el sesgo por campaña, incluso en el
  ejemplo sintético. Es justo el caso en el que más importa el score.
- **Saturación del NDVI:** en canopeos densos (maíz de alto potencial) el NDVI satura cerca de 0,85-0,9
  y discrimina mal entre rindes altos. EVI podría ayudar, pero hoy no se usa en el modelo.

## Lo que el modelo no ve
- Manejo (genética, fertilización, control de malezas y plagas), granizo después del pico,
  pérdidas de cosecha y calidad comercial.
- Precios futuros, tipo de cambio, retenciones y costos reales: el score usa **valores de referencia
  fijos** que hay que actualizar en `config/config.yaml`.
- La situación financiera del productor: otras deudas, superficie total, diversificación.

## Sobre el dataset sintético
- Las métricas sobre el ejemplo salen buenas **por construcción**. No sirven como evidencia de que el
  enfoque funcione.
- Las zonas, los factores de lluvia por campaña y los rindes son inventados, aunque buscan valores
  plausibles.

## Sobre la implementación
- La descarga desde Planetary Computer y NASA POWER **no se probó contra los servicios reales** en el
  entorno de desarrollo. Hay que validarla con unos pocos lotes conocidos antes de procesar volúmenes grandes.
- El proveedor de Copernicus Data Space está documentado pero no implementado.
- Elegir el modelo con la misma validación con la que se reportan las métricas es levemente optimista.
  Con datos reales, conviene reservar una campaña final que no se use hasta la evaluación definitiva.
