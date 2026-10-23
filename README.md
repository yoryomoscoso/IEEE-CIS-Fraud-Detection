# IEEE-CIS Fraud Detection | Certamen 1 de Machine Learning

**Detección de fraudes financieros: del modelo de ensamble a la decisión económica**

**Integrantes:** Miguel González · Jorge Moscoso · Francisca Palacios  
**Profesor:** Danilo Gómez Correa

## Descripción

Este proyecto compara algoritmos de clasificación para identificar transacciones potencialmente fraudulentas usando el conjunto de datos [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection). Además de evaluar su capacidad predictiva, analizamos el costo de los falsos positivos y falsos negativos y el efecto de cambiar el umbral de clasificación.

El repositorio incluye el código del experimento, un notebook de Google Colab, resultados obtenidos por el grupo y un simulador interactivo en **Streamlit**.

## Resultados principales

| Indicador | Resultado |
|---|---:|
| Transacciones del dataset original | 590.540 |
| Muestra estratificada utilizada | 250.000 |
| Proporción de fraudes | 3,5 % |
| Variables predictoras | 30 |
| Partición entrenamiento / prueba | 80 % / 20 % |
| Combinaciones evaluadas | 10 (5 algoritmos × 2 estrategias) |
| Modelo seleccionado por F1 CV | **XGBoost + pesos** |
| F1 CV del ganador | **0,4870** |
| F1 en test del ganador | **0,4882** |
| Recall de fraude del ganador | **74,8 %** |
| ROC-AUC del ganador | **0,9313** |

Los cinco algoritmos son **Decision Tree, Random Forest, XGBoost, AdaBoost y LightGBM**. Cada uno se evaluó con **SMOTE** o **ponderación de clases/observaciones**. El criterio de selección fue el F1 en validación cruzada estratificada de tres folds; **no se utilizó test para seleccionar el ganador**.

### Evaluación económica

Se emplearon costos **supuestos para el ejercicio**, de **$5.000 CLP por falso positivo (FP)** y **$50.000 CLP por falso negativo (FN)**.

| Política | Costo en 50.000 transacciones de test |
|---|---:|
| Umbral 0,5 | $45.370.000 CLP |
| Umbral empírico por CV (≈0,0404) | $35.400.000 CLP |
| Umbral teórico (≈0,0909) | $32.035.000 CLP |

El umbral empírico permitió un ahorro observado de **$9.970.000 CLP (≈22 %)** frente al umbral 0,5. El umbral teórico obtuvo un costo observado menor en test, pero **no fue elegido optimizando directamente sobre test**. Estas cifras son simulaciones bajo los supuestos indicados, **no ahorros bancarios reales**.

## Estructura de la entrega

```text
.
├── 00_LEER_PRIMERO.html             # Portada e instrucciones
├── 01_INICIAR_DASHBOARD.bat         # Iniciador Windows del simulador
├── Certamen1.py                     # Código de preparación, modelos y costos
├── dashboard.py                     # Aplicación Streamlit
├── GUIA_RAPIDA.txt                  # Instrucciones abreviadas
├── requirements_dashboard.txt       # Librerías para Streamlit
├── requirements_entrenamiento.txt   # Librerías de entrenamiento
├── Resultados_Certamen1.zip         # Resultados para recuperar en Colab
├── notebook/
│   └── Certamen1_IEEE_CIS_Dashboard_Colab.ipynb
├── resultados/                      # Métricas, escenarios y probabilidades
├── graficos/                        # Siete visualizaciones exportadas
└── documentacion/                   # Ficha técnica y otras aclaraciones
```

## Cómo visualizar el dashboard (Windows)

1. Descargar el repositorio como ZIP y **extraer todos los archivos**.
2. Hacer doble clic en **`01_INICIAR_DASHBOARD.bat`**.
3. La primera ejecución puede requerir **Python** (se recomienda 3.12) y conexión a internet para instalar las dependencias.
4. El navegador normalmente abrirá **http://localhost:8501**.

**Alternativa desde terminal**, en la raíz del proyecto y con las dependencias instaladas:

```powershell
python -m pip install -r requirements_dashboard.txt
python -m streamlit run dashboard.py
```

El dashboard utiliza los CSV/JSON que vienen en `resultados/`. **No requiere descargar el dataset ni reentrenar los modelos**. Permite modificar costos FP/FN, mover el umbral y examinar matrices de confusión y resultados económicos.

## Cómo abrir el notebook de Google Colab

1. Abrir `notebook/Certamen1_IEEE_CIS_Dashboard_Colab.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Para revisar resultados sin reentrenar, seleccionar **`MODO = 'resultados'`** y subir **`Resultados_Certamen1.zip`** cuando se solicite.
3. Para reproducir el entrenamiento completo, utilizar el modo **`entrenar`** y proporcionar el archivo de datos de Kaggle según las indicaciones del notebook.

> Las sesiones de Colab son temporales. Guardar los resultados antes de cerrar o reiniciar el entorno.

## Ejecución del código Python por etapas

El script principal dispone de las siguientes etapas:

```powershell
python Certamen1.py --etapa datos     # Carga, análisis y muestra estratificada
python Certamen1.py --etapa modelos   # Búsqueda y entrenamiento de 10 combinaciones
python Certamen1.py --etapa evaluar   # Métricas de train y test
python Certamen1.py --etapa costos    # Umbrales, costos y sensibilidad
```

Para esta opción hace falta disponer del archivo **`data/train_transaction.csv`**, descargado desde Kaggle. **El dataset original no está incluido en el repositorio.** El entrenamiento completo consume considerablemente más recursos y tiempo que abrir el dashboard.

## Metodología y precauciones

- La imputación numérica (mediana), categórica (moda) y la codificación `OneHotEncoder` se realizan dentro de un `ColumnTransformer` incluido en el `Pipeline`.
- En la estrategia SMOTE, el sobremuestreo ocurre **dentro de cada fold de entrenamiento**. Esto ayuda a evitar *data leakage* por aplicar transformaciones a datos de validación.
- La selección del modelo utiliza F1 CV; el conjunto test queda reservado para la evaluación final.
- Los umbrales se estudian con datos de entrenamiento y validación cruzada; test se usa para reportar su desempeño.
- Para analizar probabilidades, se vuelve a entrenar la configuración seleccionada sin SMOTE ni pesos. **Esto por sí solo no garantiza calibración perfecta**.
- **Excepción AdaBoost:** la estrategia ponderada empleó `sample_weight` en `AdaBoost.fit()` con árbol base de profundidad 1, en vez de `class_weight="balanced"` dentro del árbol. Es una adaptación técnica documentada que **no equivale literalmente** a ese requisito de la pauta.
- Los resultados oficiales son los exportados de la ejecución del grupo en Colab; versiones de bibliotecas o nuevas ejecuciones pueden producir diferencias.

## Uso de inteligencia artificial

Se utilizó **ChatGPT (OpenAI)** como apoyo para estructurar y depurar código, revisar conceptos, preparar documentación y desarrollar el simulador y material de presentación. El grupo es responsable de las decisiones metodológicas, la ejecución de los experimentos, la interpretación de los resultados y las conclusiones. Esta declaración debe reflejar el uso efectivamente realizado y ajustarse a las indicaciones del curso.

## Fuente de datos

[IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection). El acceso y uso del dataset se encuentran sujetos a las condiciones de Kaggle y de la competencia.
