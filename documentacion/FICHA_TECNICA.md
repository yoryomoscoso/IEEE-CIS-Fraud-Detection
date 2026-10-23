# Ficha técnica — Certamen 1 IEEE-CIS

**Docente:** Danilo Gómez Correa  
**Grupo:** Miguel González, Jorge Moscoso y Francisca Palacios  
**Fuente:** [IEEE-CIS Fraud Detection, Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection)

## Diseño del experimento
- Datos originales: **590.540 transacciones**; variable objetivo `isFraud` (3,5 % positiva).
- Muestra estratificada: **250.000 transacciones**; **30 predictores**.
- Partición: **80 % train / 20 % test**, estratificada; semilla **2026**.
- Preprocesamiento dentro de `ColumnTransformer` y `Pipeline`: imputación numérica mediana, categórica moda, codificación One-Hot.
- Cinco modelos: Decision Tree, Random Forest, XGBoost, AdaBoost, LightGBM; dos estrategias: SMOTE y ponderación.
- Comparación mediante F1 minoritaria en validación cruzada, tres folds. Ajuste de hiperparámetros con `RandomizedSearchCV`.

## Resultados oficiales (exportados desde Google Colab)
- **Ganador por F1 CV:** XGBoost + Pesos (F1 CV **0,486969**, F1 test **0,488160**).
- Umbral teórico: **0,090909**; umbral empírico de CV: **0,040401**.
- Costos académicos supuestos: FP $5.000 y FN $50.000 CLP.
- Costos en TEST: umbral 0,5 **$45.370.000**; umbral empírico **$35.400.000**; umbral teórico **$32.035.000**.
- **Ahorro observado** para umbral empírico frente a 0,5: **$9.970.000** (≈22 %) en el test, no ahorro anual real.

## Advertencias metodológicas
1. El ganador se selecciona con **F1 CV**, no comparando rendimientos en test.
2. El umbral empírico se escoge usando CV sobre datos de entrenamiento; test solo se utiliza para evaluación.
3. Para el análisis de probabilidades se reentrena sin SMOTE ni pesos, manteniendo los hiperparámetros elegidos. Esto **no garantiza calibración perfecta**.
4. AdaBoost ponderado utiliza `sample_weight` balanceado y árbol base de profundidad 1; **no corresponde literalmente** a `class_weight='balanced'` dentro del árbol exigido por la pauta. Esta diferencia se documenta para la revisión docente.
5. Los costos son **supuestos académicos**, no datos económicos reales de la competencia.
6. Los resultados oficiales del paquete son los exportados del Colab del grupo. Otras ejecuciones locales pueden presentar diferencias de configuración o versión de librerías.

## Cómo revisar
- **Colab:** cargar el notebook y `Resultados_Certamen1.zip` en modo `resultados`; los controles del simulador funcionan sin reentrenar.
- **Windows:** ejecutar `01_INICIAR_DASHBOARD.bat` con los resultados locales incluidos.
- **Reentrenamiento:** requiere el dataset de Kaggle y ejecución de las etapas del notebook en modo `entrenar` o del script Python.
