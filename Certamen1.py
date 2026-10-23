# -*- coding: utf-8 -*-
"""
CERTAMEN 1 - ENSEMBLES, COSTO DE CLASIFICACION Y UMBRAL OPTIMO
IEEE-CIS Fraud Detection | Python 3.12 | Visual Studio Code

VERSION DEPURADA basada en Certamen1.ipynb del grupo y en las clases de
Danilo Gomez Correa (Pipelines, Bagging/Boosting, clasificacion sensible al costo).

Se respetan la muestra estratificada de 250.000 transacciones, los 30 predictores,
la semilla 2026, cinco modelos x dos estrategias, F1 CV y costos ilustrativos
FP=$5.000 / FN=$50.000 CLP definidos por el grupo. No se inventan resultados.

EJECUTAR DESDE LA TERMINAL, EN LA CARPETA DEL PROYECTO:
    python Certamen1.py --etapa datos
    python Certamen1.py --etapa modelos
    python Certamen1.py --etapa evaluar
    python Certamen1.py --etapa costos

La etapa de entrenamiento puede durar horas. Los modelos se guardan por separado;
si se interrumpe, se continua sin volver a entrenar los que ya terminaron.
Para una prueba de funcionamiento (NO para la entrega):
    python Certamen1.py --etapa datos --prueba-rapida
    python Certamen1.py --etapa modelos --prueba-rapida

IMPORTANTE: la seleccion del mejor modelo y del umbral nunca usa el test.
El conjunto test se evalua solo despues de fijar los modelos por F1 CV.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score,
                             brier_score_loss)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     TunedThresholdClassifierCV, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline as PipelineSK
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEMILLA = 2026
N_MUESTRA = 250_000
N_BUSQUEDA = 30_000
N_ITERACIONES = 10
N_FOLDS = 3
N_JOBS = 2                # evita multiplicar el consumo de RAM en el PC local
C_FP = 5_000             # supuesto academico original del grupo
C_FN = 50_000            # supuesto academico original del grupo

VARIABLES = [
    "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14", "D1", "D10", "D15", "V95", "V130", "V279"
]
NOMBRES = ["Decision Tree", "Random Forest", "XGBoost", "AdaBoost", "LightGBM"]
ESTRATEGIAS = ["SMOTE", "Pesos"]


def carpetas(prueba=False):
    """Mantiene los datos originales y separa pruebas de resultados oficiales."""
    base = Path(__file__).resolve().parent
    salida = base / ("prueba_rapida" if prueba else "resultados")
    modelo = salida / "modelos"
    for carpeta in (base / "data", salida, modelo):
        carpeta.mkdir(parents=True, exist_ok=True)
    return base / "data", salida, modelo


def parametros_ejecucion(prueba=False):
    if prueba:
        print("\n*** MODO PRUEBA: NO UTILIZAR ESTOS RESULTADOS EN LA ENTREGA ***")
        return 4_000, 800, 2, 2
    return N_MUESTRA, N_BUSQUEDA, N_ITERACIONES, N_FOLDS


def cargar_datos(prueba=False):
    """1. Problema y base. 2. Exploracion y particion 80/20.

    La lectura selectiva de columnas es una optimizacion de memoria, NO una
    limpieza aprendida. Imputaciones y codificacion ocurren en cada fold.
    """
    datos_dir, salida, _ = carpetas(prueba)
    archivo = datos_dir / "train_transaction.csv"
    if not archivo.is_file():
        raise FileNotFoundError(f"Falta {archivo}; debe ser el CSV extraido (no un ZIP).")
    columnas_originales = list(pd.read_csv(archivo, nrows=0).columns)
    esperadas = VARIABLES + ["isFraud", "TransactionID"]
    faltantes = sorted(set(esperadas) - set(columnas_originales))
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}. ¿Se descomprimio el CSV?")

    print("\n1. PROBLEMA DE NEGOCIO Y BASE IEEE-CIS")
    print("Columnas del archivo original:", len(columnas_originales))
    df = pd.read_csv(archivo, usecols=esperadas, low_memory=False)
    print("Registros originales:", len(df))
    print("Fraudes originales:", int(df["isFraud"].sum()))
    print("Porcentaje de fraudes: %.2f%%" % (100 * df["isFraud"].mean()))
    print("Duplicados con TransactionID:", int(df.duplicated(subset="TransactionID").sum()))
    print("Nulos en las variables elegidas:", int(df.isna().sum().sum()))
    print("\nNulos principales (%):")
    print((df[VARIABLES].isna().mean() * 100).sort_values(ascending=False).head(10).round(2))

    n_muestra, _, _, _ = parametros_ejecucion(prueba)
    if len(df) < n_muestra:
        raise ValueError(f"Dataset con {len(df)} filas, menor a la muestra de {n_muestra}.")
    # Misma decision del grupo: submuestra aleatoria estratificada ANTES del split.
    df_muestra, _ = train_test_split(df[VARIABLES + ["isFraud"]],
                                    train_size=n_muestra,
                                    stratify=df["isFraud"], random_state=42)
    del df
    df_muestra = df_muestra.reset_index(drop=True)
    if not prueba and len(df_muestra) <= 100_000:
        raise ValueError("La pauta exige mas de 100.000 registros despues de limpiar.")

    print("\n2. EXPLORACION Y PIPELINE")
    print("Muestra estratificada:", df_muestra.shape)
    print("Clase 0/1 (%):")
    print((df_muestra["isFraud"].value_counts(normalize=True) * 100).round(3))
    print("Variable objetivo: isFraud (1 = fraude)")
    print("Predictoras seleccionadas:", len(VARIABLES))
    print("Monto de transaccion:")
    print(df_muestra["TransactionAmt"].describe().round(2))
    df_muestra.to_csv(salida / "Muestra_estratificada.csv", index=False)
    print("Muestra guardada en:", salida / "Muestra_estratificada.csv")
    return df_muestra


def particion(prueba=False):
    """Particion estratificada; TEST permanece reservado hasta evaluar."""
    _, salida, _ = carpetas(prueba)
    archivo = salida / "Muestra_estratificada.csv"
    if not archivo.exists():
        raise FileNotFoundError("Ejecuta primero: python Certamen1.py --etapa datos")
    df = pd.read_csv(archivo)
    X, y = df[VARIABLES], df["isFraud"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEMILLA
    )
    return X_train, X_test, y_train, y_test


def crear_preprocesador(X_train):
    """El profesor: imputacion dentro de ColumnTransformer y Pipeline."""
    num = X_train.select_dtypes(include=np.number).columns.tolist()
    cat = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    print(f"Predictores: {len(num)} numericos y {len(cat)} categoricos")
    prep_num = PipelineSK([("imputar", SimpleImputer(strategy="median"))])
    prep_cat = PipelineSK([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("numericas", prep_num, num), ("categoricas", prep_cat, cat)
    ])


def crear_modelo(nombre, estrategia, proporcion, preprocesador):
    """3. SMOTE / 4. Pesos. Misma familia y grillas del notebook grupal."""
    pesos = estrategia == "Pesos"
    if nombre == "Decision Tree":
        estimador = DecisionTreeClassifier(random_state=SEMILLA,
                                           class_weight="balanced" if pesos else None)
        grilla = {"modelo__max_depth": [5, 10, 15, 20, None],
                  "modelo__min_samples_leaf": [1, 5, 10, 20]}
    elif nombre == "Random Forest":
        estimador = RandomForestClassifier(random_state=SEMILLA, n_jobs=1,
                                          class_weight="balanced" if pesos else None)
        grilla = {"modelo__n_estimators": [100, 150, 200],
                  "modelo__max_depth": [10, 20, None],
                  "modelo__min_samples_leaf": [1, 5, 10]}
    elif nombre == "XGBoost":
        estimador = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                  tree_method="hist", random_state=SEMILLA, n_jobs=1,
                                  scale_pos_weight=proporcion if pesos else 1)
        grilla = {"modelo__n_estimators": [100, 200, 300],
                  "modelo__max_depth": [3, 5, 7],
                  "modelo__learning_rate": [0.05, 0.10, 0.20]}
    elif nombre == "AdaBoost":
        # Mantener max_depth=1 en AMBAS estrategias. Diagnostico real del grupo:
        # class_weight='balanced' en el stump fallo en 20/30 ajustes CV.
        # Para la estrategia Pesos, aplicar sample_weight en AdaBoost.fit,
        # calculado SOLO con y de entrenamiento (nunca con el conjunto TEST).
        # NOTA: esta solucion ponderada NO cumple literalmente el requisito
        # de class_weight='balanced' dentro del arbol: documentar la excepcion.
        arbol = DecisionTreeClassifier(max_depth=1, random_state=SEMILLA)
        estimador = AdaBoostClassifier(estimator=arbol, random_state=SEMILLA)
        grilla = {"modelo__n_estimators": [50, 100, 150, 200],
                  "modelo__learning_rate": [0.05, 0.10, 0.50, 1.0]}
    elif nombre == "LightGBM":
        estimador = LGBMClassifier(objective="binary", random_state=SEMILLA,
                                   n_jobs=1, verbosity=-1,
                                   class_weight="balanced" if pesos else None)
        grilla = {"modelo__n_estimators": [100, 200, 300],
                  "modelo__max_depth": [5, 10, -1],
                  "modelo__learning_rate": [0.05, 0.10, 0.20]}
    else:
        raise ValueError(f"Modelo no reconocido: {nombre}")

    pasos = [("prep", clone(preprocesador))]
    if estrategia == "SMOTE":
        pasos.append(("smote", SMOTE(random_state=SEMILLA)))
    pasos.append(("modelo", estimador))
    return Pipeline(pasos), grilla


def id_modelo(nombre, estrategia):
    return f"{nombre.replace(' ', '')}_{estrategia}"


def entrenar_modelos(prueba=False):
    """3-4. Optimiza y entrena; seleccion solo con F1 CV. No consulta TEST."""
    _, salida, modelos_dir = carpetas(prueba)
    X_train, _, y_train, _ = particion(prueba)
    prep = crear_preprocesador(X_train)
    _, n_busqueda, n_iteraciones, n_folds = parametros_ejecucion(prueba)
    X_bus, _, y_bus, _ = train_test_split(X_train, y_train,
                                         train_size=min(n_busqueda, len(X_train)-1),
                                         stratify=y_train, random_state=SEMILLA)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)
    # Ponderacion de XGBoost calculada solo con TRAIN (no con el conjunto TEST).
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    tabla = []

    for estrategia in ESTRATEGIAS:
        for nombre in NOMBRES:
            codigo = id_modelo(nombre, estrategia)
            ruta_modelo = modelos_dir / f"{codigo}.joblib"
            ruta_info = modelos_dir / f"{codigo}.json"
            if ruta_modelo.exists() and ruta_info.exists():
                print(f"[RECUPERADO] {codigo}: entrenamiento ya guardado")
                registro = json.loads(ruta_info.read_text(encoding="utf-8"))
                tabla.append(registro)
                continue

            print(f"\n{'='*60}\nEntrenando: {nombre} | {estrategia}")
            pipeline, grilla = crear_modelo(nombre, estrategia, ratio, prep)
            excepcion_pauta = "Ninguna"
            tiempo_acumulado = 0.0
            mejor = None
            es_ada_pesos = nombre == "AdaBoost" and estrategia == "Pesos"
            parametros_fit_cv = {}
            parametros_fit_train = {}
            if es_ada_pesos:
                # Replica la alternativa viable del diagnostico V2 (10/10 en CV).
                # Los pesos se calculan unicamente con las etiquetas de TRAIN.
                # RandomizedSearchCV distribuye el vector sample_weight entre folds.
                parametros_fit_cv = {
                    "modelo__sample_weight": compute_sample_weight(
                        class_weight="balanced", y=y_bus
                    )
                }
                parametros_fit_train = {
                    "modelo__sample_weight": compute_sample_weight(
                        class_weight="balanced", y=y_train
                    )
                }
                excepcion_pauta = (
                    "AdaBoost Pesos usa sample_weight balanceado en AdaBoost.fit; "
                    "arbol base max_depth=1, class_weight=None (no literal de pauta)"
                )
                print("NOTA ACADEMICA: AdaBoost con arbol max_depth=1 y pesos de "
                      "muestra; el arbol no tiene class_weight='balanced'.")

            errores = []
            for nro_intento, grilla_actual in enumerate([grilla]):
                busqueda = RandomizedSearchCV(
                    estimator=pipeline, param_distributions=grilla_actual,
                    n_iter=n_iteraciones, scoring="f1", cv=cv,
                    random_state=SEMILLA, n_jobs=N_JOBS, verbose=1,
                    error_score=np.nan, refit=False
                )
                inicio = time.perf_counter()
                busqueda.fit(X_bus, y_bus, **parametros_fit_cv)
                tiempo_acumulado += time.perf_counter() - inicio
                cv_resultados = pd.DataFrame(busqueda.cv_results_)
                validos = cv_resultados[cv_resultados["mean_test_score"].notna()].copy()
                fallidos = int(len(cv_resultados) - len(validos))
                if fallidos:
                    print(f"ADVERTENCIA: {fallidos} configuraciones CV fallidas en {codigo}.")
                if validos.empty:
                    errores.append(f"Intento {nro_intento+1}: sin parametros validos en CV")
                    continue
                # Intentar reentrenar segun el orden de F1 CV, por si un candidato
                # valido en los folds falla al ajustar TODO train.
                validos = validos.sort_values("mean_test_score", ascending=False)
                for _, candidato in validos.iterrows():
                    params_candidato = candidato["params"]
                    modelo_candidato = clone(pipeline).set_params(**params_candidato)
                    inicio_fit = time.perf_counter()
                    try:
                        modelo_candidato.fit(X_train, y_train, **parametros_fit_train)
                    except ValueError as exc:
                        errores.append(f"Fallo reentrenamiento: {str(exc)[:150]}")
                        continue
                    mejor = modelo_candidato
                    segundos_entrenamiento = time.perf_counter() - inicio_fit
                    mejor_param = params_candidato
                    mejor_cv = float(candidato["mean_test_score"])
                    mejor_std = float(candidato["std_test_score"])
                    break
                if mejor is not None:
                    break

            if mejor is None:
                raise RuntimeError(
                    f"No fue posible entrenar {codigo}. Errores: {errores}. "
                    "No se completara una tabla de 10 modelos con resultados inventados."
                )
            segundos_busqueda = tiempo_acumulado
            print("F1 CV:", round(mejor_cv, 4), "| parametros:", mejor_param)
            registro = {
                "Modelo": nombre, "Estrategia": estrategia,
                "F1 CV": mejor_cv, "Desv. F1 CV": mejor_std,
                "Tiempo busqueda (s)": segundos_busqueda,
                "Tiempo entrenamiento (s)": segundos_entrenamiento,
                "Configuraciones fallidas": fallidos,
                "Excepcion pauta": excepcion_pauta,
                "Mejores parametros": str(mejor_param),
                "Parametros modelo (json)": {k.removeprefix("modelo__"): v
                                           for k, v in mejor_param.items()}
            }
            # Guardar checkpoints: si cae VS Code, se puede continuar.
            joblib.dump(mejor, ruta_modelo, compress=3)
            ruta_info.write_text(json.dumps(registro, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            tabla.append(registro)
            pd.DataFrame(tabla).to_csv(salida / "Tabla_CV_parcial.csv", index=False)

    df = pd.DataFrame(tabla).sort_values("F1 CV", ascending=False).reset_index(drop=True)
    df.to_csv(salida / "Tabla_CV_10_combinaciones.csv", index=False)
    print("\n5. MEJOR MODELO SELECCIONADO UNICAMENTE POR CV")
    print(df[["Modelo", "Estrategia", "F1 CV", "Desv. F1 CV"]].to_string(index=False))
    print("Ganador provisional de CV:", df.iloc[0]["Modelo"], df.iloc[0]["Estrategia"])
    return df


def evaluar_modelos(prueba=False):
    """5. Evaluacion en test una vez fijados los 10 modelos y ganador por CV."""
    _, salida, modelos_dir = carpetas(prueba)
    tabla_cv = salida / "Tabla_CV_10_combinaciones.csv"
    if not tabla_cv.exists():
        raise FileNotFoundError("Primero ejecuta --etapa modelos (10 modelos completos).")
    tabla = pd.read_csv(tabla_cv)
    if len(tabla) != 10:
        raise RuntimeError("Faltan combinaciones: deben ser exactamente 10.")
    X_train, X_test, y_train, y_test = particion(prueba)
    resultados = []
    ganador = tabla.iloc[0]  # FIJADO ANTES DE ABRIR EL TEST
    for _, fila in tabla.iterrows():
        nombre, estrategia = fila["Modelo"], fila["Estrategia"]
        archivo = modelos_dir / f"{id_modelo(nombre, estrategia)}.joblib"
        modelo = joblib.load(archivo)
        pred_train = modelo.predict(X_train)
        pred_test = modelo.predict(X_test)
        prob_test = modelo.predict_proba(X_test)[:, 1]
        resultados.append({
            "Modelo": nombre, "Estrategia": estrategia,
            "F1 train": f1_score(y_train, pred_train),
            "F1 test": f1_score(y_test, pred_test),
            "Brecha train-test": f1_score(y_train, pred_train) - f1_score(y_test, pred_test),
            "Precision": precision_score(y_test, pred_test, zero_division=0),
            "Recall": recall_score(y_test, pred_test, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob_test)
        })
        del modelo
    tabla_final = tabla.merge(pd.DataFrame(resultados), on=["Modelo", "Estrategia"])
    tabla_final = tabla_final[[
        "Modelo", "Estrategia", "Mejores parametros", "F1 train", "F1 CV",
        "Desv. F1 CV", "F1 test", "Brecha train-test", "Recall", "Precision",
        "ROC-AUC", "Tiempo busqueda (s)", "Tiempo entrenamiento (s)",
        "Configuraciones fallidas", "Excepcion pauta"
    ]].sort_values("F1 CV", ascending=False)
    tabla_final.to_csv(salida / "Comparacion_10_Modelos.csv", index=False)
    print("\n5. TABLA COMPARATIVA FINAL")
    print(tabla_final.round(4).to_string(index=False))
    print("\nGanador por F1 CV:", ganador["Modelo"], "|", ganador["Estrategia"])
    return tabla_final


def costo_total(y_real, y_pred, costo_fp, costo_fn):
    """6. Matriz de costos (TP y TN tienen costo cero en este supuesto)."""
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    return int(fp*costo_fp + fn*costo_fn)


def scorer_costo(costo_fp, costo_fn):
    """Puntaje mayor = costo monetario menor."""
    return make_scorer(lambda yt, yp: -costo_total(yt, yp, costo_fp, costo_fn))


def fila_costos(nombre, y_real, y_pred, costo_fp, costo_fn, costo_ref):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    costo = costo_total(y_real, y_pred, costo_fp, costo_fn)
    return {"Politica": nombre, "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
            "Precision": precision_score(y_real, y_pred, zero_division=0),
            "Recall": recall_score(y_real, y_pred, zero_division=0),
            "F1": f1_score(y_real, y_pred, zero_division=0),
            "Costo total (CLP)": costo, "Ahorro vs 0.5 (CLP)": int(costo_ref-costo)}


def umbral_por_probabilidades(y_real, probabilidades, costo_fp, costo_fn, pasos=501):
    """Umbral empírico de exploracion sobre probabilidades OOF SOLO DE TRAIN."""
    umbrales = np.linspace(0, 1, pasos)
    costes = [costo_total(y_real, probabilidades >= t, costo_fp, costo_fn)
              for t in umbrales]
    indice = int(np.argmin(costes))
    return float(umbrales[indice])


def evaluar_costos(prueba=False):
    """6-8. Costos, threshold teorico/empirico, sensibilidad sin usar test al optimizar."""
    _, salida, modelos_dir = carpetas(prueba)
    tabla = salida / "Comparacion_10_Modelos.csv"
    if not tabla.exists():
        raise FileNotFoundError("Primero ejecuta --etapa evaluar.")
    tabla = pd.read_csv(tabla)
    X_train, X_test, y_train, y_test = particion(prueba)
    preprocesador = crear_preprocesador(X_train)
    ganador = tabla.iloc[0]
    ganador_nombre = ganador["Modelo"]
    # En esta parte, segun pauta, se conservan hiperparametros pero se retira SMOTE/pesos.
    modelo_ganador = joblib.load(modelos_dir / f"{id_modelo(ganador_nombre, ganador['Estrategia'])}.joblib")
    # Rescatar SOLO los hiperparametros elegidos por CV, no parametros accesorios
    # del estimador ajustado, que pueden cambiar entre versiones de bibliotecas.
    hiperparametros = modelo_ganador.named_steps["modelo"].get_params()
    hiper_seleccionados = {k: hiperparametros[k] for k in
                          ["max_depth", "min_samples_leaf", "n_estimators", "learning_rate"]
                          if k in hiperparametros}
    pipe_real, _ = crear_modelo(ganador_nombre, "Real", proporcion=1,
                               preprocesador=preprocesador)
    # Mantener la parametrizacion del modelo escogido por CV (sin sus pesos).
    ajustes = {f"modelo__{k}": v for k, v in hiper_seleccionados.items()
               if k in pipe_real.named_steps["modelo"].get_params()}
    for k in ["modelo__class_weight", "modelo__scale_pos_weight", "modelo__is_unbalance"]:
        ajustes.pop(k, None)
    # AdaBoost: el arbol interno del ganador puede tener class_weight.
    if ganador_nombre == "AdaBoost":
        ajustes.pop("modelo__estimator", None)
    pipe_real.set_params(**ajustes)
    print("\n7. Reentrenando modelo con proporcion real, sin SMOTE ni pesos")
    pipe_real.fit(X_train, y_train)
    joblib.dump(pipe_real, modelos_dir / "Modelo_probabilidades_reales.joblib", compress=3)
    prob_test = pipe_real.predict_proba(X_test)[:, 1]
    _, n_busqueda, _, n_folds = parametros_ejecucion(prueba)
    X_bus, _, y_bus, _ = train_test_split(X_train, y_train,
                                         train_size=min(n_busqueda, len(X_train)-1),
                                         stratify=y_train, random_state=SEMILLA)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)

    # OOF evita usar probabilidades de entrenamiento vistas por el modelo.
    print("Calculando probabilidades out-of-fold para sensibilidad y dashboard...")
    prob_cv = cross_val_predict(clone(pipe_real), X_bus, y_bus, cv=cv,
                                method="predict_proba", n_jobs=N_JOBS)[:, 1]
    pd.DataFrame({"y_real": y_bus.to_numpy(), "proba": prob_cv}).to_csv(
        salida / "Probabilidades_CV_train.csv", index=False)
    brier = brier_score_loss(y_bus, prob_cv)
    print("Brier en submuestra de entrenamiento por CV:", round(brier, 5))
    print("AVISO: quitar SMOTE/pesos no garantiza calibracion perfecta.")

    # TunedThresholdClassifierCV, como en notebook de clasificacion sensible al costo.
    umbral_teorico = C_FP / (C_FP + C_FN)
    escenarios = [("Bajo", C_FP, 10_000), ("Moderado", C_FP, 25_000),
                  ("Base", C_FP, C_FN), ("Alto", C_FP, 100_000),
                  ("Extremo", C_FP, 200_000)]
    tabla_escenarios = []
    umbral_empirico = None
    for etiqueta, cfp, cfn in escenarios:
        umbral_teo = cfp/(cfp+cfn)
        print(f"Optimizando costo CV: {etiqueta} (FP={cfp:,}, FN={cfn:,})")
        tuned = TunedThresholdClassifierCV(
            estimator=clone(pipe_real), scoring=scorer_costo(cfp, cfn),
            response_method="predict_proba", thresholds=100,
            cv=cv, n_jobs=N_JOBS, refit=True
        )
        tuned.fit(X_bus, y_bus)  # SOLO TRAIN (submuestra estratificada)
        emp = float(tuned.best_threshold_)
        if etiqueta == "Base":
            umbral_empirico = emp
        pred = (prob_test >= emp).astype(int)
        referencia = costo_total(y_test, prob_test >= 0.5, cfp, cfn)
        tabla_escenarios.append({
            "Escenario": etiqueta, "C_FP": cfp, "C_FN": cfn,
            "Relacion FN/FP": cfn/cfp, "Umbral teorico": umbral_teo,
            "Umbral empirico CV": emp, "FP": int(((pred==1)&(y_test==0)).sum()),
            "FN": int(((pred==0)&(y_test==1)).sum()),
            "Costo total (CLP)": costo_total(y_test, pred, cfp, cfn),
            "Ahorro vs 0.5 (CLP)": referencia-costo_total(y_test, pred, cfp, cfn)
        })
    sensibilidad = pd.DataFrame(tabla_escenarios)
    sensibilidad.to_csv(salida / "Sensibilidad_Costos.csv", index=False)

    # Comparar la etapa B: mejor modelo de estrategia ponderada por F1 CV.
    ganador_pesos = tabla[tabla["Estrategia"] == "Pesos"].sort_values("F1 CV", ascending=False).iloc[0]
    pipe_pesos = joblib.load(modelos_dir / f"{id_modelo(ganador_pesos['Modelo'], 'Pesos')}.joblib")
    prob_pesos = pipe_pesos.predict_proba(X_test)[:, 1]
    pred_pesos = pipe_pesos.predict(X_test)
    costo_base = costo_total(y_test, prob_test >= 0.5, C_FP, C_FN)
    politicas = {
        "Todo 0": np.zeros(len(y_test), dtype=int),
        "Umbral 0.5": (prob_test >= 0.5).astype(int),
        "Umbral teorico": (prob_test >= umbral_teorico).astype(int),
        "Umbral empirico CV": (prob_test >= umbral_empirico).astype(int),
        f"{ganador_pesos['Modelo']} Pesos": pred_pesos
    }
    tabla_politicas = pd.DataFrame([
        fila_costos(n, y_test, p, C_FP, C_FN, costo_base)
        for n, p in politicas.items()
    ])
    tabla_politicas.to_csv(salida / "Comparacion_Costos.csv", index=False)
    (salida / "Configuracion_Costos.json").write_text(
        json.dumps({"C_FP": C_FP, "C_FN": C_FN,
                    "Umbral teorico": float(umbral_teorico),
                    "Umbral empirico CV": float(umbral_empirico),
                    "Modelo ganador": str(ganador_nombre),
                    "Estrategia ganadora": str(ganador["Estrategia"]),
                    "Estrategia pesos comparada": str(ganador_pesos["Modelo"]),
                    "Muestra": 250000 if not prueba else 4000,
                    "Prueba rapida": bool(prueba)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    pd.DataFrame({"y_real": y_test.to_numpy(), "proba": prob_test,
                  "proba_pesos": prob_pesos}).to_csv(
                      salida / "Probabilidades_Test.csv", index=False)
    print("\n7. COMPARACION ECONOMICA EN TEST")
    print(tabla_politicas.round(4).to_string(index=False))
    print("\n8. SENSIBILIDAD ECONOMICA EN TEST (umbrales elegidos en CV)")
    print(sensibilidad.round(4).to_string(index=False))
    print("\nResultados guardados:", salida)


def main():
    parser = argparse.ArgumentParser(description="Certamen 1 IEEE-CIS - etapas reproducibles")
    parser.add_argument("--etapa", choices=["datos", "modelos", "evaluar", "costos", "todo"],
                        default="datos", help="Por seguridad, solo audita los datos por defecto")
    parser.add_argument("--prueba-rapida", action="store_true",
                        help="Prueba de ejecucion reducida (NO usar para la evaluacion)")
    args = parser.parse_args()
    np.random.seed(SEMILLA)
    print("CERTAMEN 1 - Danilo Gomez Correa | Grupo IEEE-CIS")
    print("Etapa solicitada:", args.etapa)
    if args.etapa in ["datos", "todo"]:
        cargar_datos(args.prueba_rapida)
    if args.etapa in ["modelos", "todo"]:
        entrenar_modelos(args.prueba_rapida)
    if args.etapa in ["evaluar", "todo"]:
        evaluar_modelos(args.prueba_rapida)
    if args.etapa in ["costos", "todo"]:
        evaluar_costos(args.prueba_rapida)
    print("\nEtapa completada.")


if __name__ == "__main__":
    main()
