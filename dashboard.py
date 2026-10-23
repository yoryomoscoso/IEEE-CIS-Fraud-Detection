# -*- coding: utf-8 -*-
"""Dashboard de costos - Certamen 1 IEEE-CIS.

Ejecutar DESPUES de python Certamen1.py --etapa costos:
    streamlit run dashboard.py

Se muestran los costos en test, pero el umbral recomendado se elige solo
con predicciones fuera-de-fold (OOF) del conjunto de entrenamiento.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score

st.set_page_config(page_title="IEEE-CIS | Costos de fraude", page_icon="📊",
                   layout="wide")

BASE = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def cargar_resultados():
    # Version de entrega: solo resultados oficiales.
    carpeta = BASE / "resultados"
    archivos = {
        "cv": carpeta / "Probabilidades_CV_train.csv",
        "test": carpeta / "Probabilidades_Test.csv",
        "config": carpeta / "Configuracion_Costos.json",
        "sensibilidad": carpeta / "Sensibilidad_Costos.csv",
        "politicas": carpeta / "Comparacion_Costos.csv",
    }
    for nombre, ruta in archivos.items():
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontro {ruta.name}. Ejecuta primero: "
                                    "python Certamen1.py --etapa costos")
    cv = pd.read_csv(archivos["cv"])
    test = pd.read_csv(archivos["test"])
    configuracion = json.loads(archivos["config"].read_text(encoding="utf-8"))
    sensibilidad = pd.read_csv(archivos["sensibilidad"])
    politicas = pd.read_csv(archivos["politicas"])
    return cv, test, configuracion, sensibilidad, politicas


def metrica_costos(y, pred, fp, fn):
    tn, falsos_p, falsos_n, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(falsos_p), "FN": int(falsos_n),
            "TP": int(tp), "Costo": int(falsos_p*fp + falsos_n*fn),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0)}


def obtener_curva_cv(y, proba, fp, fn, umbrales):
    # Umbrales examinados solo sobre probabilidades fuera de fold de TRAIN.
    return np.array([metrica_costos(y, proba >= t, fp, fn)["Costo"]
                     for t in umbrales], dtype=np.int64)


st.title("Deteccion de fraude: del modelo a la decision")
st.caption("Certamen 1 | IEEE-CIS Fraud Detection | Umbral y costo de clasificacion")

st.sidebar.caption("Resultados oficiales | Certamen 1")
try:
    cv, test, configuracion, sensibilidad, politicas = cargar_resultados()
except (FileNotFoundError, ValueError, KeyError) as e:
    st.error(str(e))
    st.stop()

if configuracion.get("Prueba rapida", False):
    st.error("Se detectaron resultados de prueba en la carpeta oficial. "
             "Ejecuta: python Certamen1.py --etapa costos")
    st.stop()

if "y_real" not in test or "proba" not in test or "y_real" not in cv or "proba" not in cv:
    st.error("Faltan columnas y_real/proba en los CSV generados.")
    st.stop()

st.sidebar.header("Decisiones y costos")
c_fp = st.sidebar.slider("Costo de un falso positivo (CLP)", 1000, 30000,
                         int(configuracion.get("C_FP", 5000)), step=1000)
c_fn = st.sidebar.slider("Costo de un falso negativo (CLP)", 5000, 300000,
                         int(configuracion.get("C_FN", 50000)), step=5000)
umbral_manual = st.sidebar.slider("Umbral manual", 0.0, 1.0, 0.5, step=0.005)

y_cv = cv["y_real"].to_numpy(dtype=int)
p_cv = cv["proba"].to_numpy(dtype=float)
y_test = test["y_real"].to_numpy(dtype=int)
p_test = test["proba"].to_numpy(dtype=float)
umbral_teorico = c_fp / (c_fp + c_fn)

umbrales = np.linspace(0.0, 1.0, 501)
curva = obtener_curva_cv(y_cv, p_cv, c_fp, c_fn, umbrales)
umbral_emp_cv_oof = float(umbrales[int(np.argmin(curva))])
# Para el escenario base, respetar el umbral oficial de TunedThresholdClassifierCV,
# guardado por el proceso de costes. Para otros costos, utilizar OOF de train.
valores_base = (c_fp == configuracion["C_FP"] and c_fn == configuracion["C_FN"])
umbral_empirico = (float(configuracion["Umbral empirico CV"])
                   if valores_base else umbral_emp_cv_oof)
fuente_empirico = ("TunedThresholdClassifierCV (escenario base)" if valores_base
                   else "Busqueda sobre predicciones OOF de TRAIN")

st.write(f"**Modelo ganador por F1 CV:** {configuracion['Modelo ganador']} "
         f"({configuracion['Estrategia ganadora']}). "
         "**La seleccion del modelo y de los umbrales no utiliza test.**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Umbral teorico", f"{umbral_teorico:.3f}")
c2.metric("Umbral empirico (TRAIN CV)", f"{umbral_empirico:.3f}")
c3.metric("Umbral manual", f"{umbral_manual:.3f}")
c4.metric("Relacion C_FN / C_FP", f"{c_fn/c_fp:.1f}x")
st.caption(f"Fuente del umbral empirico actual: {fuente_empirico}.")

politicas_actuales = [
    ("Todo 0", np.zeros(len(test), dtype=int)),
    ("Umbral 0.5", (p_test >= 0.5).astype(int)),
    ("Umbral teorico", (p_test >= umbral_teorico).astype(int)),
    ("Umbral empirico", (p_test >= umbral_empirico).astype(int)),
    ("Umbral manual", (p_test >= umbral_manual).astype(int)),
]
if "proba_pesos" in test.columns:
    # Ponderado: clasificacion con umbral 0.5, comparador exigido en la pauta.
    politicas_actuales.append(("Modelo con pesos (0.5)",
                               (test["proba_pesos"].to_numpy() >= 0.5).astype(int)))

filas = []
for titulo, prediccion in politicas_actuales:
    evaluacion = metrica_costos(y_test, prediccion, c_fp, c_fn)
    filas.append({"Politica": titulo, **evaluacion})
tabla = pd.DataFrame(filas)
costo_referencia = int(tabla.loc[tabla["Politica"] == "Umbral 0.5", "Costo"].iloc[0])
tabla["Ahorro vs 0.5"] = costo_referencia - tabla["Costo"]

st.subheader("Costo economico de las decisiones (TEST reservado)")
costo_empirico = int(tabla.loc[tabla["Politica"] == "Umbral empirico", "Costo"].iloc[0])
costo_manual = int(tabla.loc[tabla["Politica"] == "Umbral manual", "Costo"].iloc[0])
col1, col2, col3 = st.columns(3)
col1.metric("Costo: umbral 0.5", f"${costo_referencia:,.0f}")
col2.metric("Costo: umbral empirico", f"${costo_empirico:,.0f}",
            delta=f"Ahorro ${costo_referencia-costo_empirico:,.0f}")
col3.metric("Costo: umbral manual", f"${costo_manual:,.0f}",
            delta=f"Ahorro ${costo_referencia-costo_manual:,.0f}")

st.dataframe(tabla[["Politica", "TN", "FP", "FN", "TP", "Precision", "Recall", "Costo", "Ahorro vs 0.5"]],
             hide_index=True, use_container_width=True)

st.subheader("Curva costo vs. umbral (VALIDACION CRUZADA EN TRAIN)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=umbrales, y=curva, mode="lines", name="Costo estimado OOF"))
for x, nombre, color in [
    (0.5, "Umbral 0.5", "#8792A2"),
    (umbral_teorico, "Umbral teorico", "#D5A338"),
    (umbral_empirico, "Umbral empirico CV", "#168F83"),
    (umbral_manual, "Manual", "#7859B3"),
]:
    fig.add_vline(x=x, line_dash="dash", line_color=color,
                  annotation_text=nombre, annotation_position="top")
fig.update_layout(xaxis_title="Umbral de clasificacion", yaxis_title="Costo total CLP (OOF de TRAIN)",
                  legend_title_text="", height=420, margin=dict(t=50, b=30))
st.plotly_chart(fig, use_container_width=True)
st.caption("La curva usa una submuestra de TRAIN. Su costo absoluto no es directamente "
           "comparable con el costo de TEST mostrado arriba, porque contienen distintos registros.")

st.subheader("Matriz de confusion y costos del umbral seleccionado")
opciones = tabla["Politica"].tolist()
seleccion = st.selectbox("Politica para inspeccionar", opciones, index=4)
r = tabla.loc[tabla["Politica"] == seleccion].iloc[0]
matriz = pd.DataFrame([[int(r["TN"]), int(r["FP"])], [int(r["FN"]), int(r["TP"]) ]],
                      index=["Real 0 (legitima)", "Real 1 (fraude)"],
                      columns=["Predicha 0", "Predicha 1"])
left, right = st.columns([1, 1])
with left:
    st.dataframe(matriz, use_container_width=True)
with right:
    st.metric("Recall", f"{r['Recall']:.3f}")
    st.metric("Precision", f"{r['Precision']:.3f}")
    st.markdown(f"FP: {int(r['FP']):,} x ${c_fp:,} = **${int(r['FP'])*c_fp:,}**")
    st.markdown(f"FN: {int(r['FN']):,} x ${c_fn:,} = **${int(r['FN'])*c_fn:,}**")

st.subheader("Sensibilidad economica")
st.caption("Cinco escenarios definidos por el grupo. Resultados guardados para los costos "
           "originales; los controles superiores permiten explorar otros valores en tiempo real.")
st.dataframe(sensibilidad, hide_index=True, use_container_width=True)

st.divider()
st.caption("Limitaciones: costos FP/FN son supuestos del grupo; la ausencia de SMOTE/pesos "
           "no asegura probabilidades perfectamente calibradas. Cambiar el umbral manual en "
           "este dashboard es una simulacion, no una nueva seleccion de modelo con TEST.")
