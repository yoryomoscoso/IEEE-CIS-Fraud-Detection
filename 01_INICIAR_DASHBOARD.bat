@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
color 0B
title Certamen 1 IEEE-CIS - Dashboard de fraude

echo.
echo ============================================================
echo   CERTAMEN 1 - IEEE-CIS FRAUD DETECTION
echo   Miguel Gonzalez / Jorge Moscoso / Francisca Palacios
echo ============================================================
echo.
echo Se iniciara el dashboard con resultados oficiales.
echo No se necesita dataset ni volver a entrenar modelos.
echo.

if not exist "dashboard.py" goto archivo_faltante
if not exist "resultados\Probabilidades_Test.csv" goto archivo_faltante
if not exist "resultados\Probabilidades_CV_train.csv" goto archivo_faltante
if not exist "resultados\Configuracion_Costos.json" goto archivo_faltante

if exist ".venv\Scripts\python.exe" goto preparar_dependencias

echo [1/3] Creando entorno Python local (.venv)...
rem Primero se intenta Python 3.12, recomendado para este proyecto.
py -3.12 -m venv ".venv" >nul 2>&1
if exist ".venv\Scripts\python.exe" goto preparar_dependencias
py -3 -m venv ".venv" >nul 2>&1
if exist ".venv\Scripts\python.exe" goto preparar_dependencias
python -m venv ".venv" >nul 2>&1
if exist ".venv\Scripts\python.exe" goto preparar_dependencias

echo.
echo ERROR: No se encontro Python para Windows.
echo Instala Python 3.12 y vuelve a abrir este archivo.
echo https://www.python.org/downloads/
goto fin_error

:preparar_dependencias
set "PY=.venv\Scripts\python.exe"
"%PY%" -c "import streamlit, plotly, sklearn, pandas, numpy" >nul 2>&1
if not errorlevel 1 goto iniciar

echo [2/3] Instalando dependencias (requiere internet la primera vez)...
"%PY%" -m pip install --disable-pip-version-check -r "requirements_dashboard.txt"
if errorlevel 1 (
    echo.
    echo ERROR: No se pudieron instalar las dependencias.
    echo Comprueba tu conexion a Internet y vuelve a intentarlo.
    goto fin_error
)

:iniciar
echo [3/3] Iniciando Streamlit...
echo.
echo Se abrira el navegador con el simulador oficial.
echo Direccion habitual: http://localhost:8501
echo Cierra esta ventana para detener el dashboard.
echo.
"%PY%" -m streamlit run "dashboard.py" --server.headless false --browser.gatherUsageStats false
if errorlevel 1 goto fin_error
goto fin

:archivo_faltante
echo ERROR: Faltan archivos del proyecto.
echo Extrae TODO el ZIP antes de hacer doble clic en este BAT.
goto fin_error

:fin_error
echo.
echo Si necesitas ayuda, abre 00_LEER_PRIMERO.html o GUIA_RAPIDA.txt.
pause
exit /b 1

:fin
endlocal
exit /b 0
