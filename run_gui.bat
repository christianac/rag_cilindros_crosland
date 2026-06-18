@echo off
REM Lanzador de la GUI Streamlit en Windows
echo ========================================
echo   RAG Cilindros - Interfaz Grafica
echo ========================================
echo.

REM Verificar que existe .env
if not exist .env (
    echo [ADVERTENCIA] No se encontro archivo .env
    echo Copia .env.example a .env y configura tus API keys
    echo.
    pause
)

REM Iniciar Streamlit
echo Iniciando Streamlit...
echo La interfaz se abrira en tu navegador
echo Para detener: Ctrl+C
echo.

streamlit run app_gui.py
pause
