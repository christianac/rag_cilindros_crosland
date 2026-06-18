#!/bin/bash
# Lanzador de la GUI Streamlit en Linux/Mac
echo "========================================"
echo "  RAG Cilindros - Interfaz Gráfica"
echo "========================================"
echo

# Verificar que existe .env
if [ ! -f .env ]; then
    echo "[ADVERTENCIA] No se encontró archivo .env"
    echo "Copia .env.example a .env y configura tus API keys"
    echo
fi

# Iniciar Streamlit
echo "Iniciando Streamlit..."
echo "La interfaz se abrirá en tu navegador"
echo "Para detener: Ctrl+C"
echo

streamlit run app_gui.py
