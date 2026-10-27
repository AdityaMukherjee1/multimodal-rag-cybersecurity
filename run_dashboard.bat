@echo off
echo Starting Evidence-Grounded Multimodal RAG Research Dashboard...
cd /d "%~dp0"
.\venv\Scripts\streamlit.exe run frontend\app.py
pause
