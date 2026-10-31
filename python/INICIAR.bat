@echo off
chcp 65001 >nul
title Traductor en vivo
cd /d "%~dp0"
echo Revisando instalacion (unos segundos)...
python -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo No se pudo instalar. Revisa que tengas Python instalado y conexion a internet.
  pause
  exit /b
)
echo Instalacion lista. Iniciando...
python traductor.py
pause
