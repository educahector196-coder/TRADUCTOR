# Traductor en vivo — versión Python (respaldo)

Escucha **directamente el sonido de tu laptop** (Teams, videos, lo que suene), transcribe y traduce.
No necesita compartir pantalla ni «Mezcla estéreo», y no se detiene sola.

## Usarlo
1. Descarga esta carpeta `python` (o todo el repositorio: botón verde **Code → Download ZIP** y descomprime).
2. Doble clic en **INICIAR.bat**.
   - La primera vez instala lo necesario y descarga el modelo de voz (~500 MB). Tarda unos minutos.
   - Las siguientes veces abre en segundos.
3. Elige el idioma que se escucha y al que quieres traducir. Listo.

## Controles
- **Sensibilidad**: súbela si no capta voces bajas; bájala si escribe ruidos. La barra verde es el sonido; la línea amarilla es el límite: lo que pasa la línea se toma como voz.
- **Solo traducción**, **A+ / A−**, **Siempre encima** (para tenerlo sobre Teams).
- Todo se guarda solo en la carpeta `transcripciones`.

## Si algo falla
- **Va con retraso**: abre `traductor.py` con el Bloc de notas y cambia `MODELO = "small"` por `MODELO = "base"`.
- **«python no se reconoce»**: instala Python 3.11 o 3.12 desde python.org y marca **Add Python to PATH**.
