# -*- coding: utf-8 -*-
"""
TRADUCTOR EN VIVO (versión Python, respaldo de la página web)
--------------------------------------------------------------
- Escucha DIRECTAMENTE el sonido de la laptop (lo que dicen en Teams, un video, etc.).
  No necesita compartir pantalla ni activar "Mezcla estéreo".
- Transcribe con Whisper (en tu PC) y traduce con Google Translate.
- Nunca se detiene sola: si cambia el audio (audífonos, parlantes) se reconecta.
- Guarda todo automáticamente en la carpeta "transcripciones".

Solo Windows. Ejecuta INICIAR.bat (instala todo la primera vez).
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import warnings
warnings.filterwarnings("ignore")
import queue
import threading
import time
import collections
import datetime
import traceback

import numpy as np
import tkinter as tk
from tkinter import ttk

# ======================= CONFIGURACIÓN =======================
MODELO = "base"         # "tiny" = rapidísimo · "base" = rápido (recomendado) · "small" = más preciso pero lento
FRAGMENTO_MAX = 4.0     # segundos máximos por frase antes de transcribir (menos = aparece antes)
PAUSA_FIN_FRASE = 0.45  # segundos de silencio que marcan el fin de una frase
# =============================================================

IDIOMAS = {
    "Inglés": "en", "Español": "es", "Portugués": "pt", "Francés": "fr",
    "Alemán": "de", "Italiano": "it", "Chino": "zh", "Japonés": "ja",
    "Coreano": "ko", "Ruso": "ru", "Árabe": "ar", "Neerlandés": "nl",
}
FRASES_FANTASMA = {  # cosas que Whisper "inventa" en silencio
    "thank you.", "thanks for watching!", "thank you for watching.", "you", ".",
    "subtítulos realizados por la comunidad de amara.org", "gracias.", "¡gracias por ver!",
    "subtitles by the amara.org community", "bye.", "música", "[music]", "(music)",
}

SR = 16000  # Whisper trabaja a 16 kHz


def log(msg):
    """Muestra en la ventana negra qué está pasando."""
    try:
        print(datetime.datetime.now().strftime("[%H:%M:%S] ") + str(msg), flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1) CAPTURA DEL SONIDO DE LA PC (WASAPI loopback) con reconexión automática
# ---------------------------------------------------------------------------
class CapturaPC(threading.Thread):
    def __init__(self, cola_audio, estado):
        super().__init__(daemon=True)
        self.cola = cola_audio
        self.estado = estado
        self.dispositivo_actual = None

    def _abrir(self, pa, pyaudio):
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        salida = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        disp = salida
        if not salida.get("isLoopbackDevice"):
            for lb in pa.get_loopback_device_info_generator():
                if salida["name"] in lb["name"]:
                    disp = lb
                    break
            else:
                raise RuntimeError("No encontré el audio de salida: " + salida["name"])
        canales = max(1, int(disp["maxInputChannels"]))
        tasa = int(disp["defaultSampleRate"])

        def callback(in_data, frame_count, time_info, status):
            self.cola.put((in_data, canales, tasa))
            return (None, pyaudio.paContinue)

        stream = pa.open(format=pyaudio.paFloat32, channels=canales, rate=tasa,
                         input=True, input_device_index=disp["index"],
                         frames_per_buffer=int(tasa * 0.05), stream_callback=callback)
        stream.start_stream()
        return stream, salida["name"], disp["name"]

    def run(self):
        import pyaudiowpatch as pyaudio
        while True:
            pa = stream = None
            try:
                pa = pyaudio.PyAudio()
                stream, nombre_salida, nombre = self._abrir(pa, pyaudio)
                self.dispositivo_actual = nombre_salida
                self.estado(f"🔊 Escuchando: {nombre_salida}")
                # vigilar: si cambia la salida predeterminada o el stream muere, reconectar
                while True:
                    time.sleep(2)
                    if not stream.is_active():
                        raise RuntimeError("el audio se detuvo")
                    pa2 = pyaudio.PyAudio()
                    try:
                        w = pa2.get_host_api_info_by_type(pyaudio.paWASAPI)
                        actual = pa2.get_device_info_by_index(w["defaultOutputDevice"])["name"]
                    finally:
                        pa2.terminate()
                    if actual != nombre_salida:
                        self.estado(f"🔁 Cambió la salida de audio a: {actual}. Reconectando…")
                        break
            except Exception as e:
                self.estado(f"🔁 Reconectando audio… ({e})")
                time.sleep(1.5)
            finally:
                try:
                    if stream:
                        stream.stop_stream(); stream.close()
                except Exception:
                    pass
                try:
                    if pa:
                        pa.terminate()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 2) CORTE EN FRASES (detecta voz y pausas)
# ---------------------------------------------------------------------------
def a_16k_mono(raw, canales, tasa):
    x = np.frombuffer(raw, dtype=np.float32)
    if canales > 1:
        x = x.reshape(-1, canales).mean(axis=1)
    if tasa != SR and len(x):
        n = int(len(x) * SR / tasa)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


class Segmentador(threading.Thread):
    def run(self):
        while True:
            try:
                self._run()
            except Exception as e:
                self.app.cola_ui.put(("estado_modelo", 0, "", f"🔁 Reiniciando detector de voz… ({e})"))
                time.sleep(1)

    def __init__(self, cola_audio, cola_frases, app):
        super().__init__(daemon=True)
        self.cola_audio = cola_audio
        self.cola_frases = cola_frases
        self.app = app

    piso = 0.0005   # ruido de fondo estimado (se ajusta solo)

    def umbral(self):
        s = self.app.sens_val                    # 1..10 (copiado desde la ventana)
        factor = 6.0 - 0.45 * (s - 1)            # sens 1 -> 6x el ruido · sens 10 -> ~2x el ruido
        return max(0.0004, self.piso * factor, 0.012 * (0.62 ** (s - 1)) if s < 3 else 0.0)

    def _ajustar_piso(self, nivel):
        # sube despacio y baja rápido: sigue al silencio de fondo, no a la voz
        if nivel < self.piso:
            self.piso = 0.7 * self.piso + 0.3 * max(nivel, 0.0001)
        else:
            self.piso = 0.998 * self.piso + 0.002 * nivel

    def _run(self):
        buf, dur = [], 0.0
        previo = collections.deque(maxlen=8)    # ~0.4 s antes de que empiece la voz
        hablando = False
        ultima_voz = 0.0      # reloj: por si deja de llegar audio (nada suena)
        silencio = 0.0        # segundos de silencio medidos en el propio audio
        while True:
            try:
                raw, ch, tasa = self.cola_audio.get(timeout=0.2)
                x = a_16k_mono(raw, ch, tasa)
            except queue.Empty:
                x = None
            except Exception:
                continue
            ahora = time.time()
            if x is not None and len(x):
                nivel = float(np.sqrt(np.mean(x * x)))
                self.app.nivel = nivel
                self.app.umbral_actual = self.umbral()
                if not hablando:
                    self._ajustar_piso(nivel)
                if nivel > self.umbral():
                    if not hablando:
                        buf = list(previo); dur = sum(len(b) for b in buf) / SR
                    hablando = True
                    ultima_voz = ahora
                    silencio = 0.0
                elif hablando:
                    silencio += len(x) / SR
                if hablando:
                    buf.append(x); dur += len(x) / SR
                else:
                    previo.append(x)
            else:
                self.app.nivel = 0.0
            if ahora - getattr(self, "_ultimo_aviso", 0) > 10:
                self._ultimo_aviso = ahora
                log(f"   nivel de sonido: {self.app.nivel:.4f}  (límite para voz: {self.umbral():.4f})")
            sin_audio = x is None and (ahora - ultima_voz) > PAUSA_FIN_FRASE
            if hablando and (silencio >= PAUSA_FIN_FRASE or sin_audio or dur >= FRAGMENTO_MAX):
                silencio = 0.0
                audio = np.concatenate(buf) if buf else None
                buf, dur, hablando = [], 0.0, False
                if audio is not None and len(audio) > SR * 0.35:
                    pico = float(np.max(np.abs(audio))) or 1.0
                    audio = np.clip(audio * (0.9 / pico), -1, 1).astype(np.float32)  # normaliza volumen
                    self.cola_frases.put((datetime.datetime.now().strftime("%H:%M:%S"), audio))
                    self.app.frases_detectadas += 1
                    log(f"🎤 Frase detectada ({len(audio)/SR:.1f} s) — transcribiendo…")


# ---------------------------------------------------------------------------
# 3) TRANSCRIPCIÓN (Whisper en tu PC) y 4) TRADUCCIÓN
# ---------------------------------------------------------------------------
class Transcriptor(threading.Thread):
    def __init__(self, cola_frases, cola_trad, app):
        super().__init__(daemon=True)
        self.cola_frases = cola_frases
        self.cola_trad = cola_trad
        self.app = app

    def run(self):
        self.app.estado_modelo(f"⏳ Cargando modelo de voz «{MODELO}» (la 1.ª vez se descarga, espera unos minutos)…")
        from faster_whisper import WhisperModel
        modelo = None
        while modelo is None:
            try:
                modelo = WhisperModel(MODELO, device="cpu", compute_type="int8",
                                      cpu_threads=max(2, (os.cpu_count() or 4) - 1))
            except Exception as e:
                self.app.estado_modelo(f"🔁 Reintentando cargar el modelo… ({e})")
                time.sleep(5)
        self.app.estado_modelo("✅ Modelo de voz listo")
        n = 0
        while True:
            hora, audio = self.cola_frases.get()
            partes = [audio]
            while not self.cola_frases.empty() and sum(len(p) for p in partes) < SR * 15:
                _, extra = self.cola_frases.get_nowait()
                partes.append(extra)
            if len(partes) > 1:
                audio = np.concatenate(partes)
                log(f"   (poniéndose al día: {len(partes)} frases juntas)")
            try:
                t0 = time.time()
                segs, _ = modelo.transcribe(audio, language=self.app.idioma_origen(), beam_size=1,
                                            best_of=1, temperature=0.0, vad_filter=False,
                                            without_timestamps=True, condition_on_previous_text=False,
                                            no_speech_threshold=0.6)
                segs = list(segs)
                tarda = time.time() - t0
                if tarda > len(audio) / SR:
                    self.app.estado_modelo(f"⚠️ La PC tarda {tarda:.1f}s por frase: si se atrasa, pon MODELO = \"tiny\"")
                else:
                    self.app.estado_modelo(f"✅ Velocidad OK ({tarda:.1f}s por frase)")
                texto = " ".join(s.text.strip() for s in segs
                                 if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)).strip()
            except Exception as e:
                self.app.estado_modelo(f"🔁 Error al transcribir, sigo… ({e})")
                continue
            if not texto or texto.lower() in FRASES_FANTASMA:
                log("   (no se entendió voz en esa frase)")
                continue
            log("   📝 " + texto)
            n += 1
            self.app.cola_ui.put(("orig", n, hora, texto))
            self.cola_trad.put((n, hora, texto))


# Usar los certificados de Windows: así las conexiones seguras funcionan
# aunque un antivirus (McAfee, Kaspersky…) inspeccione el tráfico.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import json
import urllib.parse
import urllib.request


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read().decode("utf-8"))


def _google(texto, src, dst):
    url = ("https://translate.googleapis.com/translate_a/single?client=gtx&dt=t"
           f"&sl={src}&tl={dst}&q=" + urllib.parse.quote(texto))
    j = _get_json(url)
    return "".join(p[0] for p in j[0] if p and p[0])


def _mymemory(texto, src, dst):
    url = ("https://api.mymemory.translated.net/get?langpair="
           + urllib.parse.quote(f"{src}|{dst}") + "&q=" + urllib.parse.quote(texto))
    j = _get_json(url)
    t = j.get("responseData", {}).get("translatedText")
    if not t or "MYMEMORY WARNING" in t:
        raise RuntimeError("MyMemory sin respuesta")
    return t


def _deep(texto, src, dst):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source=src, target=dst).translate(texto)


RUTAS = [("Google", _google), ("Google (librería)", _deep), ("MyMemory", _mymemory)]


class Traductor(threading.Thread):
    def __init__(self, cola_trad, app):
        super().__init__(daemon=True)
        self.cola_trad = cola_trad
        self.app = app
        self.ruta_buena = 0   # recuerda qué servicio funcionó

    def traducir(self, texto, src, dst):
        orden = list(range(self.ruta_buena, len(RUTAS))) + list(range(0, self.ruta_buena))
        errores = []
        for i in orden:
            nombre, fn = RUTAS[i]
            try:
                t = fn(texto, src, dst)
                if t:
                    if i != self.ruta_buena:
                        self.ruta_buena = i
                        self.app.estado_modelo(f"🌎 Traduciendo con: {nombre}")
                    return t
            except Exception as e:
                errores.append(f"{nombre}: {type(e).__name__} {str(e)[:100]}")
        log("   ❌ No se pudo traducir. Detalles: " + " | ".join(errores))
        return None

    def run(self):
        while True:
            item = self.cola_trad.get()
            threading.Thread(target=self._una, args=item, daemon=True).start()

    def _una(self, n, hora, texto):
        if True:
            src, dst = self.app.idioma_origen(), self.app.idioma_destino()
            trad = texto if src == dst else None
            for intento in range(3):
                if trad:
                    break
                trad = self.traducir(texto, src, dst)
                if not trad:
                    time.sleep(0.8 * (intento + 1))
            log("   🌎 " + (trad or "[sin traducción — revisa tu internet]"))
            self.app.cola_ui.put(("trad", n, hora, trad or "[sin traducción — revisa tu internet]"))


# ---------------------------------------------------------------------------
# 5) VENTANA
# ---------------------------------------------------------------------------
class App:
    BG, PANEL, TXT, MUTED, ACC = "#0f1115", "#181b22", "#e8eaf0", "#8b92a5", "#5b8cff"

    def __init__(self, root):
        self.root = root
        self.cola_ui = queue.Queue()
        self.nivel = 0.0
        self.fuente = 20
        self.sens_val, self.src_code, self.dst_code = 8.0, "en", "es"
        self.frases_detectadas = 0
        self.archivo = self._archivo_nuevo()

        root.title("Traductor en vivo — Teams")
        root.geometry("1200x650")
        root.configure(bg=self.BG)
        root.protocol("WM_DELETE_WINDOW", self._cerrar)

        barra = tk.Frame(root, bg=self.PANEL, padx=10, pady=8)
        barra.pack(fill="x")
        estilo = dict(bg=self.PANEL, fg=self.TXT, font=("Segoe UI", 10))

        tk.Label(barra, text="Idioma que se escucha", **estilo).pack(side="left")
        self.cb_src = ttk.Combobox(barra, values=list(IDIOMAS), width=11, state="readonly")
        self.cb_src.set("Inglés"); self.cb_src.pack(side="left", padx=4)
        tk.Button(barra, text="⇄", command=self._intercambiar, bg="#222632", fg=self.TXT,
                  relief="flat", padx=8).pack(side="left", padx=2)
        tk.Label(barra, text="Traducir a", **estilo).pack(side="left", padx=(6, 0))
        self.cb_dst = ttk.Combobox(barra, values=list(IDIOMAS), width=11, state="readonly")
        self.cb_dst.set("Español"); self.cb_dst.pack(side="left", padx=4)
        for cb in (self.cb_src, self.cb_dst):
            cb.bind("<<ComboboxSelected>>", lambda e: self._titulos())

        tk.Label(barra, text="   Sensibilidad", **estilo).pack(side="left")
        self.sensibilidad = tk.DoubleVar(value=8)
        tk.Scale(barra, from_=1, to=10, resolution=0.5, orient="horizontal", variable=self.sensibilidad,
                 length=110, showvalue=False, bg=self.PANEL, fg=self.TXT, highlightthickness=0,
                 troughcolor="#222632").pack(side="left")

        self.medidor = tk.Canvas(barra, width=120, height=10, bg="#222632", highlightthickness=0)
        self.medidor.pack(side="left", padx=8)
        self.barra_nivel = self.medidor.create_rectangle(0, 0, 0, 10, fill="#3ecf8e", width=0)
        self.linea_umbral = self.medidor.create_line(0, 0, 0, 10, fill="#ffd38a", width=2)

        for txt, cmd in (("A+", lambda: self._letra(4)), ("A−", lambda: self._letra(-4)),
                         ("Solo traducción", self._solo), ("Limpiar", self._limpiar)):
            b = tk.Button(barra, text=txt, command=cmd, bg="#222632", fg=self.TXT, relief="flat", padx=8)
            b.pack(side="right", padx=3)
            if txt == "Solo traducción":
                self.btn_solo = b
        self.encima = tk.BooleanVar(value=False)
        tk.Checkbutton(barra, text="Siempre encima", variable=self.encima, command=self._encima,
                       bg=self.PANEL, fg=self.TXT, selectcolor="#222632", activebackground=self.PANEL
                       ).pack(side="right", padx=6)

        self.panel = tk.PanedWindow(root, orient="horizontal", bg="#2a2f3a", sashwidth=2, bd=0)
        self.panel.pack(fill="both", expand=True)
        self.col_orig, self.txt_orig, self.tit_orig = self._columna()
        self.col_trad, self.txt_trad, self.tit_trad = self._columna()
        self.panel.add(self.col_orig, stretch="always")
        self.panel.add(self.col_trad, stretch="always")
        self._titulos()
        self.solo = False

        pie = tk.Frame(root, bg=self.PANEL, padx=10, pady=4)
        pie.pack(fill="x")
        self.lbl_audio = tk.Label(pie, text="Iniciando audio…", bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 9))
        self.lbl_audio.pack(side="left")
        self.lbl_modelo = tk.Label(pie, text="", bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 9))
        self.lbl_modelo.pack(side="left", padx=20)
        self.lbl_frases = tk.Label(pie, text="", bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 9))
        self.lbl_frases.pack(side="left", padx=10)
        tk.Label(pie, text=f"Se guarda en: {os.path.relpath(self.archivo)}", bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9)).pack(side="right")

        self._copiar_controles()
        # hilos de trabajo
        cola_audio, cola_frases, cola_trad = queue.Queue(maxsize=2000), queue.Queue(), queue.Queue()
        CapturaPC(cola_audio, self.estado_audio).start()
        Segmentador(cola_audio, cola_frases, self).start()
        Transcriptor(cola_frases, cola_trad, self).start()
        Traductor(cola_trad, self).start()
        self._tick()

    # ----- ayudantes de interfaz -----
    def _columna(self):
        marco = tk.Frame(self.panel, bg=self.BG)
        tit = tk.Label(marco, text="", bg=self.BG, fg=self.MUTED, font=("Segoe UI", 9, "bold"), anchor="w", padx=12, pady=6)
        tit.pack(fill="x")
        txt = tk.Text(marco, bg=self.BG, fg=self.TXT, font=("Segoe UI", self.fuente), wrap="word",
                      relief="flat", padx=14, pady=8, insertbackground=self.TXT, spacing3=8)
        txt.pack(fill="both", expand=True)
        txt.tag_configure("hora", foreground=self.MUTED, font=("Segoe UI", 9))
        txt.configure(state="disabled")
        return marco, txt, tit

    def _titulos(self):
        self.tit_orig.config(text=f"{self.cb_src.get().upper()} — LO QUE SE DICE")
        self.tit_trad.config(text=f"{self.cb_dst.get().upper()} — TRADUCCIÓN")

    def _intercambiar(self):
        a, b = self.cb_src.get(), self.cb_dst.get()
        self.cb_src.set(b); self.cb_dst.set(a); self._titulos()

    def idioma_origen(self):
        return self.src_code

    def idioma_destino(self):
        return self.dst_code

    def _copiar_controles(self):
        # SOLO en el hilo principal: copiar los valores de la ventana a variables normales
        try:
            self.sens_val = float(self.sensibilidad.get())
            self.src_code = IDIOMAS.get(self.cb_src.get(), "en")
            self.dst_code = IDIOMAS.get(self.cb_dst.get(), "es")
        except Exception:
            pass

    def _letra(self, d):
        self.fuente = max(12, min(60, self.fuente + d))
        for t in (self.txt_orig, self.txt_trad):
            t.configure(font=("Segoe UI", self.fuente))

    def _solo(self):
        self.solo = not self.solo
        if self.solo:
            self.panel.forget(self.col_orig); self.btn_solo.config(text="Ver ambos")
        else:
            self.panel.forget(self.col_trad)
            self.panel.add(self.col_orig, stretch="always"); self.panel.add(self.col_trad, stretch="always")
            self.btn_solo.config(text="Solo traducción")

    def _encima(self):
        self.root.attributes("-topmost", self.encima.get())

    def _limpiar(self):
        for t in (self.txt_orig, self.txt_trad):
            t.configure(state="normal"); t.delete("1.0", "end"); t.configure(state="disabled")

    def _archivo_nuevo(self):
        carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripciones")
        os.makedirs(carpeta, exist_ok=True)
        return os.path.join(carpeta, "reunion-" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + ".txt")

    def _guardar(self, hora, orig, trad):
        try:
            with open(self.archivo, "a", encoding="utf-8") as f:
                f.write(f"[{hora}] {orig}\n           → {trad}\n")
        except Exception:
            pass

    def estado_audio(self, msg):
        log(msg)
        self.cola_ui.put(("estado_audio", 0, "", msg))

    def estado_modelo(self, msg):
        log(msg)
        self.cola_ui.put(("estado_modelo", 0, "", msg))

    def _escribir(self, txt, n, hora, texto, marcador=False):
        txt.configure(state="normal")
        txt.insert("end", hora + "  ", "hora")
        txt.insert("end", texto, (f"n{n}",))
        txt.insert("end", "\n")
        txt.configure(state="disabled")
        txt.see("end")

    def _tick(self):
        self._copiar_controles()
        try:
            while True:
                tipo, n, hora, texto = self.cola_ui.get_nowait()
                if tipo == "orig":
                    self._escribir(self.txt_orig, n, hora, texto)
                    self._escribir(self.txt_trad, n, hora, "…")
                    self._pendiente = getattr(self, "_pendiente", {}); self._pendiente[n] = texto
                elif tipo == "trad":
                    t = self.txt_trad
                    rng = t.tag_ranges(f"n{n}")
                    t.configure(state="normal")
                    if rng:
                        t.delete(rng[0], rng[1]); t.insert(rng[0], texto, (f"n{n}",))
                    t.configure(state="disabled"); t.see("end")
                    orig = getattr(self, "_pendiente", {}).pop(n, "")
                    self._guardar(hora, orig, texto)
                elif tipo == "estado_audio":
                    self.lbl_audio.config(text=texto)
                elif tipo == "estado_modelo":
                    self.lbl_modelo.config(text=texto)
        except queue.Empty:
            pass
        # medidor de nivel y línea del umbral
        w = 120
        import math
        def px(v):  # 0.0001 -> 0 · 0.3 -> w (escala logarítmica)
            return max(0, min(w, (math.log10(max(v, 1e-4)) + 4) / 3.5 * w))
        self.medidor.coords(self.barra_nivel, 0, 0, px(self.nivel), 10)
        umbral = getattr(self, "umbral_actual", 0.002)
        x = min(w - 1, px(umbral))
        self.medidor.coords(self.linea_umbral, x, 0, x, 10)
        self.lbl_frases.config(text=f"Frases detectadas: {self.frases_detectadas}")
        self.root.after(60, self._tick)

    def _cerrar(self):
        from tkinter import messagebox
        if messagebox.askyesno("Cerrar traductor", "¿Seguro que quieres cerrar el traductor?"):
            self.root.destroy()
            os._exit(0)


if __name__ == "__main__":
    try:
        log("Abriendo la ventana del traductor…")
        root = tk.Tk()
        App(root)
        log("✅ Ventana abierta. Si no la ves, búscala en la barra de tareas (icono de pluma).")
        root.mainloop()
    except Exception:
        traceback.print_exc()
        input("Ocurrió un error. Presiona Enter para salir…")
