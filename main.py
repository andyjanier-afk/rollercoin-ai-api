"""RollerCoin Expedition AI v3.0.

- Predicciones: estimaciones estadisticas, NO resultados garantizados.
- 'porcentaje' se recibe en escala 0..100, antes de conocer el resultado.
- El historial de la web puede ser incompleto: NO se usan rachas como predictors.
- La referencia es el porcentaje proporcionado por el frontend; no se
  supone que sea un valor oficial verificado por RollerCoin.
- Mantiene la estructura de las respuestas v2 para no romper el frontend.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rollercoin-ai")

VERSION = "3.0.0"
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
CORS_ORIGINS = [
    x.strip() for x in os.getenv(
        "CORS_ORIGINS", "https://rollercoin-ai-panel.vercel.app,http://localhost:3000,http://localhost:5173"
    ).split(",") if x.strip()
]
CACHE_SEGUNDOS = max(0, int(os.getenv("CACHE_SEGUNDOS", "15")))
REVALIDAR_CADA = max(10, int(os.getenv("REVALIDAR_CADA", "25")))
PAGINA = 1000
MIN_ENTRENAMIENTO = 30
MIN_SELECCION = 145
N_INICIAL_VALIDACION = 100
PASO_VALIDACION = 20
UMBRAL_MEJORA_BRIER = 0.008  # Exigir ventaja para apartarnos de la referencia

MAPAS_INFO = {
    "3h": {"horas": 3, "dificultad_orientativa": 2},
    "12h": {"horas": 12, "dificultad_orientativa": 5},
    "18h": {"horas": 18, "dificultad_orientativa": 8},
    "20h": {"horas": 20, "dificultad_orientativa": 6},
    "24h Dark": {"horas": 24, "dificultad_orientativa": 7},
    "24h Beach": {"horas": 24, "dificultad_orientativa": 8},
    "30h": {"horas": 30, "dificultad_orientativa": 10},
}

app = FastAPI(title="RollerCoin Expedition AI API", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class SimulacionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    hamster: str = Field(min_length=1, max_length=150)
    stats: float = Field(ge=0, le=1_000_000)
    expedicion: str
    porcentaje: float = Field(ge=0, le=100)

    @field_validator("hamster", "expedicion")
    @classmethod
    def normalizar_nombre(cls, texto: str) -> str:
        texto = texto.strip()
        if not texto:
            raise ValueError("No puede estar vacio")
        return texto

    @field_validator("expedicion")
    @classmethod
    def comprobar_mapa(cls, mapa: str) -> str:
        if mapa not in MAPAS_INFO:
            raise ValueError("Mapa desconocido. Valores: " + ", ".join(MAPAS_INFO))
        return mapa


@dataclass
class Estado:
    modelo: Pipeline | None
    modelo_nombre: str
    estado_modelo: str
    historial: pd.DataFrame
    entrenables: int
    descartadas: int
    evaluacion: dict[str, Any] | None
    ultimo_por_hamster: dict[str, int]
    derrotas_por_hamster: dict[str, int]
    envios_por_hamster: dict[str, int]
    huella: str = ""
    seleccion_evaluada_con: int = 0


def interpretar_booleano(valor: Any) -> int | None:
    if valor is None or pd.isna(valor):
        return None
    if isinstance(valor, (bool, np.bool_)):
        return int(valor)
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return int(valor) if math.isfinite(float(valor)) and float(valor) in (0, 1) else None
    texto = str(valor).strip().lower()
    if texto in {"true", "1", "1.0", "yes", "si", "sí", "vive", "vivió", "vivio"}:
        return 1
    if texto in {"false", "0", "0.0", "no", "muere", "murió", "murio"}:
        return 0
    return None


def convertir_numeros(columna: pd.Series) -> pd.Series:
    limpia = (columna.astype("string").str.strip()
              .str.replace("%", "", regex=False)
              .str.replace(",", ".", regex=False))
    return pd.to_numeric(limpia, errors="coerce")


def consultar_expediciones() -> pd.DataFrame:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_KEY en Render/entorno")
    from supabase import create_client
    cliente = create_client(SUPABASE_URL, SUPABASE_KEY)
    filas: list[dict[str, Any]] = []
    inicio = 0
    while True:
        consulta = (cliente.table("expediciones").select("*")
                    .order("fecha", desc=False).order("id", desc=False)
                    .range(inicio, inicio + PAGINA - 1).execute())
        lote = consulta.data or []
        filas.extend(lote)
        if len(lote) < PAGINA:
            break
        inicio += PAGINA
    return pd.DataFrame(filas)


def ordenar_registros(df: pd.DataFrame) -> pd.DataFrame:
    # La fecha tiene precision diaria: el id desempata. No asumimos que
    # created_at sea la hora real del resultado (importaciones masivas).
    return df.sort_values(["fecha", "id"], kind="stable").reset_index(drop=True)


def construir_historial(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0
    cols = {"id", "fecha", "hamster", "expedicion", "stats", "porcentaje", "finalizado"}
    faltan = cols - set(df.columns)
    if faltan:
        raise ValueError(f"Columnas faltantes en expediciones: {sorted(faltan)}")
    datos = ordenar_registros(df.copy())
    datos["hamster"] = datos["hamster"].fillna("").astype(str).str.strip()
    datos["expedicion"] = datos["expedicion"].fillna("").astype(str).str.strip()
    datos["finalizado_int"] = datos["finalizado"].map(interpretar_booleano)
    historial = datos.loc[datos["finalizado_int"].notna() & datos["hamster"].ne("")].copy()
    if historial.empty:
        return pd.DataFrame(), pd.DataFrame(), 0
    historial["finalizado_int"] = historial["finalizado_int"].astype(int)
    ml = historial.copy()
    ml["stats"] = convertir_numeros(ml["stats"])
    ml["porcentaje"] = convertir_numeros(ml["porcentaje"])
    es_valido = (
        ml["expedicion"].isin(MAPAS_INFO)
        & ml["stats"].notna() & ml["stats"].between(0, 1_000_000)
        & np.isfinite(ml["stats"].astype(float))
        & ml["porcentaje"].notna() & ml["porcentaje"].between(0, 100)
        & np.isfinite(ml["porcentaje"].astype(float))
    )
    ml = ml.loc[es_valido].copy()
    ml["porcentaje_norm"] = ml["porcentaje"].astype(float) / 100.0
    # No incorporamos nombre, rachas, ni orden global como variables predictivas:
    # la web no necesariamente contiene todos los envios del juego.
    return historial, ml, int(len(historial) - len(ml))


def columnas_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[["porcentaje_norm", "stats", "expedicion"]].copy()


def modelos_candidatos() -> dict[str, Pipeline]:
    log_solo = Pipeline([
        ("prep", ColumnTransformer([
            ("pct", StandardScaler(), ["porcentaje_norm"]),
        ], remainder="drop")),
        ("modelo", LogisticRegression(C=1.0, max_iter=1000)),
    ])
    log_mapa = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), ["porcentaje_norm", "stats"]),
            ("mapa", OneHotEncoder(handle_unknown="ignore"), ["expedicion"]),
        ], remainder="drop")),
        ("modelo", LogisticRegression(C=0.4, max_iter=1000)),
    ])
    bosque = Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", ["porcentaje_norm", "stats"]),
            ("mapa", OneHotEncoder(handle_unknown="ignore"), ["expedicion"]),
        ], remainder="drop")),
        ("modelo", RandomForestClassifier(
            n_estimators=160, max_depth=4, min_samples_leaf=8,
            max_features=0.8, random_state=42, n_jobs=-1,
        )),
    ])
    return {
        "logistica_porcentaje": log_solo,
        "logistica_mapa": log_mapa,
        "random_forest": bosque,
    }


def predecir_prob(modelo: Pipeline | None, X: pd.DataFrame) -> np.ndarray:
    if modelo is None:
        return X["porcentaje_norm"].astype(float).to_numpy()
    indice = list(modelo.classes_).index(1)
    return modelo.predict_proba(X)[:, indice]


def resumen_predicciones(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    prob = np.clip(np.asarray(prob, dtype=float), 0, 1)
    cls = (prob >= 0.5).astype(int)
    y_true = np.asarray(y_true, dtype=int)
    matriz = confusion_matrix(y_true, cls, labels=[0, 1])
    return {
        "exactitud_porc": round(100 * float(accuracy_score(y_true, cls)), 2),
        "brier": round(float(brier_score_loss(y_true, prob)), 5),
        "log_loss": round(float(log_loss(y_true, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])), 5),
        "matriz_confusion": {
            "verdaderos_negativos": int(matriz[0, 0]),
            "falsos_positivos": int(matriz[0, 1]),
            "falsos_negativos": int(matriz[1, 0]),
            "verdaderos_positivos": int(matriz[1, 1]),
        },
    }


def validacion_progresiva(ml: pd.DataFrame) -> dict[str, Any] | None:
    """Evaluacion interna SOLO en la parte anterior al holdout final."""
    if len(ml) < N_INICIAL_VALIDACION + 40:
        return None
    X = columnas_features(ml)
    y = ml["finalizado_int"].astype(int)
    estimaciones: dict[str, list[float]] = {**{k: [] for k in modelos_candidatos()}, "porcentaje_referencia": []}
    reales: list[int] = []
    for corte in range(N_INICIAL_VALIDACION, len(ml), PASO_VALIDACION):
        fin = min(corte + PASO_VALIDACION, len(ml))
        y_train = y.iloc[:corte]
        if y_train.nunique() != 2:
            continue
        y_batch = y.iloc[corte:fin]
        reales.extend(y_batch.tolist())
        estimaciones["porcentaje_referencia"].extend(X.iloc[corte:fin]["porcentaje_norm"].tolist())
        for nombre, plantilla in modelos_candidatos().items():
            estimador = clone(plantilla)
            estimador.fit(X.iloc[:corte], y_train)
            estimaciones[nombre].extend(predecir_prob(estimador, X.iloc[corte:fin]).tolist())
    if len(reales) < 40:
        return None
    y_arr = np.asarray(reales, dtype=int)
    puntajes = {k: resumen_predicciones(y_arr, np.asarray(v)) for k, v in estimaciones.items()}
    return {"numero_evaluado": len(reales), "resultados": puntajes}


def seleccionar_modelo(puntuaciones: dict[str, Any] | None) -> str:
    """Guardarrail: preferir la referencia si no hay ganancia clara de Brier."""
    if puntuaciones is None:
        return "porcentaje_referencia"
    resultados = puntuaciones["resultados"]
    br_ref = resultados["porcentaje_referencia"]["brier"]
    nombres = ["logistica_porcentaje", "logistica_mapa", "random_forest"]
    mejor = min(nombres, key=lambda nombre: (resultados[nombre]["brier"], nombre))
    if br_ref - resultados[mejor]["brier"] < UMBRAL_MEJORA_BRIER:
        return "porcentaje_referencia"
    # Evitar elegir un modelo que mejore muy poco Brier pero empeore
    # notablemente la clasificacion binaria sobre los mismos casos.
    if (resultados[mejor]["exactitud_porc"] + 3
            < resultados["porcentaje_referencia"]["exactitud_porc"]):
        return "porcentaje_referencia"
    return mejor


def evaluar_separado(ml: pd.DataFrame) -> tuple[str, dict[str, Any] | None]:
    """Selecciona en datos anteriores y evalua en el ultimo 20% no usado."""
    n = len(ml)
    if n < MIN_SELECCION:
        return "porcentaje_referencia", None
    corte = max(N_INICIAL_VALIDACION + 40, math.floor(n * .8))
    if corte >= n:
        return "porcentaje_referencia", None
    primeros, ultimos = ml.iloc[:corte], ml.iloc[corte:]
    score_eleccion = validacion_progresiva(primeros)
    elegido = seleccionar_modelo(score_eleccion)
    X_tr, X_te = columnas_features(primeros), columnas_features(ultimos)
    y_tr, y_te = primeros["finalizado_int"].astype(int), ultimos["finalizado_int"].to_numpy(dtype=int)
    if y_tr.nunique() != 2:
        return "porcentaje_referencia", None
    if elegido == "porcentaje_referencia":
        p_ia = predecir_prob(None, X_te)
    else:
        modelo = clone(modelos_candidatos()[elegido])
        modelo.fit(X_tr, y_tr)
        p_ia = predecir_prob(modelo, X_te)
    p_ref = predecir_prob(None, X_te)
    p_frec = np.full(len(y_te), (int(y_tr.sum()) + 2) / (len(y_tr) + 4))
    met_ia = resumen_predicciones(y_te, p_ia)
    met_ref = resumen_predicciones(y_te, p_ref)
    met_frec = resumen_predicciones(y_te, p_frec)
    p_siempre = np.ones(len(y_te))
    met_siempre = resumen_predicciones(y_te, p_siempre)
    reporte = {
        "metodo": "holdout cronologico (ultimo 20%) sin usar para elegir el modelo",
        "modelo_seleccionado": elegido,
        "expediciones_entrenamiento": len(primeros),
        "expediciones_prueba": len(ultimos),
        "exactitud_ia_porc": met_ia["exactitud_porc"],
        "exactitud_oficial_porc": met_ref["exactitud_porc"],  # Campo compatible v2, NO necesariamente oficial.
        "exactitud_referencia_porc": met_ref["exactitud_porc"],
        "exactitud_siempre_vive_porc": met_siempre["exactitud_porc"],
        "brier_ia": met_ia["brier"],
        "brier_oficial": met_ref["brier"],  # Campo compatible v2.
        "brier_referencia": met_ref["brier"],
        "brier_frecuencia": met_frec["brier"],
        "ia_supero_porcentaje_oficial": met_ia["brier"] < met_ref["brier"],
        "matriz_confusion_ia": met_ia["matriz_confusion"],
        "evaluacion_interna": score_eleccion,
        "nota": (
            "El porcentaje es el introducido por la web, no necesariamente oficial. "
            "La seleccion basada en los datos pasados y el holdout historico "
            "no garantizan mejores resultados futuros."
        ),
    }
    return elegido, reporte


def construir_info_hamster(hist: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    ultimos: dict[str, int] = {}
    derrotas: dict[str, int] = {}
    partidas: dict[str, int] = {}
    if hist.empty:
        return ultimos, derrotas, partidas
    for _, fila in hist.iterrows():
        nombre = fila["hamster"]
        res = int(fila["finalizado_int"])
        partidas[nombre] = partidas.get(nombre, 0) + 1
        derrotas[nombre] = derrotas.get(nombre, 0) + 1 if res == 0 else 0
        ultimos[nombre] = res
    return ultimos, derrotas, partidas


def entrenar_modelo(df: pd.DataFrame, anterior: Estado | None = None) -> Estado:
    """Reajusta el modelo con cada resultado; no repite la evaluación costosa en cada envío."""
    hist, ml, descartadas = construir_historial(df)
    ultimo, derrotas, partidas = construir_info_hamster(hist)
    nombre = "porcentaje_referencia"
    evaluacion = None
    modelo = None
    seleccion_evaluada_con = 0
    status = "referencia_sin_suficientes_datos"
    if len(ml) >= MIN_ENTRENAMIENTO and ml["finalizado_int"].nunique() == 2:
        puede_reutilizar = (
            anterior is not None
            and anterior.evaluacion is not None
            and anterior.seleccion_evaluada_con > 0
            and 0 <= len(ml) - anterior.seleccion_evaluada_con < REVALIDAR_CADA
        )
        if puede_reutilizar:
            # Se conservan los resultados históricos de evaluación hasta el siguiente corte.
            nombre = anterior.modelo_nombre
            evaluacion = anterior.evaluacion
            seleccion_evaluada_con = anterior.seleccion_evaluada_con
        else:
            nombre, evaluacion = evaluar_separado(ml)
            seleccion_evaluada_con = len(ml)
        if nombre != "porcentaje_referencia":
            modelo = clone(modelos_candidatos()[nombre])
            modelo.fit(columnas_features(ml), ml["finalizado_int"].astype(int))
            status = "entrenado"
        else:
            status = "referencia_preferida_por_validacion" if evaluacion else "referencia_por_pocos_datos"
    elif len(ml) >= MIN_ENTRENAMIENTO:
        status = "referencia_una_sola_clase"
    return Estado(
        modelo=modelo, modelo_nombre=nombre, estado_modelo=status,
        historial=hist, entrenables=len(ml), descartadas=descartadas,
        evaluacion=evaluacion, ultimo_por_hamster=ultimo,
        derrotas_por_hamster=derrotas, envios_por_hamster=partidas,
        seleccion_evaluada_con=seleccion_evaluada_con,
    )


def huella_dataframe(df: pd.DataFrame) -> str:
    if df.empty:
        return "vacio"
    ordenado = df.sort_values(["id"], kind="stable") if "id" in df.columns else df
    valores = ordenado.astype("string").fillna("<NULL>")
    hashes = pd.util.hash_pandas_object(valores, index=False).to_numpy()
    return hashlib.sha256("|".join(valores.columns).encode() + hashes.tobytes()).hexdigest()


_cache_lock = threading.Lock()
_cache_listo = threading.Event()
_cache: dict[str, Any] = {
    "momento": -float("inf"), "estado": None, "actualizando": False,
}


def _recargar_cache() -> None:
    """Consulta Supabase y entrena fuera del hilo que responde las predicciones."""
    try:
        df = consultar_expediciones()
        huella = huella_dataframe(df)
        with _cache_lock:
            anterior: Estado | None = _cache["estado"]
        if anterior is not None and anterior.huella == huella:
            return
        inicio = time.monotonic()
        nuevo = entrenar_modelo(df, anterior=anterior)
        nuevo.huella = huella
        with _cache_lock:
            _cache["estado"] = nuevo
        logger.info(
            "modelo=%s filas=%s entrenamiento_segundos=%.2f",
            nuevo.modelo_nombre, nuevo.entrenables, time.monotonic() - inicio,
        )
    except Exception:
        logger.exception("Error consultando Supabase / entrenando v3")
        with _cache_lock:
            if _cache["estado"] is not None:
                logger.warning("Se conserva el ultimo modelo valido en cache")
    finally:
        with _cache_lock:
            _cache["momento"] = time.monotonic()
            _cache["actualizando"] = False
            _cache_listo.set()


def _iniciar_recarga() -> None:
    with _cache_lock:
        if _cache["actualizando"]:
            return
        _cache["actualizando"] = True
        _cache_listo.clear()
    threading.Thread(target=_recargar_cache, name="rollercoin-reentrenar", daemon=True).start()


def obtener_estado() -> Estado:
    with _cache_lock:
        anterior: Estado | None = _cache["estado"]
        actualizar = (not _cache["actualizando"]
                      and (anterior is None or time.monotonic() - _cache["momento"] >= CACHE_SEGUNDOS))
    if actualizar:
        _iniciar_recarga()
    # Si hay un modelo en memoria, la consulta es inmediata.
    # Una actualización concurrente nunca detiene /simular ni /metricas.
    if anterior is not None:
        return anterior
    # Primera carga de Render: es necesario obtener un modelo real.
    _cache_listo.wait(timeout=60)
    with _cache_lock:
        listo: Estado | None = _cache["estado"]
    if listo is not None:
        return listo
    raise HTTPException(
        status_code=503,
        detail="La IA todavía se está preparando. Vuelve a consultar en unos momentos.",
    )


@app.on_event("startup")
def precargar_ia() -> None:
    # Render inicia el entrenamiento una sola vez al arrancar, antes de recibir tráfico.
    _iniciar_recarga()


@app.get("/")
def inicio() -> dict[str, Any]:
    return {"mensaje": "RollerCoin Expedition AI", "version": VERSION, "documentacion": "/docs"}


@app.get("/metricas")
def obtener_metricas() -> dict[str, Any]:
    estado = obtener_estado()
    h = estado.historial
    if h.empty:
        return {
            "metricas": None, "resumen_hamsters": [], "evaluacion_modelo": estado.evaluacion,
            "estado_modelo": estado.estado_modelo,
            "modelo_seleccionado": estado.modelo_nombre,
            "version_modelo": VERSION, "expediciones_entrenables": estado.entrenables,
            "expediciones_descartadas": estado.descartadas,
        }
    total = len(h)
    vivieron = int(h["finalizado_int"].sum())
    aciertos = 0
    fallos = 0
    if "vive_ia" in h.columns:
        for _, fila in h.iterrows():
            prediccion = interpretar_booleano(fila["vive_ia"])
            if prediccion is None:
                continue
            if prediccion == int(fila["finalizado_int"]):
                aciertos += 1
            else:
                fallos += 1
    n_eval = aciertos + fallos
    resumen = []
    for hamster, grupo in h.groupby("hamster", sort=False):
        victorias = int(grupo["finalizado_int"].sum())
        cuenta = len(grupo)
        ultimo = int(grupo["finalizado_int"].iloc[-1])
        racha = 0
        for res in reversed(grupo["finalizado_int"].tolist()):
            if int(res) != ultimo:
                break
            racha += 1
        resumen.append({
            "hamster": hamster,
            "total": cuenta,
            "victorias": victorias,
            "derrotas": cuenta - victorias,
            "winrate": round(victorias / cuenta * 100, 1),
            "racha_tipo": "VIVIÓ" if ultimo == 1 else "MURIÓ",
            "racha_conteo": racha,
        })
    resumen.sort(key=lambda d: d["total"], reverse=True)
    return {
        "metricas": {
            "total_partidas": total,
            "vivieron_num": vivieron,
            "vivieron_porc": round(vivieron / total * 100, 1),
            "murieron_num": total - vivieron,
            "murieron_porc": round((total - vivieron) / total * 100, 1),
            "ia_aciertos": aciertos,
            "ia_fallos": fallos,
            "ia_precision_porc": round(aciertos / n_eval * 100, 1) if n_eval else 0.,
            "ia_exactitud_porc": round(aciertos / n_eval * 100, 1) if n_eval else None,
            "ia_predicciones_evaluadas": n_eval,
        },
        "resumen_hamsters": resumen,
        "evaluacion_modelo": estado.evaluacion,
        "estado_modelo": estado.estado_modelo,
        "modelo_seleccionado": estado.modelo_nombre,
        "version_modelo": VERSION,
        "expediciones_entrenables": estado.entrenables,
        "expediciones_descartadas": estado.descartadas,
        "nota_historial": "Las rachas son solo de registros guardados, no de todos los envios en RollerCoin.",
        "nota_precision_historica": "Aciertos acumulados pueden incluir versiones anteriores de la IA.",
    }


@app.post("/simular")
def simular_expedicion(datos: SimulacionInput) -> dict[str, Any]:
    estado = obtener_estado()
    X = pd.DataFrame([{
        "porcentaje_norm": datos.porcentaje / 100.0,
        "stats": datos.stats,
        "expedicion": datos.expedicion,
    }])
    prob_ia = round(float(predecir_prob(estado.modelo, X)[0]) * 100, 2)
    referencia = round(datos.porcentaje, 2)
    nombre = datos.hamster
    existe = nombre in estado.ultimo_por_hamster
    return {
        "probabilidad_ia": prob_ia,
        "vive_ia": "SI" if prob_ia >= 50.0 else "NO",
        "probabilidad_oficial": referencia,  # Alias de compatibilidad con frontend v2/v5.
        "vive_oficial": "SI" if referencia >= 50.0 else "NO",
        "probabilidad_referencia": referencia,
        "diferencia_puntos": round(prob_ia - referencia, 2),
        "modelo_seleccionado": estado.modelo_nombre,
        "version_modelo": VERSION,
        "estado_modelo": estado.estado_modelo,
        "fatiga_hamster": estado.envios_por_hamster.get(nombre, 0),
        "fatiga_utilizada": False,
        "nota_fatiga": "Conteo de registros, no mide fatiga real.",
        "racha_actual": estado.derrotas_por_hamster.get(nombre, 0),
        "racha_utilizada": False,
        "resultado_anterior": estado.ultimo_por_hamster.get(nombre) if existe else None,
        "hamster_con_historial": existe,
        "expediciones_entrenables": estado.entrenables,
        "evaluacion_disponible": estado.evaluacion is not None,
        "advertencia": (
            "El porcentaje de referencia lo proporciona el frontend y puede ser una estimacion. "
            "Una probabilidad no es una garantia; rachas y envios omitidos no se usan para predecir."
        ),
    }