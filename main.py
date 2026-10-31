"""API de prediccion de expediciones de RollerCoin.

El campo 'porcentaje' se interpreta como porcentaje de supervivencia de 0 a 100,
registrado ANTES de conocer el resultado de cada expedicion.
"""

from __future__ import annotations

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rollercoin-ai")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]
CACHE_SEGUNDOS = max(0, int(os.getenv("CACHE_SEGUNDOS", "60")))
PAGINA = 1000
MIN_ENTRENAMIENTO = 10
MIN_EVALUACION = 30

# Valores de dificultad ORIENTATIVOS: no deben presentarse como oficiales.
MAPAS_INFO = {
    "3h": {"horas": 3, "dificultad": 2},
    "12h": {"horas": 12, "dificultad": 5},
    "18h": {"horas": 18, "dificultad": 8},
    "20h": {"horas": 20, "dificultad": 6},
    "24h Dark": {"horas": 24, "dificultad": 7},
    "24h Beach": {"horas": 24, "dificultad": 8},
    "30h": {"horas": 30, "dificultad": 10},
}

COLUMNAS_NUM = [
    "horas", "dificultad", "stats", "porcentaje_norm",
    "ratio_stats_dif", "resultado_anterior", "racha_derrotas",
    "tiene_historial_hamster",
]

app = FastAPI(title="RollerCoin Expedition AI API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,  # Si usas cookies/credenciales, ajusta origenes y esta opcion.
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class SimulacionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    hamster: str = Field(min_length=1, max_length=150)
    stats: float = Field(ge=0)
    expedicion: str
    porcentaje: float = Field(ge=0, le=100)

    @field_validator("hamster", "expedicion")
    @classmethod
    def limpiar_texto(cls, texto: str) -> str:
        texto = texto.strip()
        if not texto:
            raise ValueError("No puede estar vacio")
        return texto

    @field_validator("expedicion")
    @classmethod
    def comprobar_mapa(cls, mapa: str) -> str:
        if mapa not in MAPAS_INFO:
            raise ValueError(f"Expedicion desconocida. Opciones: {', '.join(MAPAS_INFO)}")
        return mapa


@dataclass
class Estado:
    modelo: RandomForestClassifier | None
    columnas: list[str]
    historial: pd.DataFrame
    evaluacion: dict[str, Any] | None
    estado_modelo: str
    total_entrenamiento: int
    filas_descartadas: int
    ultimo_por_hamster: dict[str, int]
    racha_por_hamster: dict[str, int]
    partidas_por_hamster: dict[str, int]


def interpretar_booleano(valor: Any) -> int | None:
    """1/0 si es un resultado conocido; None si falta o no es valido."""
    if valor is None:
        return None
    if isinstance(valor, (bool, np.bool_)):
        return int(valor)
    if isinstance(valor, (int, float, np.integer, np.floating)):
        numero = float(valor)
        if math.isfinite(numero) and numero in (0.0, 1.0):
            return int(numero)
        return None
    texto = str(valor).strip().lower()
    if texto in {"true", "1", "1.0", "yes", "si", "sí", "vive", "vivio", "vivió"}:
        return 1
    if texto in {"false", "0", "0.0", "no", "muere", "murio", "murió"}:
        return 0
    return None


def convertir_numeros(columna: pd.Series) -> pd.Series:
    """Acepta numeros o texto como '75', '75.5', '75,5' y '75%'."""
    limpia = (
        columna.astype("string")
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(limpia, errors="coerce")


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Misma transformacion para entrenar y predecir."""
    datos = df.copy()
    datos["horas"] = datos["expedicion"].map(lambda m: MAPAS_INFO[m]["horas"])
    datos["dificultad"] = datos["expedicion"].map(
        lambda m: MAPAS_INFO[m]["dificultad"]
    )
    # La escala del porcentaje es SIEMPRE 0-100, incluso para 0.8 o 1.
    datos["porcentaje_norm"] = datos["porcentaje"].astype(float) / 100.0
    datos["ratio_stats_dif"] = datos["stats"] / datos["dificultad"]
    for columna, por_defecto in (
        ("resultado_anterior", 0),
        ("racha_derrotas", 0),
        ("tiene_historial_hamster", 0),
    ):
        if columna not in datos:
            datos[columna] = por_defecto

    numericas = datos[COLUMNAS_NUM].astype(float)
    categorias = pd.get_dummies(
        datos[["hamster", "expedicion"]], dtype=int
    )
    return pd.concat([numericas, categorias], axis=1)


def incorporar_historial_por_hamster(
    df_hist: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Calcula variables usando UNICAMENTE partidas anteriores del mismo hamster."""
    df_hist = df_hist.copy()
    previos, rachas, tiene_historial = [], [], []
    ultimo: dict[str, int] = {}
    derrotas: dict[str, int] = {}

    # df_hist llega ordenado de mas antiguo a mas reciente desde Supabase.
    for _, fila in df_hist.iterrows():
        hamster = fila["hamster"]
        resultado = int(fila["finalizado_int"])
        tiene_historial.append(int(hamster in ultimo))
        previos.append(ultimo.get(hamster, 0))
        rachas.append(derrotas.get(hamster, 0))
        ultimo[hamster] = resultado
        derrotas[hamster] = derrotas.get(hamster, 0) + 1 if resultado == 0 else 0

    df_hist["resultado_anterior"] = previos
    df_hist["racha_derrotas"] = rachas
    df_hist["tiene_historial_hamster"] = tiene_historial
    return df_hist, ultimo, derrotas


def consultar_expediciones() -> pd.DataFrame:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Configura SUPABASE_URL y SUPABASE_KEY en .env")

    from supabase import create_client

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    filas: list[dict[str, Any]] = []
    desde = 0
    while True:
        respuesta = (
            supabase.table("expediciones")
            .select("*")
            .order("fecha", desc=False)
            .order("id", desc=False)
            .range(desde, desde + PAGINA - 1)
            .execute()
        )
        lote = respuesta.data or []
        filas.extend(lote)
        if len(lote) < PAGINA:
            break
        desde += PAGINA

    return pd.DataFrame(filas)


def construir_historial(df: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int], int
]:
    """Separa historial finalizado para metricas del subconjunto entrenable."""
    if df.empty:
        vacio = pd.DataFrame()
        return vacio, vacio, {}, {}, 0

    requeridas = {"id", "fecha", "hamster", "expedicion", "stats", "porcentaje", "finalizado"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en expediciones: {sorted(faltantes)}")

    datos = df.copy()
    datos["hamster"] = datos["hamster"].fillna("").astype(str).str.strip()
    datos["expedicion"] = datos["expedicion"].fillna("").astype(str).str.strip()
    datos["finalizado_int"] = datos["finalizado"].map(interpretar_booleano)

    # Los registros pendientes/no interpretables no cuentan como derrotas.
    historial = datos.loc[
        datos["finalizado_int"].notna() & datos["hamster"].ne("")
    ].copy()
    if historial.empty:
        vacio = pd.DataFrame()
        return vacio, vacio, {}, {}, 0

    historial["finalizado_int"] = historial["finalizado_int"].astype(int)
    historial, ultimos, rachas = incorporar_historial_por_hamster(historial)

    entrenables = historial.copy()
    entrenables["stats"] = convertir_numeros(entrenables["stats"])
    entrenables["porcentaje"] = convertir_numeros(entrenables["porcentaje"])
    valido = (
        entrenables["expedicion"].isin(MAPAS_INFO)
        & entrenables["stats"].notna()
        & entrenables["stats"].ge(0)
        & np.isfinite(entrenables["stats"])
        & entrenables["porcentaje"].notna()
        & entrenables["porcentaje"].between(0, 100)
        & np.isfinite(entrenables["porcentaje"])
    )
    entrenables = entrenables.loc[valido].copy()
    descartadas = len(historial) - len(entrenables)
    return historial, entrenables, ultimos, rachas, descartadas


def nueva_ia() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def probabilidades_victoria(modelo: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    clase_victoria = list(modelo.classes_).index(1)
    return modelo.predict_proba(X)[:, clase_victoria]


def evaluar_temporalmente(X: pd.DataFrame, df_ml: pd.DataFrame) -> dict[str, Any] | None:
    """La evaluacion no usa los resultados futuros para ajustar el modelo."""
    cantidad = len(df_ml)
    if cantidad < MIN_EVALUACION:
        return None

    n_prueba = max(8, math.ceil(cantidad * 0.20))
    corte = cantidad - n_prueba
    y = df_ml["finalizado_int"].astype(int)
    X_entreno, X_prueba = X.iloc[:corte], X.iloc[corte:]
    y_entreno, y_prueba = y.iloc[:corte], y.iloc[corte:]

    if y_entreno.nunique() < 2:
        return None

    evaluador = nueva_ia()
    evaluador.fit(X_entreno, y_entreno)
    probs_ia = probabilidades_victoria(evaluador, X_prueba)
    pred_ia = (probs_ia >= 0.5).astype(int)
    probs_oficial = df_ml["porcentaje"].iloc[corte:].to_numpy(dtype=float) / 100.0
    pred_oficial = (probs_oficial >= 0.5).astype(int)
    matriz = confusion_matrix(y_prueba, pred_ia, labels=[0, 1])

    brier_ia = float(brier_score_loss(y_prueba, probs_ia))
    brier_oficial = float(brier_score_loss(y_prueba, probs_oficial))
    return {
        "metodo": "ultimas expediciones reservadas, en orden cronologico",
        "expediciones_entrenamiento": int(corte),
        "expediciones_prueba": int(n_prueba),
        "exactitud_ia_porc": round(float(accuracy_score(y_prueba, pred_ia)) * 100, 2),
        "exactitud_oficial_porc": round(float(accuracy_score(y_prueba, pred_oficial)) * 100, 2),
        "brier_ia": round(brier_ia, 5),
        "brier_oficial": round(brier_oficial, 5),
        "ia_supero_porcentaje_oficial": brier_ia < brier_oficial,
        "matriz_confusion_ia": {
            "verdaderos_negativos": int(matriz[0, 0]),
            "falsos_positivos": int(matriz[0, 1]),
            "falsos_negativos": int(matriz[1, 0]),
            "verdaderos_positivos": int(matriz[1, 1]),
        },
        "nota": "Brier menor es mejor. Una sola particion temporal no prueba superioridad estadistica.",
    }


def entrenar_modelo(df: pd.DataFrame) -> Estado:
    historial, df_ml, ultimo, rachas, descartadas = construir_historial(df)
    cantidad = len(df_ml)
    partidas = historial.groupby("hamster").size().to_dict() if not historial.empty else {}
    base = dict(
        modelo=None,
        columnas=[],
        historial=historial,
        evaluacion=None,
        total_entrenamiento=cantidad,
        filas_descartadas=descartadas,
        ultimo_por_hamster=ultimo,
        racha_por_hamster=rachas,
        partidas_por_hamster=partidas,
    )
    if cantidad < MIN_ENTRENAMIENTO:
        return Estado(
            **base,
            estado_modelo=f"Se necesitan al menos {MIN_ENTRENAMIENTO} expediciones validas para entrenar",
        )
    if df_ml["finalizado_int"].nunique() < 2:
        return Estado(**base, estado_modelo="Se necesitan victorias y derrotas para entrenar")

    X = preparar_features(df_ml)
    evaluacion = evaluar_temporalmente(X, df_ml)

    # Despues de medir, entrenamos el modelo de produccion con TODO el historial.
    modelo = nueva_ia()
    modelo.fit(X, df_ml["finalizado_int"].astype(int))
    return Estado(
        modelo=modelo,
        columnas=list(X.columns),
        historial=historial,
        evaluacion=evaluacion,
        estado_modelo="entrenado",
        total_entrenamiento=cantidad,
        filas_descartadas=descartadas,
        ultimo_por_hamster=ultimo,
        racha_por_hamster=rachas,
        partidas_por_hamster=partidas,
    )


_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"momento": -float("inf"), "estado": None}


def obtener_estado() -> Estado:
    """Reutiliza el modelo; se vuelve a consultar y entrenar al vencer la cache."""
    with _cache_lock:
        ahora = time.monotonic()
        if _cache["estado"] is not None and ahora - _cache["momento"] < CACHE_SEGUNDOS:
            return _cache["estado"]
        try:
            df = consultar_expediciones()
            estado = entrenar_modelo(df)
        except Exception as exc:
            logger.exception("No fue posible leer el historial o entrenar")
            raise HTTPException(
                status_code=503,
                detail="No se pudo consultar o procesar Supabase. Revisa el servidor y el esquema.",
            ) from exc
        _cache.update({"momento": time.monotonic(), "estado": estado})
        return estado


@app.get("/")
def inicio():
    return {"mensaje": "RollerCoin Expedition AI", "version": "2.0.0", "documentacion": "/docs"}


@app.get("/metricas")
def obtener_metricas():
    estado = obtener_estado()
    df_hist = estado.historial
    if df_hist.empty:
        return {
            "metricas": None,
            "resumen_hamsters": [],
            "evaluacion_modelo": None,
            "estado_modelo": estado.estado_modelo,
            "expediciones_entrenables": estado.total_entrenamiento,
            "expediciones_descartadas": estado.filas_descartadas,
        }

    total_historial = len(df_hist)
    vivieron = int(df_hist["finalizado_int"].sum())
    murieron = total_historial - vivieron
    aciertos, fallos = 0, 0

    if "vive_ia" in df_hist.columns:
        for _, fila in df_hist.iterrows():
            estimacion = interpretar_booleano(fila["vive_ia"])
            if estimacion is None:
                continue
            if estimacion == int(fila["finalizado_int"]):
                aciertos += 1
            else:
                fallos += 1

    total_eval = aciertos + fallos
    resumen = []
    for hamster, grupo in df_hist.groupby("hamster", sort=False):
        total = len(grupo)
        victorias = int(grupo["finalizado_int"].sum())
        resultados = grupo["finalizado_int"].tolist()
        ultimo = resultados[-1]
        racha = 0
        for resultado in reversed(resultados):
            if resultado != ultimo:
                break
            racha += 1
        resumen.append({
            "hamster": hamster,
            "total": total,
            "victorias": victorias,
            "derrotas": total - victorias,
            "winrate": round(victorias / total * 100, 1),
            "racha_tipo": "VIVIÓ" if ultimo == 1 else "MURIÓ",
            "racha_conteo": racha,
        })
    resumen.sort(key=lambda fila: fila["total"], reverse=True)

    return {
        "metricas": {
            "total_partidas": total_historial,
            "vivieron_num": vivieron,
            "vivieron_porc": round(100 * vivieron / total_historial, 1),
            "murieron_num": murieron,
            "murieron_porc": round(100 * murieron / total_historial, 1),
            "ia_aciertos": aciertos,
            "ia_fallos": fallos,
            # Campo original para mantener compatibilidad con tu frontend.
            # En rigor esta medida es accuracy, no precision de clase positiva.
            "ia_precision_porc": round(100 * aciertos / total_eval, 1) if total_eval else 0.0,
            "ia_exactitud_porc": round(100 * aciertos / total_eval, 1) if total_eval else None,
            "ia_predicciones_evaluadas": total_eval,
        },
        "resumen_hamsters": resumen,
        "evaluacion_modelo": estado.evaluacion,
        "estado_modelo": estado.estado_modelo,
        "expediciones_entrenables": estado.total_entrenamiento,
        "expediciones_descartadas": estado.filas_descartadas,
    }


@app.post("/simular")
def simular_expedicion(datos: SimulacionInput):
    estado = obtener_estado()
    if estado.modelo is None:
        # Se conserva el campo 'error' usado por el frontend anterior.
        return {"error": estado.estado_modelo}

    hamster = datos.hamster
    existe = hamster in estado.ultimo_por_hamster
    df_sim = pd.DataFrame([{
        **datos.model_dump(),
        "resultado_anterior": estado.ultimo_por_hamster.get(hamster, 0),
        "racha_derrotas": estado.racha_por_hamster.get(hamster, 0),
        "tiene_historial_hamster": int(existe),
    }])
    X = preparar_features(df_sim).reindex(columns=estado.columnas, fill_value=0)
    prob_ia = float(probabilidades_victoria(estado.modelo, X)[0]) * 100.0
    prob_oficial = datos.porcentaje

    return {
        "probabilidad_ia": round(prob_ia, 2),
        "vive_ia": "SI" if prob_ia >= 50 else "NO",
        "probabilidad_oficial": round(prob_oficial, 2),
        "vive_oficial": "SI" if prob_oficial >= 50 else "NO",
        "diferencia_puntos": round(prob_ia - prob_oficial, 2),
        "fatiga_hamster": estado.partidas_por_hamster.get(hamster, 0),
        "fatiga_utilizada": False,
        "nota_fatiga": "Este campo es un conteo de expediciones, NO mide fatiga.",
        "racha_actual": estado.racha_por_hamster.get(hamster, 0),
        "resultado_anterior": estado.ultimo_por_hamster.get(hamster) if existe else None,
        "hamster_con_historial": existe,
        "expediciones_entrenables": estado.total_entrenamiento,
        "evaluacion_disponible": estado.evaluacion is not None,
        "advertencia": (
            "Estimacion estadistica; no garantiza el resultado. "
            "Si el porcentaje ingresado es oficial, comparalo con la IA."
        ),
    }
