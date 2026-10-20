from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SUPABASE_URL = "https://khyuzutnvoqeqkllnsmy.supabase.co"
SUPABASE_KEY = "sb_publishable_Z9NzjlREhW6RgIqQ3zZ58A_5G6ivYHX"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI(title="RollerCoin Expedition AI API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MAPAS_INFO = {
    '3h':        {'horas': 3,  'dificultad': 2},
    '12h':       {'horas': 12, 'dificultad': 5},
    '18h':       {'horas': 18, 'dificultad': 8},
    '20h':       {'horas': 20, 'dificultad': 6},
    '24h Dark':  {'horas': 24, 'dificultad': 7},
    '24h Beach': {'horas': 24, 'dificultad': 8},
    '30h':       {'horas': 30, 'dificultad': 10},
}

class SimulacionInput(BaseModel):
    hamster: str
    stats: float
    expedicion: str
    porcentaje: float

def preparar_features(df: pd.DataFrame):
    df_proc = df.copy()
    df_proc['horas'] = df_proc['expedicion'].map(lambda x: MAPAS_INFO.get(str(x).strip(), {'horas': 20})['horas'])
    df_proc['dificultad'] = df_proc['expedicion'].map(lambda x: MAPAS_INFO.get(str(x).strip(), {'dificultad': 6})['dificultad'])
    
    df_proc['porcentaje_norm'] = pd.to_numeric(df_proc['porcentaje'], errors='coerce').fillna(0)
    df_proc['porcentaje_norm'] = df_proc['porcentaje_norm'].apply(lambda x: x / 100.0 if x > 1.0 else x)
    df_proc['stats'] = pd.to_numeric(df_proc['stats'], errors='coerce').fillna(0)
    df_proc['ratio_stats_dif'] = df_proc['stats'] / df_proc['dificultad']

    if 'resultado_anterior' not in df_proc: df_proc['resultado_anterior'] = 1
    if 'racha_derrotas' not in df_proc: df_proc['racha_derrotas'] = 0

    # Variable de fatiga eliminada del X_num para que el modelo ya no la tome en cuenta
    X_num = df_proc[['horas', 'dificultad', 'stats', 'porcentaje_norm', 'ratio_stats_dif', 'resultado_anterior', 'racha_derrotas']].fillna(0)
    X_cat = pd.get_dummies(df_proc[['hamster', 'expedicion']], drop_first=False)
    return pd.concat([X_num, X_cat], axis=1)

def entrenar_modelo():
    res = supabase.table("expediciones").select("*").order("fecha", desc=False).order("id", desc=False).execute()
    df = pd.DataFrame(res.data)
    
    if df.empty or len(df.dropna(subset=['finalizado'])) < 5:
        return None, None, df, 1, 0, {}

    df_hist = df.dropna(subset=['finalizado']).copy()
    
    # =========================================================================
    # CORRECCIÓN CRÍTICA: Conversión Absoluta. Si en Supabase dice "true" (texto), 
    # True (booleano), "1" (texto) o 1 (entero), lo sumará como victoria.
    # =========================================================================
    df_hist['finalizado_int'] = df_hist['finalizado'].apply(lambda x: 1 if x is True or str(x).strip().lower() in ['true', '1', '1.0', 'yes'] else 0)
    
    df_hist['fatiga_hamster'] = df_hist.groupby('hamster').cumcount()
    
    resultados = df_hist['finalizado_int'].tolist()
    res_anteriores, rachas_derrotas = [], []
    racha_acumulada, res_previo = 0, 1
    
    for res_actual in resultados:
        res_anteriores.append(res_previo)
        rachas_derrotas.append(racha_acumulada)
        res_previo = res_actual
        racha_acumulada = racha_acumulada + 1 if res_actual == 0 else 0

    df_hist['resultado_anterior'] = res_anteriores
    df_hist['racha_derrotas'] = rachas_derrotas

    fatiga_actual = df.groupby('hamster').size().to_dict()

    X = preparar_features(df_hist)
    y = df_hist['finalizado_int']

    model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=4, min_samples_leaf=2, random_state=42)
    model.fit(X, y)
    
    return model, X.columns, df_hist, res_previo, racha_acumulada, fatiga_actual

@app.get("/metricas")
def obtener_metricas():
    model, _, df_hist, _, _, _ = entrenar_modelo()
    if df_hist.empty:
        return {"metricas": None, "resumen_hamsters": []}

    total_historial = len(df_hist)
    # SUMA USANDO EL FILTRO BLINDADO
    vivieron = int(df_hist['finalizado_int'].sum())
    murieron = total_historial - vivieron

    aciertos = 0
    fallos = 0
    
    if 'vive_ia' in df_hist.columns:
        df_evaluables = df_hist.dropna(subset=['vive_ia', 'finalizado'])
        for _, row in df_evaluables.iterrows():
            # Filtro blindado para evaluar la IA
            real = (row['finalizado'] is True or str(row['finalizado']).strip().lower() in ['true', '1', 'yes'])
            ia_dice_vive = (str(row['vive_ia']).strip().upper() == 'SI')
            if real == ia_dice_vive: aciertos += 1
            else: fallos += 1

    total_eval = aciertos + fallos

    metricas = {
        "total_partidas": total_historial,
        "vivieron_num": vivieron,
        "vivieron_porc": round((vivieron / total_historial) * 100, 1) if total_historial > 0 else 0.0,
        "murieron_num": murieron,
        "murieron_porc": round((murieron / total_historial) * 100, 1) if total_historial > 0 else 0.0,
        "ia_aciertos": aciertos,
        "ia_fallos": fallos,
        "ia_precision_porc": round((aciertos / total_eval) * 100, 1) if total_eval > 0 else 0.0
    }

    resumen = []
    for hamster, grupo in df_hist.groupby('hamster'):
        total_h = len(grupo)
        
        # SUMA DE VICTORIAS USANDO EL FILTRO BLINDADO
        victorias_h = int(grupo['finalizado_int'].sum())
        
        resultados_h = grupo['finalizado_int'].tolist()
        if not resultados_h:
            racha_h, tipo_racha_h = 0, "Ninguna"
        else:
            ultimo_res = resultados_h[-1]
            racha_h = 0
            for res in reversed(resultados_h):
                if res == ultimo_res:
                    racha_h += 1
                else:
                    break
            tipo_racha_h = "VIVIÓ" if ultimo_res == 1 else "MURIÓ"

        resumen.append({
            "hamster": hamster,
            "total": total_h,
            "victorias": victorias_h,
            "derrotas": total_h - victorias_h,
            "winrate": round((victorias_h / total_h) * 100, 1),
            "racha_tipo": tipo_racha_h,
            "racha_conteo": racha_h
        })
    
    resumen = sorted(resumen, key=lambda x: x['total'], reverse=True)

    return {"metricas": metricas, "resumen_hamsters": resumen}

@app.post("/simular")
def simular_expedicion(datos: SimulacionInput):
    model, features_entrenadas, _, ultimo_resultado, racha_actual, fatiga_actual = entrenar_modelo()
    if model is None:
        return {"error": "No hay suficientes datos históricos"}

    fatiga = fatiga_actual.get(datos.hamster, 0)
    df_sim = pd.DataFrame([datos.dict()])
    df_sim['resultado_anterior'] = ultimo_resultado
    df_sim['racha_derrotas'] = racha_actual
    df_sim['fatiga_hamster'] = fatiga
    
    X_sim = preparar_features(df_sim).reindex(columns=features_entrenadas, fill_value=0)
    prob = round(float(model.predict_proba(X_sim)[:, 1][0] * 100), 2)
    vive = "SI" if prob >= 50.0 else "NO"

    return {
        "probabilidad_ia": prob, 
        "vive_ia": vive,
        "fatiga_hamster": int(fatiga),
        "racha_actual": int(racha_actual)
    }