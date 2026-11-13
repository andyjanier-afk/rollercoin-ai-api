import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import main


def muestra(n=175):
    rng = np.random.default_rng(12)
    filas = []
    for i in range(n):
        p = float(rng.uniform(20, 84))
        filas.append({
            "id": i + 1,
            "fecha": f"2026-10-{(i // 10) % 28 + 1:02d}",
            "hamster": ["Cuscuz", "Mamita", "Banjo Walker"][i % 3],
            "expedicion": ["3h", "18h", "24h Dark", "30h"][i % 4],
            "stats": int(rng.integers(120, 275)),
            "porcentaje": p,
            "finalizado": bool(rng.random() < p / 100),
            "vive_ia": "SI" if i % 2 else None
        })
    return pd.DataFrame(filas)


def test_pendientes_y_booleanos():
    assert main.interpretar_booleano(None) is None
    assert main.interpretar_booleano("") is None
    assert main.interpretar_booleano(False) == 0
    assert main.interpretar_booleano("false") == 0
    assert main.interpretar_booleano(True) == 1
    assert main.interpretar_booleano("true") == 1


def test_features_y_escala_porcentaje():
    hist, datos, descartadas = main.construir_historial(muestra(20))
    X = main.columnas_features(datos)
    assert list(X.columns) == ["porcentaje_norm", "stats", "expedicion"]
    assert np.allclose(X.porcentaje_norm.to_numpy(), datos.porcentaje.to_numpy() / 100)
    assert len(hist) == len(datos) == 20 and descartadas == 0


def test_resultados_pendientes_no_se_usan():
    df = muestra(20)
    df["finalizado"] = df.finalizado.astype(object)
    df.loc[0, "finalizado"] = None
    hist, datos, _ = main.construir_historial(df)
    assert len(hist) == len(datos) == 19


def test_seleccion_temporal_y_holdout():
    e = main.entrenar_modelo(muestra())
    assert e.evaluacion is not None
    assert e.evaluacion["expediciones_prueba"] > 0
    assert e.evaluacion["expediciones_entrenamiento"] + e.evaluacion["expediciones_prueba"] == 175
    assert e.modelo_nombre in {"porcentaje_referencia", "logistica_porcentaje", "logistica_mapa", "random_forest"}


def test_historial_pequeno_usa_referencia():
    e = main.entrenar_modelo(muestra(8))
    assert e.modelo is None
    assert e.modelo_nombre == "porcentaje_referencia"


def test_endpoints_frontend_sin_supabase_real(monkeypatch):
    e = main.entrenar_modelo(muestra())
    monkeypatch.setattr(main, "obtener_estado", lambda: e)
    client = TestClient(main.app)
    assert client.get("/").json()["version"] == "3.0.0"
    m = client.get("/metricas")
    assert m.status_code == 200
    assert m.json()["metricas"]["total_partidas"] == 175
    r = client.post("/simular", json={
        "hamster": "Cuscuz", "stats": 255, "expedicion": "30h", "porcentaje": 67.2
    })
    assert r.status_code == 200
    p = r.json()
    assert p["probabilidad_oficial"] == 67.2
    assert 0 <= p["probabilidad_ia"] <= 100
    assert p["vive_ia"] in ("SI", "NO")
    for k in ("version_modelo", "modelo_seleccionado", "racha_actual", "diferencia_puntos"):
        assert k in p


def test_nombre_no_se_usa_para_modificar_probabilidad(monkeypatch):
    e = main.entrenar_modelo(muestra())
    monkeypatch.setattr(main, "obtener_estado", lambda: e)
    c = TestClient(main.app)
    datos = {"hamster": "Cuscuz", "stats": 210, "expedicion": "18h", "porcentaje": 60}
    a = c.post("/simular", json=datos).json()["probabilidad_ia"]
    b = c.post("/simular", json={**datos, "hamster": "Mamita"}).json()["probabilidad_ia"]
    assert a == b


def test_mapa_invalido_rechazado():
    r = TestClient(main.app).post("/simular", json={
        "hamster": "Cuscuz", "stats": 220, "expedicion": "Lost Shrine", "porcentaje": 65
    })
    assert r.status_code == 422


def test_cache_si_historial_no_cambia(monkeypatch):
    df = muestra(8)
    llamadas = []
    def consulta():
        llamadas.append(1)
        return df.copy()
    monkeypatch.setattr(main, "consultar_expediciones", consulta)
    monkeypatch.setattr(main, "CACHE_SEGUNDOS", 0)
    main._cache.update({"momento": -float("inf"), "estado": None, "actualizando": False})
    main._cache_listo.clear()
    e1 = main.obtener_estado()
    assert e1.modelo_nombre == "porcentaje_referencia"
    e2 = main.obtener_estado()
    assert e2 is e1
    assert main._cache_listo.wait(timeout=10)
    assert len(llamadas) == 2
    assert main.obtener_estado() is e1
    assert main._cache_listo.wait(timeout=10)
    main._cache.update({"momento": -float("inf"), "estado": None, "actualizando": False})
    main._cache_listo.clear()


def test_actualizar_sin_repetir_validacion_costosa(monkeypatch):
    anterior = main.entrenar_modelo(muestra(175))
    assert anterior.evaluacion is not None
    assert anterior.seleccion_evaluada_con == 175
    df = muestra(176)
    monkeypatch.setattr(main, "REVALIDAR_CADA", 25)
    def prohibida(_):
        raise AssertionError("La validación completa no debe repetirse con cada resultado")
    monkeypatch.setattr(main, "evaluar_separado", prohibida)
    nuevo = main.entrenar_modelo(df, anterior=anterior)
    assert nuevo.entrenables == 176
    assert nuevo.modelo_nombre == anterior.modelo_nombre
    assert nuevo.seleccion_evaluada_con == 175
    assert nuevo.modelo is not None or anterior.modelo is None


def test_no_bloquea_consultas_mientras_actualiza(monkeypatch):
    import threading
    started = threading.Event()
    resume = threading.Event()
    previous = main.entrenar_modelo(muestra(8))
    previous.huella = "anterior"
    def lenta():
        started.set()
        assert resume.wait(timeout=5)
        return muestra(9)
    monkeypatch.setattr(main, "consultar_expediciones", lenta)
    monkeypatch.setattr(main, "CACHE_SEGUNDOS", 0)
    main._cache.update({"momento": -float("inf"), "estado": previous, "actualizando": False})
    main._cache_listo.set()
    try:
        received = main.obtener_estado()
        assert received is previous
        assert started.wait(timeout=2)
        assert main.obtener_estado() is previous
    finally:
        resume.set()
        assert main._cache_listo.wait(timeout=10)
        main._cache.update({"momento": -float("inf"), "estado": None, "actualizando": False})
        main._cache_listo.clear()
