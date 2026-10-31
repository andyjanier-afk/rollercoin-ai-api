import pandas as pd
import pytest
from pydantic import ValidationError

from main import (
    SimulacionInput, interpretar_booleano, construir_historial, preparar_features,
    entrenar_modelo, probabilidades_victoria,
)


def ejemplo(n=40):
    filas = []
    for i in range(n):
        filas.append({
            "id": i + 1,
            "fecha": f"2026-01-{(i % 28) + 1:02d}",
            "hamster": "A" if i % 2 == 0 else "B",
            "expedicion": "3h" if i % 3 == 0 else "12h",
            "stats": 120 + i,
            "porcentaje": 70 if i % 3 else 30,
            "finalizado": i % 3 != 0,
            "vive_ia": "SI" if i % 2 else "NO",
        })
    return pd.DataFrame(filas)


def test_booleanos_y_pendientes():
    assert interpretar_booleano(True) == 1
    assert interpretar_booleano("true") == 1
    assert interpretar_booleano("1.0") == 1
    assert interpretar_booleano(False) == 0
    assert interpretar_booleano("NO") == 0
    assert interpretar_booleano(None) is None
    assert interpretar_booleano("pendiente") is None


def test_rachas_separadas_por_hamster():
    df = ejemplo(4)
    df["hamster"] = ["A", "B", "A", "B"]
    df["finalizado"] = [False, True, False, False]
    hist, datos, ultimos, rachas, _ = construir_historial(df)
    assert hist["racha_derrotas"].tolist() == [0, 0, 1, 0]
    assert hist["resultado_anterior"].tolist() == [0, 0, 0, 1]
    assert rachas == {"A": 2, "B": 1}
    assert ultimos == {"A": 0, "B": 0}
    assert len(datos) == 4


def test_porcentaje_no_se_confunde_con_fraccion():
    df = pd.DataFrame([{
        "hamster": "A", "expedicion": "3h", "stats": 10,
        "porcentaje": 0.8,
    }])
    X = preparar_features(df)
    assert X.iloc[0]["porcentaje_norm"] == pytest.approx(0.008)


def test_una_clase_no_entrena():
    df = ejemplo(40)
    df["finalizado"] = True
    estado = entrenar_modelo(df)
    assert estado.modelo is None
    assert "victorias y derrotas" in estado.estado_modelo


def test_modelo_entrena_y_evalua():
    estado = entrenar_modelo(ejemplo(40))
    assert estado.modelo is not None
    assert estado.evaluacion is not None
    assert estado.evaluacion["expediciones_prueba"] == 8
    df = pd.DataFrame([{
        "hamster": "A", "expedicion": "3h", "stats": 160,
        "porcentaje": 70,
    }])
    X = preparar_features(df).reindex(columns=estado.columnas, fill_value=0)
    prob = probabilidades_victoria(estado.modelo, X)[0]
    assert 0 <= prob <= 1


def test_input_invalido():
    with pytest.raises(ValidationError):
        SimulacionInput(hamster="A", stats=10, expedicion="inexistente", porcentaje=60)
    with pytest.raises(ValidationError):
        SimulacionInput(hamster="A", stats=10, expedicion="3h", porcentaje=101)


def test_endpoints_con_modelo_en_cache(monkeypatch):
    import time
    from fastapi.testclient import TestClient
    import main

    estado = entrenar_modelo(ejemplo(40))
    monkeypatch.setattr(main, "_cache", {"momento": time.monotonic(), "estado": estado})
    cliente = TestClient(main.app)

    respuesta = cliente.get("/metricas")
    assert respuesta.status_code == 200
    met = respuesta.json()
    assert met["metricas"]["total_partidas"] == 40
    assert met["evaluacion_modelo"]["expediciones_prueba"] == 8
    assert met["metricas"]["ia_predicciones_evaluadas"] == 40

    simulacion = cliente.post("/simular", json={
        "hamster": "A", "stats": 100, "expedicion": "3h", "porcentaje": 77
    })
    assert simulacion.status_code == 200
    datos = simulacion.json()
    assert datos["probabilidad_oficial"] == 77.0
    assert datos["fatiga_utilizada"] is False
    assert datos["hamster_con_historial"] is True
    assert 0 <= datos["probabilidad_ia"] <= 100

    malo = cliente.post("/simular", json={
        "hamster": "A", "stats": 100, "expedicion": "invalido", "porcentaje": 77
    })
    assert malo.status_code == 422
