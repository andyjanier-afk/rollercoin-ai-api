# RollerCoin Expedition AI API v2

API compatible con los endpoints del proyecto original: `POST /simular` y `GET /metricas`.

## Suposiciones importantes

- `finalizado`: `True` significa **sobrevivio**; `False` significa **murio**. `NULL` significa expedicion sin resultado. Si en tu base `finalizado` solamente quiere decir «terminada», el esquema debe corregirse ANTES de entrenar.
- `porcentaje`: probabilidad **previa** a la expedicion, expresada SIEMPRE entre `0` y `100` (por ejemplo, `75` significa 75%). Si la base antigua guardaba `0.75` para representar 75%, primero debes migrar/converter esos registros. **No mezcles escalas.**
- Las dificultades del diccionario `MAPAS_INFO` son heuristicas; no se verifica que coincidan con reglas oficiales.
- El nombre de cada hamster debe mantenerse consistente (`Hamster Uno` no es igual a `hamster uno`).
- La evaluacion reservada solo se muestra con al menos 30 expediciones entrenables y ambas clases en el periodo inicial. Es una comprobacion inicial, no una prueba estadistica concluyente.
- Solo se evalua historicamente `vive_ia` si esa prediccion se guardo ANTES de conocer el resultado. De lo contrario, la metrica no es valida.

## Instalacion (Python 3.10+)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

Copia `.env.example` a `.env` y agrega tu URL, **clave publicable** y origenes reales del frontend. No utilices claves `service_role` o secretas en clientes ni en repositorios. La tabla `expediciones` debe tener al menos: `id`, `fecha`, `hamster`, `expedicion`, `stats`, `porcentaje`, `finalizado`; `vive_ia` es opcional. La clave publicable debe tener permisos de lectura conforme a RLS de Supabase.

```bash
uvicorn main:app --reload
```

Documentacion interactiva: http://127.0.0.1:8000/docs

### Prueba de simulacion

```json
POST /simular
{
  "hamster": "Hamster1",
  "stats": 150,
  "expedicion": "12h",
  "porcentaje": 72
}
```

### Cambios respecto al codigo original

- Rachas y resultado anterior POR hamster.
- Prohibicion de entrenar con una sola clase o datos insuficientes.
- Datos incompletos no se consideran automaticamente derrotas.
- Escala de porcentaje fija 0-100 y validacion de entradas.
- Validacion temporal cronologica y comparacion de Brier con el porcentaje del juego.
- Entrenamiento final de produccion con todos los datos validos.
- Historial completo paginado de Supabase.
- Cache configurable del modelo (60 segundos por defecto). Cambios nuevos en Supabase pueden tardar hasta ese tiempo en reflejarse.
- CORS restringido a los origenes configurados; cambia CORS_ORIGINS segun tu web.
- `fatiga_hamster` devuelve el conteo historico de expediciones por hamster, solo para compatibilidad con el frontend anterior: NO es fatiga real y NO se utiliza en el modelo. Renombra el campo visual en tu interfaz como "Expediciones registradas".

## Pruebas

```bash
python -m pytest -q
```

La API no crea ni modifica registros. Tu frontend/sistema actual debe continuar guardando cada prediccion `vive_ia` **antes** del resultado y despues actualizar `finalizado`. Evita almacenar contraseñas o claves en frontend.

**Seguridad:** CORS no sustituye la autenticacion ni las politicas RLS. Si los datos son privados, implementa autenticacion y reglas de acceso del lado del servidor. No uses esta API de lectura sin protecciones apropiadas en produccion.
