# Metadata del Corpus — TT2026-A081

> **Template formal estándar ESCOM-IPN.** Los campos marcados con
> `[POR LLENAR]` deben completarse antes de la defensa oral. Esta plantilla
> cubre los requisitos de reproducibilidad y ética que típicamente revisan
> los sinodales de IA / BD.

---

## 1. Identificación del corpus

| Campo | Valor |
|-------|-------|
| Nombre del corpus | LSM2_TT2026A081 |
| Versión | 1.0 |
| Fecha de creación | `[POR LLENAR — rango de fechas de captura, p.ej. 2026-03-15 a 2026-04-30]` |
| Responsables (autores TT) | `[POR LLENAR — nombre completo, boleta, correo]` |
| Director de TT | `[POR LLENAR]` |
| Co-director (si aplica) | `[POR LLENAR]` |
| Modalidad de captura | Captura activa con script propio (`sign_data_collector.py`) |
| Idioma de señas | Lengua de Señas Mexicana (LSM) |

## 2. Contenido del corpus

| Campo | Valor |
|-------|-------|
| Número de clases | 18 |
| Etiquetas | `1, 2, 3, 4, 5, 6, 7, 8, 9, a, bien, comer, dias cuantos, hotel, personas cuantas, reservacion, taxi, transporte` |
| Muestras totales | 560 |
| Muestras por clase | 30 (excepto `a` con 50) |
| Formato de cada muestra | `.npy` con shape `(90, 168)` float32 |
| Frames por muestra | 90 (30 fps × 3 segundos) |
| Features por frame | 168 = 42 pose (cintura↑) + 63 mano izq + 63 mano der |
| Tamaño en disco | ~33 MB en `model/data/` |
| Tipo de contenido | Solo landmarks normalizados (NO contiene imagen RGB ni audio) |

## 3. Participantes (señantes)

| Campo | Valor |
|-------|-------|
| Número de señantes | `[POR LLENAR — número entero]` |
| Distribución por género | `[POR LLENAR — p.ej. 1H / 1M]` |
| Rango de edades | `[POR LLENAR]` |
| Nivel de fluidez en LSM | `[POR LLENAR — nativo / intérprete certificado / estudiante / aprendiz]` |
| Mano dominante | `[POR LLENAR — diestro / zurdo / mixto]` |
| Origen geográfico | `[POR LLENAR — relevante por variación dialectal en LSM]` |

> **Nota crítica:** Si el corpus fue capturado por **un solo señante**, declararlo
> explícitamente. Es la limitación principal del estudio y el sinodal lo
> preguntará. La justificación válida es "prueba de concepto de pipeline" y
> debe ir acompañada del plan de ampliación.

## 4. Condiciones técnicas de captura

| Campo | Valor |
|-------|-------|
| Cámara / dispositivo | `[POR LLENAR — marca, modelo, resolución nativa]` |
| Resolución de captura | `[POR LLENAR — p.ej. 1280×720]` |
| Resolución pasada a MediaPipe | 320 × 240 (definido en `config.py:DETECT_W/DETECT_H`) |
| Frame rate objetivo | 30 fps |
| Iluminación | `[POR LLENAR — natural / artificial / mixta; horario]` |
| Fondo | `[POR LLENAR — uniforme / con textura / clutter]` |
| Distancia cámara-señante | `[POR LLENAR — en cm o m]` |
| Ángulo de la cámara | `[POR LLENAR — frontal / lateral / superior; en grados]` |
| Encuadre | Cintura para arriba (requerido por la extracción de pose) |

## 5. Procedimiento de captura

Definido en [model/sign_data_collector.py](model/sign_data_collector.py):

1. El operador selecciona una seña del catálogo `signs.json`.
2. Cuenta regresiva de 2 segundos antes de grabar.
3. Grabación continua de 3 segundos (90 frames a 30 fps).
4. MediaPipe Hands + Pose se ejecutan en cada frame.
5. Se extraen y normalizan 168 features por frame (ver Sección 6).
6. Se guarda un archivo `rep_NNN.npy` con shape `(90, 168)`.
7. Repetir hasta alcanzar el `reps_target` declarado.

## 6. Normalización aplicada

| Bloque | Centro | Escala |
|--------|--------|--------|
| Pose | midpoint de caderas (LM 23 + LM 24)/2 | `‖LM11 − LM12‖` (distancia entre hombros) |
| Mano izquierda | muñeca (LM 0) | `‖LM0 − LM9‖` (muñeca → MCP medio) |
| Mano derecha | idem | idem |

Si MediaPipe no detecta una mano, esa parte del vector se rellena con ceros.
Esto introduce un sesgo conocido frente a oclusión parcial.

## 7. Particiones

| Partición | Fracción | Muestras | Uso |
|-----------|----------|----------|-----|
| Train | 70 % | 392 | Actualización de pesos |
| Val | 15 % | 84 | EarlyStopping + ReduceLROnPlateau |
| Test | 15 % | 84 | Evaluación final (nunca visto durante entrenamiento) |

Estratificación por clase, semilla 42. Índices fijados en
[model/test_indices.npz](model/test_indices.npz) para reproducibilidad bit-a-bit.

## 8. Aspectos éticos y de privacidad

| Campo | Valor |
|-------|-------|
| ¿Se obtuvo consentimiento informado? | `[POR LLENAR — Sí / No]` |
| Formato del consentimiento | `[POR LLENAR — escrito / oral grabado / N/A]` |
| Referencia al documento de consentimiento | `[POR LLENAR — ruta al PDF firmado]` |
| ¿El corpus contiene datos personales identificables? | **No directamente.** Los `.npy` almacenan únicamente coordenadas de landmarks normalizados; no incluyen video RGB, audio ni metadatos identificatorios. |
| ¿Puede reconstruirse la identidad a partir de los landmarks? | Riesgo bajo: la normalización por hombros/muñecas elimina escala absoluta. Sin embargo, patrones idiosincrásicos de movimiento (rúbrica gestual) podrían ser distintivos si se cruzaran con otro corpus. |
| Política de retención | `[POR LLENAR — años; ubicación de respaldo]` |
| Política de redistribución | `[POR LLENAR — privado / IPN-interno / abierto bajo licencia]` |
| Licencia propuesta | `[POR LLENAR — sugerido: CC BY-NC-SA 4.0 para reutilización académica]` |

## 9. Limitaciones declaradas

- **Tamaño reducido del corpus** (~30 muestras / clase) limita la generalización a señantes no vistos.
- **Variabilidad inter-señante no muestreada** (ver Sección 3).
- **Condiciones de captura homogéneas** (un solo entorno típico) → no se evaluó robustez ante cambios de iluminación, fondo o ángulo (ver `brechas_lstm.md` B5).
- **Resolución de detección reducida** (320×240) prioriza latencia sobre precisión de landmarks; aceptable para señas amplias pero subóptimo para configuraciones finas de dedos.

## 10. Versionado

| Versión | Fecha | Cambios |
|---------|-------|---------|
| 1.0     | `[POR LLENAR]` | Versión inicial del corpus para TT2026-A081 |

---

**Para el sinodal:** este template documenta exactamente qué datos respaldan los
resultados reportados en `documentacion_lstm.md`. Cualquier discrepancia entre
los valores `[POR LLENAR]` y la realidad del corpus debe resolverse antes de
la defensa.
