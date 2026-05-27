# SmartInventory — Sistema de Gestión de Inventario con IA

> Plataforma web que aplica Machine Learning para predecir el riesgo de agotamiento de productos, recomendar acciones de reposición y reducir pérdidas económicas en pequeños y medianos negocios.

---

## 1. Problema o Necesidad

El manejo de inventarios en PyMEs presenta desafíos críticos:

- **Falta de predicción:** no se sabe cuándo un producto se agotará antes de que ocurra.
- **Sobre-stock o desabastecimiento:** exceso de productos que vencen, o ausencia de artículos clave en el momento de venta.
- **Decisiones manuales:** dependencia de la experiencia del encargado sin datos objetivos.
- **Pérdidas económicas:** productos vencidos sin rotación y ventas perdidas por falta de stock.

**Solución con IA:** el sistema predice la probabilidad de agotamiento de cada producto y recomienda acciones específicas (mover de bodega, comprar al proveedor) basándose en datos históricos y patrones de ventas.

---

## 2. Librerías, Frameworks y Recursos Utilizados

### Backend (Python)

| Librería | Uso |
|---|---|
| `FastAPI` | Framework web para crear la API REST |
| `Uvicorn` | Servidor ASGI para ejecutar FastAPI |
| `Pandas` | Manipulación y análisis de datos tabulares |
| `NumPy` | Operaciones numéricas de bajo nivel |
| `Scikit-learn` | Modelos de Machine Learning (clasificación y regresión) |
| `OpenPyXL` | Lectura y escritura de archivos Excel (.xlsx) |
| `FPDF2` | Generación de reportes en formato PDF |

### Frontend (HTML / CSS / JavaScript)

| Recurso | Uso |
|---|---|
| HTML5 | Estructura semántica de la interfaz |
| CSS3 | Diseño responsive, animaciones y variables de color |
| JavaScript Vanilla | Lógica del frontend sin dependencias externas |
| Canvas API | Gráfico de pastel dinámico sin librerías externas |
| Fetch API | Comunicación asíncrona con el backend (JSON / FormData) |

---

## 3. Cómo se Construyó el Dataset

### Columnas requeridas

```
producto · stock_tienda · stock_bodega · ventas_semana · frecuencia
dias_sin_repos · stock_minimo · dias_para_vencer · se_agota_pronto
```

### Proceso de construcción

- **Datos de ejemplo:** generados con `numpy.random` simulando 14 tipos de productos.
- **Variables controladas:** stock entre 0–30 (tienda) y 0–50 (bodega); ventas semanales entre 5 y 25 unidades.
- **Etiqueta objetivo (`se_agota_pronto`):** calculada como `1` si `stock_tienda / (ventas_semana / 7) ≤ 7 días`, `0` en caso contrario.
- **Datos reales:** el usuario puede cargar su propio Excel con el mismo formato de columnas.
- **Persistencia:** los datos se guardan en `inventario_actual.xlsx` entre sesiones.

### Feature Engineering

| Feature | Descripción |
|---|---|
| `familia_cod` | Codificación numérica de la categoría del producto (Lácteos=0, Granos=1, etc.) |
| `ventas_dia` | Derivada de `ventas_semana / 7` para calcular consumo diario |
| `dias_real` | `stock_tienda / ventas_dia` — variable objetivo del regresor |

---

## 4. Cantidad de Entradas para Entrenar

| Escenario | Registros |
|---|---|
| Mínimo requerido | 10 productos con al menos 2 clases en `se_agota_pronto` |
| Dataset de ejemplo incluido | ~520 registros (40 por tipo × 13 tipos) |
| Dataset del usuario (recomendado) | 500+ productos para mayor precisión |
| Validación cruzada | 5-fold cross-validation (`cv=5`) |

---

## 5. Modelos de Machine Learning Utilizados

### Clasificación — Riesgo de agotamiento

| Modelo | Tipo | Hiperparámetros |
|---|---|---|
| `LogisticRegression` | Regresión logística | `max_iter=1000` |
| `RandomForestClassifier` | Ensemble de árboles | Configuración por defecto |
| `GradientBoostingClassifier` | Boosting secuencial | Configuración por defecto |

### Regresión — Días hasta agotarse

| Modelo | Tipo | Objetivo |
|---|---|---|
| `GradientBoostingRegressor` | Boosting para regresión | Predecir los días restantes de stock en tienda |

---

## 6. Por Qué Se Eligieron Esos Modelos

**LogisticRegression**
- Interpretabilidad: coeficientes claros para entender el peso de cada variable.
- Eficiencia: rápido de entrenar, ideal para datos pequeños y medianos.
- Línea base confiable para comparar con modelos más complejos.

**RandomForestClassifier**
- Robustez ante relaciones no lineales y datos con valores atípicos.
- Feature importance integrada para identificar las variables más relevantes.
- Bajo riesgo de overfitting al combinar múltiples árboles independientes.

**GradientBoostingClassifier**
- Alta precisión: corrige errores del árbol anterior de forma secuencial.
- Flexible para capturar patrones complejos en datos tabulares.
- Estado del arte para problemas de clasificación estructurada.

**GradientBoostingRegressor**
- Predicción continua: estima valores numéricos (días) con alta fidelidad.
- Mismo paradigma que el clasificador, facilitando el mantenimiento.

---

## 7. Métricas de Efectividad

Métrica principal: **accuracy** mediante validación cruzada de 5 folds.

| Modelo | Accuracy típica | Observaciones |
|---|---|---|
| LogisticRegression | 75 – 82 % | Buen rendimiento base, interpretable |
| RandomForestClassifier | 80 – 88 % | Mejor que regresión logística |
| **GradientBoostingClassifier ★** | **85 – 92 %** | **Ganador habitual; seleccionado automáticamente** |

> El sistema selecciona el modelo con mayor accuracy y reporta su nombre y precisión al usuario tras el entrenamiento.

---

## 8. Predicciones Generadas por el Sistema

Para cada producto, el sistema calcula:

| Campo | Tipo | Descripción |
|---|---|---|
| `riesgo` | `float` (0–100 %) | Probabilidad de agotarse en los próximos 7 días |
| `dias` | `int` | Días estimados hasta agotar el stock de tienda |
| `mover` | `int` | Unidades a trasladar de bodega a tienda |
| `comprar` | `int` | Unidades sugeridas de compra al proveedor |
| `prioridad` | `float` | Índice combinado de urgencia |
| `estado` | `str` | 🔴 Crítico / 🟡 Alerta / 🟢 Sin Riesgo |

---

## 9. Uso de Predicciones para Construir la Solución

### Fórmula de prioridad

```python
prioridad = riesgo * 0.5 + (1 / (dias + 1)) * 50 + (1 / (dias_para_vencer + 1)) * 50
```

### Acciones automáticas

```python
mover   = max(0, stock_minimo - stock_tienda)     # lo que falta en tienda
comprar = max(0, stock_minimo * 2 - stock_bodega) # reposición sugerida
```

### Reglas de negocio

| Condición | Estado | Acción |
|---|---|---|
| `riesgo > 70 %` | 🔴 Crítico | Acción inmediata requerida |
| `40 % < riesgo ≤ 70 %` | 🟡 Alerta | Monitorear esta semana |
| `riesgo ≤ 40 %` | 🟢 Sin Riesgo | Sin acción necesaria |

---

## 10. Despliegue en la Web

### Arquitectura

```
┌──────────────────────┐     HTTP/REST      ┌────────────────────────┐
│  Frontend            │ ◄────────────────► │  Backend               │
│  HTML / CSS / JS     │   JSON / FormData  │  FastAPI · Python 3.9+ │
│  localhost:8000      │                    │  api.py · Uvicorn       │
└──────────────────────┘                    └────────────────────────┘
```

### Endpoints principales

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/` | Sirve la interfaz web (index.html) |
| `POST` | `/entrenar` | Carga Excel y entrena los modelos |
| `POST` | `/producto/agregar` | Agrega un nuevo producto |
| `PUT` | `/producto/editar/{i}` | Edita un producto existente |
| `DELETE` | `/producto/eliminar/{i}` | Elimina un producto por índice |
| `POST` | `/producto/eliminar-varios` | Elimina múltiples productos en lote |
| `GET` | `/productos` | Retorna todos los productos con predicciones |
| `POST` | `/descargar-excel` | Descarga reporte Excel |
| `POST` | `/descargar-pdf` | Descarga reporte PDF |
| `POST` | `/descargar-csv` | Descarga reporte CSV |

### Ejecución local

```bash
# 1. Instalar dependencias
pip install fastapi uvicorn pandas numpy scikit-learn openpyxl fpdf2

# 2. Iniciar el servidor
uvicorn api:app --reload

# 3. Abrir en el navegador
# http://localhost:8000
```

---

## 11. Frontend y Backend — Explicación General

### Backend (`api.py`)

- **FastAPI** maneja peticiones HTTP asíncronas con validación automática de tipos.
- **Modelos ML** entrenados con scikit-learn, almacenados en memoria RAM durante la sesión.
- **Persistencia** del inventario en `inventario_actual.xlsx`; se carga automáticamente al reiniciar.
- **CORS** configurado para aceptar peticiones desde cualquier origen.
- **Validación** con `FormData` y tipos estrictos (`int`, `str`) en cada endpoint.

### Frontend (`index.html`)

- **SPA (Single Page Application):** 6 secciones con navegación por pestañas sin recarga de página.
- **Diseño responsive:** adaptable a dispositivos móviles y escritorio con CSS Grid y Flexbox.
- **Comunicación:** Fetch API para llamadas asíncronas con manejo de errores.
- **Gráficos:** Canvas API para el gráfico de pastel dinámico, sin librerías externas.
- **CRUD completo:** crear, leer, actualizar y eliminar productos desde la interfaz.
- **Filtros avanzados:** por familia, estado de riesgo, días para vencer y búsqueda textual.

---

## 12. Predicciones → Nuevas Reglas y Comportamientos

### Priorización automática

- Los productos se ordenan por el índice `prioridad` (riesgo + urgencia temporal + vencimiento próximo).
- El Dashboard muestra únicamente el **Top 20** más urgente para focalizar la atención.

### Alertas visuales dinámicas

- **Barra de progreso tricolor:** proporción de productos en cada estado (verde / naranja / rojo).
- **KPIs codificados por color:** contadores de críticos, alertas y sin riesgo en tiempo real.
- **Badges por fila:** cada producto muestra su nivel de riesgo con color asociado.

### Recomendaciones accionables

```
🔴 Crítico:    "Mover X unidades de bodega | Comprar Y unidades al proveedor"
🟡 Alerta:     "Revisar en Z días"
🟢 Sin Riesgo: "✅ OK — sin acción necesaria"
```

### Filtros inteligentes

- Filtrar por `dias_para_vencer` para priorizar productos próximos a caducar.
- Filtrar por `estado` para enfocarse exclusivamente en urgencias.
- Búsqueda por nombre para localizar un producto específico de forma inmediata.

---

## 13. Interfaz Final — Cómo Funciona y Cuál Es Su Objetivo

### Estructura de las 6 secciones

| Sección | Objetivo | Funcionalidades |
|---|---|---|
| 🏠 **Inicio** | Presentación | Explicación del sistema y accesos rápidos |
| 📤 **Cargar** | Entrada de datos | Subir Excel; entrena los 3 modelos automáticamente |
| 📊 **Dashboard** | Vista general | KPIs, Top 20 prioritarios, barra de estado, resumen por familias |
| 🔍 **Filtrar** | Análisis detallado | Filtros combinados por familia, estado, vencimiento y texto |
| ✏️ **Gestionar** | CRUD completo | Agregar, editar y eliminar productos individual o en lote |
| 📈 **Gráficos** | Visualización | Pastel de estado, barras de riesgo, riesgo por familia, vencimientos |

### Flujo de uso típico

1. **Cargar datos** — subir el archivo Excel o agregar productos manualmente.
2. **Revisar el Dashboard** — identificar productos críticos de inmediato.
3. **Filtrar** — análisis específico por familia, estado o vencimiento.
4. **Gestionar** — agregar nuevos productos o actualizar/eliminar los existentes.
5. **Descargar reporte** — exportar en Excel, PDF o CSV para compartir.

### Objetivo final

> Permitir que **cualquier persona, sin conocimientos técnicos**, gestione su inventario de forma inteligente, tome decisiones basadas en datos y **reduzca pérdidas por desabastecimiento o caducidad**.

---

## Estructura del Proyecto

```
smartinventory/
├── api.py                   # Backend FastAPI
├── index.html               # Frontend SPA
├── inventario_actual.xlsx   # Persistencia del inventario (autogenerado)
└── README.md
```

