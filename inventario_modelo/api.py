from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from io import BytesIO
import io
import os
import traceback

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

modelo_accion_obj = None
modelo_dias_obj = None
mejor_modelo_nombre = ""
datos_actuales = None

ARCHIVO_INVENTARIO = "inventario_actual.xlsx"

# ============================================================
# PERSISTENCIA
# ============================================================

def guardar_datos():
    global datos_actuales
    try:
        if datos_actuales is not None and len(datos_actuales) > 0:
            datos_actuales.to_excel(ARCHIVO_INVENTARIO, index=False, engine='openpyxl')
            print(f"💾 Guardados {len(datos_actuales)} productos")
        else:
            pd.DataFrame(columns=["producto"]).to_excel(ARCHIVO_INVENTARIO, index=False, engine='openpyxl')
            print("💾 Archivo vacío guardado")
    except Exception as e:
        print(f"⚠️ Error al guardar: {e}")

@app.on_event("startup")
async def startup_event():
    global datos_actuales, modelo_accion_obj, modelo_dias_obj
    if os.path.exists(ARCHIVO_INVENTARIO):
        try:
            df = pd.read_excel(ARCHIVO_INVENTARIO, engine='openpyxl')
            if df.empty:
                print("📭 Archivo vacío")
                datos_actuales = None
                return
            # Asegurar columnas mínimas
            cols_min = ["producto","stock_tienda","stock_bodega","ventas_semana",
                       "frecuencia","dias_sin_repos","stock_minimo","dias_para_vencer","se_agota_pronto"]
            for col in cols_min:
                if col not in df.columns:
                    df[col] = 0
            datos_actuales = df.copy()
            print(f"✅ {len(datos_actuales)} productos cargados")
            intentar_entrenar(datos_actuales)
            datos_actuales = predecir_todo(datos_actuales)
            guardar_datos()
        except Exception as e:
            print(f"❌ Error al cargar: {e}")
            datos_actuales = None

# ============================================================
# FAMILIAS (RESPETA LA COLUMNA SI YA EXISTE)
# ============================================================

def agregar_familia(df):
    """Si el DataFrame ya tiene la columna 'familia', solo rellena nulos;
    si no la tiene, la crea a partir del nombre del producto."""
    mapa = {
        "Leche": "Lacteos", "Queso": "Lacteos", "Yogurt": "Lacteos",
        "Arroz": "Granos", "Frijoles": "Granos", "Lentejas": "Granos",
        "Pasta": "Granos", "Azúcar": "Granos", "Pan": "Panaderia",
        "Café": "Bebidas", "Atún": "Enlatados",
        "Jabón": "Aseo", "Shampoo": "Aseo", "Detergente": "Aseo"
    }
    if "familia" not in df.columns:
        # Si no existe la columna, la creamos mapeando
        df["familia"] = df["producto"].map(mapa).fillna("Otros")
    else:
        # Si ya existe, rellenamos los nulos con el mapeo
        df["familia"] = df["familia"].fillna(df["producto"].map(mapa)).fillna("Otros")
    # Asegurar que familia_cod existe
    df["familia_cod"] = df["familia"].astype("category").cat.codes
    return df

# ============================================================
# MODELOS
# ============================================================

class ModeloAccion:
    def __init__(self):
        self.features = ["familia_cod","stock_tienda","stock_bodega","frecuencia","dias_sin_repos","stock_minimo"]
        self.modelos = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "GradientBoosting": GradientBoostingClassifier()
        }
    def entrenar(self, df):
        X = df[self.features]; y = df["se_agota_pronto"]
        resultados = {}
        for nombre, modelo in self.modelos.items():
            scores = cross_val_score(modelo, X, y, cv=min(5, len(df)), scoring="accuracy")
            resultados[nombre] = scores.mean()
        self.mejor_nombre = max(resultados, key=resultados.get)
        self.modelo = self.modelos[self.mejor_nombre]
        self.modelo.fit(X, y)
        return self.mejor_nombre, resultados[self.mejor_nombre]
    def predecir(self, df):
        return self.modelo.predict_proba(df[self.features])

class ModeloDias:
    def __init__(self):
        self.modelo = GradientBoostingRegressor()
        self.features = ["stock_tienda","ventas_semana","frecuencia"]
    def entrenar(self, df):
        df_temp = df.copy()
        df_temp["ventas_dia"] = df_temp["ventas_semana"] / 7
        df_temp["dias_real"] = df_temp["stock_tienda"] / df_temp["ventas_dia"].replace(0,1)
        self.modelo.fit(df_temp[self.features], df_temp["dias_real"])
    def predecir(self, df):
        return self.modelo.predict(df[self.features])

# ============================================================
# OPTIMIZACIÓN
# ============================================================

def optimizar(df):
    df["mover"] = (df["stock_minimo"] - df["stock_tienda"]).clip(0)
    df["comprar"] = (df["stock_minimo"]*2 - df["stock_bodega"]).clip(0)
    df["prioridad"] = (df["riesgo"]*0.5 + (1/(df["dias"]+1))*50 + (1/(df["dias_para_vencer"]+1))*50)
    df["estado"] = df["riesgo"].apply(lambda x: "🔴 Crítico" if x>70 else "🟡 Alerta" if x>40 else "🟢 Sin Riesgo")
    return df.sort_values(by="prioridad", ascending=False).reset_index(drop=True)

# ============================================================
# LIMPIEZA DE DATOS (NO BORRA "familia")
# ============================================================

# Solo columnas generadas por el modelo, NO las del usuario
COLS_DERIVADAS = ["riesgo","dias","mover","comprar","prioridad","estado"]

def limpiar_para_entrenar(df):
    """Elimina solo columnas derivadas del modelo y asegura tipos numéricos en las columnas requeridas."""
    df = df.copy()
    # Eliminar solo columnas generadas por el sistema
    for col in COLS_DERIVADAS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    # Asegurar columnas numéricas requeridas
    required = ["stock_tienda","stock_bodega","ventas_semana","frecuencia",
                "dias_sin_repos","stock_minimo","dias_para_vencer","se_agota_pronto"]
    for col in required:
        if col not in df.columns:
            df[col] = 0
        else:
            # Convertir a numérico, valores no numéricos se vuelven NaN y luego 0
            antes = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).clip(lower=0)
            if col == "se_agota_pronto":
                df[col] = df[col].clip(upper=1)
            if (antes.astype(str).str.contains(r'[^0-9]', na=False).any()):
                print(f"⚠️ Valores no numéricos en '{col}' fueron convertidos a 0")
    # No se elimina "familia" ni "familia_cod", para respetar los datos del usuario
    return df

# ============================================================
# ENTRENAMIENTO
# ============================================================

def intentar_entrenar(df):
    global modelo_accion_obj, modelo_dias_obj, mejor_modelo_nombre
    if len(df) < 5:
        return None, None
    df_limpio = limpiar_para_entrenar(df)
    df_limpio = agregar_familia(df_limpio)  # ahora no borrará la familia original
    if "se_agota_pronto" not in df_limpio.columns or df_limpio["se_agota_pronto"].nunique() < 2:
        print("[WARN] 'se_agota_pronto' debe tener valores 0 y 1")
        return None, None
    try:
        modelo_accion_obj = ModeloAccion()
        mejor, precision = modelo_accion_obj.entrenar(df_limpio)
        mejor_modelo_nombre = mejor
        modelo_dias_obj = ModeloDias()
        modelo_dias_obj.entrenar(df_limpio)
        print(f"[OK] Mejor modelo: {mejor} ({precision:.2%})")
        return mejor, precision
    except Exception as e:
        print(f"[ERROR] Entrenamiento: {e}")
        traceback.print_exc()
        return None, None

# ============================================================
# PREDICCIÓN
# ============================================================

def predecir_todo(df):
    global modelo_accion_obj, modelo_dias_obj
    df = agregar_familia(df)  # respeta familia existente
    if modelo_accion_obj and modelo_dias_obj:
        try:
            probas = modelo_accion_obj.predecir(df)
            df["riesgo"] = probas[:,1]*100
            df["dias"] = modelo_dias_obj.predecir(df).clip(0)
            df = optimizar(df)
        except Exception as e:
            print(f"[WARN] Error en predicción: {e}")
            df = valores_por_defecto(df)
    else:
        df = valores_por_defecto(df)
    return df

def valores_por_defecto(df):
    defaults = {"riesgo":0.0,"dias":0.0,"mover":0.0,"comprar":0.0,"prioridad":0.0,"estado":"🟢 Sin Riesgo","dias_para_vencer":30}
    for k,v in defaults.items():
        if k not in df.columns: df[k] = v
    return df

# ============================================================
# RESPUESTA JSON
# ============================================================

def formatear_respuesta(df):
    if df is None or len(df) == 0:
        return {"resumen":{"criticos":0,"alertas":0,"sin_riesgo":0,"total":0},
                "top_prioridad":[],"todos":[],"lista_familias":[],"resumen_familias":[],
                "graficos":{"estado":{"criticos":0,"alertas":0,"sin_riesgo":0},"top_riesgo":[],"por_familia":{}},
                "vencimiento":{"proximo_7_dias":0,"proximo_15_dias":0,"proximo_30_dias":0,"mas_30_dias":0}}
    criticos = int(len(df[df["riesgo"]>70]))
    alertas = int(len(df[(df["riesgo"]>40)&(df["riesgo"]<=70)]))
    sin_riesgo = int(len(df[df["riesgo"]<=40]))
    cols = ["producto","familia","stock_tienda","stock_bodega","ventas_semana","dias_sin_repos","stock_minimo",
            "se_agota_pronto","dias","riesgo","mover","comprar","dias_para_vencer","prioridad","estado"]
    cols = [c for c in cols if c in df.columns]
    return {
        "resumen":{"criticos":criticos,"alertas":alertas,"sin_riesgo":sin_riesgo,"total":len(df)},
        "top_prioridad":df[cols].head(20).to_dict(orient="records"),
        "todos":df[cols].to_dict(orient="records"),
        "lista_familias":sorted(df["familia"].unique().tolist()) if "familia" in df.columns else [],
        "resumen_familias":df.groupby("familia").agg({"riesgo":"mean","dias":"mean","mover":"sum","comprar":"sum"}).round(1).reset_index().to_dict(orient="records") if "familia" in df.columns else [],
        "graficos":{
            "estado":{"criticos":criticos,"alertas":alertas,"sin_riesgo":sin_riesgo},
            "top_riesgo":df.nlargest(5,"riesgo")[["producto","riesgo"]].to_dict(orient="records"),
            "por_familia":df.groupby("familia")["riesgo"].mean().round(1).to_dict() if "familia" in df.columns else {}
        },
        "vencimiento":{
            "proximo_7_dias":int(len(df[df["dias_para_vencer"]<=7])),
            "proximo_15_dias":int(len(df[(df["dias_para_vencer"]>7)&(df["dias_para_vencer"]<=15)])),
            "proximo_30_dias":int(len(df[(df["dias_para_vencer"]>15)&(df["dias_para_vencer"]<=30)])),
            "mas_30_dias":int(len(df[df["dias_para_vencer"]>30]))
        }
    }

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/entrenar")
async def entrenar(archivo: UploadFile = File(...)):
    try:
        global modelo_accion_obj, modelo_dias_obj, mejor_modelo_nombre, datos_actuales
        contenido = await archivo.read()
        df_nuevo = pd.read_excel(BytesIO(contenido), engine='openpyxl')
        if "producto" not in df_nuevo.columns:
            return {"error": "El archivo debe tener columna 'producto'"}
        # Limpiar nombres
        df_nuevo["producto"] = df_nuevo["producto"].astype(str).str.strip()
        df_nuevo = df_nuevo[df_nuevo["producto"] != ""]
        if df_nuevo.empty:
            return {"error": "No hay productos válidos"}

        # Reemplazar completamente los datos anteriores, pero respetando la familia original
        datos_actuales = limpiar_para_entrenar(df_nuevo)  # no borra familia
        
        # Entrenar con los nuevos datos
        mejor, precision = intentar_entrenar(datos_actuales)
        if mejor is None:
            return {"error": "No se pudo entrenar. Asegúrate de tener al menos 5 filas y la columna 'se_agota_pronto' con valores 0 y 1."}
        
        # Predecir con el modelo entrenado
        datos_actuales = predecir_todo(datos_actuales)
        guardar_datos()
        return {
            "mensaje": f"✅ Modelo {mejor} entrenado ({precision:.2%})",
            "modelo_seleccionado": mejor,
            "precision": f"{precision:.2%}",
            "total_productos": len(datos_actuales),
            "familias": datos_actuales["familia"].nunique(),
            **formatear_respuesta(datos_actuales)
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Error: {str(e)}"}

@app.post("/producto/agregar")
async def agregar_producto(
    producto: str = Form(...), familia: str = Form("Otros"), stock_tienda: int = Form(0),
    stock_bodega: int = Form(0), ventas_semana: int = Form(0), dias_sin_repos: int = Form(0),
    stock_minimo: int = Form(5), dias_para_vencer: int = Form(30), se_agota_pronto: int = Form(0)
):
    global datos_actuales
    producto = producto.strip()
    if not producto: return {"error": "Nombre requerido"}
    if datos_actuales is None:
        datos_actuales = pd.DataFrame(columns=["producto","familia","familia_cod","stock_tienda","stock_bodega",
                                               "ventas_semana","frecuencia","dias_sin_repos","stock_minimo",
                                               "dias_para_vencer","se_agota_pronto"])
    if producto.lower() in datos_actuales["producto"].str.lower().values:
        return {"error": f"'{producto}' ya existe"}
    nuevo = pd.DataFrame([{
        "producto": producto,
        "familia": familia,
        "stock_tienda": max(0, stock_tienda),
        "stock_bodega": max(0, stock_bodega),
        "ventas_semana": max(0, ventas_semana),
        "frecuencia": max(0, ventas_semana),
        "dias_sin_repos": max(0, dias_sin_repos),
        "stock_minimo": max(0, stock_minimo),
        "dias_para_vencer": max(0, dias_para_vencer),
        "se_agota_pronto": se_agota_pronto
    }])
    datos_actuales = pd.concat([datos_actuales, nuevo], ignore_index=True)
    intentar_entrenar(datos_actuales)
    datos_actuales = predecir_todo(datos_actuales)
    guardar_datos()
    return {"mensaje": f"✅ {producto} agregado", "total": len(datos_actuales), **formatear_respuesta(datos_actuales)}

@app.put("/producto/editar/{indice}")
async def editar_producto(
    indice: int, producto: str = Form(...), familia: str = Form("Otros"),
    stock_tienda: int = Form(0), stock_bodega: int = Form(0), ventas_semana: int = Form(0),
    dias_sin_repos: int = Form(0), stock_minimo: int = Form(5), dias_para_vencer: int = Form(30),
    se_agota_pronto: int = Form(0)
):
    global datos_actuales
    if datos_actuales is None: return {"error": "No hay productos"}
    if indice < 0 or indice >= len(datos_actuales): return {"error": "Índice inválido"}
    producto = producto.strip()
    if not producto: return {"error": "Nombre requerido"}
    datos_actuales.at[indice, "producto"] = producto
    datos_actuales.at[indice, "familia"] = familia
    datos_actuales.at[indice, "stock_tienda"] = max(0, stock_tienda)
    datos_actuales.at[indice, "stock_bodega"] = max(0, stock_bodega)
    datos_actuales.at[indice, "ventas_semana"] = max(0, ventas_semana)
    datos_actuales.at[indice, "frecuencia"] = max(0, ventas_semana)
    datos_actuales.at[indice, "dias_sin_repos"] = max(0, dias_sin_repos)
    datos_actuales.at[indice, "stock_minimo"] = max(0, stock_minimo)
    datos_actuales.at[indice, "dias_para_vencer"] = max(0, dias_para_vencer)
    datos_actuales.at[indice, "se_agota_pronto"] = se_agota_pronto
    intentar_entrenar(datos_actuales)
    datos_actuales = predecir_todo(datos_actuales)
    guardar_datos()
    return {"mensaje": f"✅ {producto} actualizado", "total": len(datos_actuales), **formatear_respuesta(datos_actuales)}

@app.delete("/producto/eliminar/{indice}")
async def eliminar_producto(indice: int):
    global datos_actuales
    if datos_actuales is None: return {"error": "No hay productos"}
    if indice < 0 or indice >= len(datos_actuales): return {"error": "Índice inválido"}
    nombre = datos_actuales.at[indice, "producto"]
    datos_actuales = datos_actuales.drop(indice).reset_index(drop=True)
    if len(datos_actuales) == 0:
        datos_actuales = pd.DataFrame(columns=datos_actuales.columns)
    guardar_datos()
    return {"mensaje": f"🗑️ {nombre} eliminado", "total": len(datos_actuales), **formatear_respuesta(datos_actuales)}

@app.post("/producto/eliminar-varios")
async def eliminar_varios(request: Request):
    global datos_actuales
    if datos_actuales is None: return {"error": "No hay productos"}
    form = await request.form()
    indices = [int(i) for i in form.getlist('indices')]
    validos = sorted([i for i in indices if 0 <= i < len(datos_actuales)], reverse=True)
    if not validos: return {"error": "No válidos"}
    nombres = [datos_actuales.at[i, "producto"] for i in validos]
    datos_actuales = datos_actuales.drop(validos).reset_index(drop=True)
    if len(datos_actuales) == 0:
        datos_actuales = pd.DataFrame(columns=datos_actuales.columns)
    guardar_datos()
    return {"mensaje": f"{len(nombres)} eliminados", "total": len(datos_actuales), **formatear_respuesta(datos_actuales)}

@app.get("/productos")
async def obtener_productos():
    global datos_actuales
    if datos_actuales is None or len(datos_actuales) == 0:
        return {"error": "No hay datos", "todos": [], "resumen": {"total": 0}}
    return formatear_respuesta(datos_actuales)

# ============================================================
# DESCARGAS
# ============================================================

@app.post("/descargar-excel")
async def descargar_excel():
    global datos_actuales
    if datos_actuales is None or len(datos_actuales) == 0: return {"error": "No hay datos"}
    cols = ["producto","familia","stock_tienda","stock_bodega","dias","riesgo","mover","comprar","estado","dias_para_vencer"]
    cols = [c for c in cols if c in datos_actuales.columns]
    df_export = datos_actuales[cols].copy()
    if "riesgo" in df_export.columns: df_export["riesgo"] = df_export["riesgo"].round(1)
    if "dias" in df_export.columns: df_export["dias"] = df_export["dias"].round(1)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as w: df_export.to_excel(w, sheet_name='Inventario', index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=inventario.xlsx"})

@app.post("/descargar-csv")
async def descargar_csv():
    if datos_actuales is None or len(datos_actuales) == 0: return {"error": "No hay datos"}
    cols = ["producto","familia","stock_tienda","stock_bodega","dias","riesgo","mover","comprar","estado"]
    cols = [c for c in cols if c in datos_actuales.columns]
    output = io.StringIO()
    datos_actuales[cols].to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=inventario.csv"})

@app.post("/descargar-pdf")
async def descargar_pdf():
    if datos_actuales is None or len(datos_actuales) == 0: return {"error": "No hay datos"}
    try:
        from fpdf import FPDF
        pdf = FPDF(); pdf.add_page()
        pdf.set_font("Arial","B",18); pdf.cell(0,12,"REPORTE DE INVENTARIO",ln=True,align="C")
        pdf.set_font("Arial","",10); pdf.cell(0,8,f"Fecha: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}",ln=True,align="C")
        pdf.ln(8)
        criticos = len(datos_actuales[datos_actuales["riesgo"]>70])
        alertas = len(datos_actuales[(datos_actuales["riesgo"]>40)&(datos_actuales["riesgo"]<=70)])
        sin_riesgo = len(datos_actuales[datos_actuales["riesgo"]<=40])
        pdf.set_font("Arial","B",14); pdf.cell(0,10,"RESUMEN",ln=True)
        pdf.set_font("Arial","",11)
        for txt,val in [("Críticos",criticos),("Alerta",alertas),("Sin Riesgo",sin_riesgo),("Total",len(datos_actuales))]:
            pdf.cell(60,7,f"{txt}: {val}",ln=True)
        pdf.ln(6)
        pdf.set_font("Arial","B",12); pdf.cell(0,10,"PRODUCTOS PRIORITARIOS",ln=True); pdf.ln(2)
        pdf.set_font("Arial","B",8); pdf.set_fill_color(26,35,126); pdf.set_text_color(255,255,255)
        headers = ["Producto","Familia","Tienda","Bodega","Dias","Riesgo","Mover","Comprar"]
        widths = [45,30,18,18,15,22,18,18]
        for h,w in zip(headers,widths): pdf.cell(w,7,h,1,0,"C",True)
        pdf.ln(); pdf.set_text_color(0,0,0)
        for _,row in datos_actuales.head(25).iterrows():
            pdf.set_font("Arial","",8)
            riesgo = round(row.get("riesgo",0),1)
            if riesgo>70: pdf.set_text_color(200,0,0)
            elif riesgo>40: pdf.set_text_color(200,150,0)
            else: pdf.set_text_color(0,150,0)
            vals = [str(row.get("producto",""))[:22], str(row.get("familia",""))[:14],
                    str(int(row.get("stock_tienda",0))), str(int(row.get("stock_bodega",0))),
                    str(int(row.get("dias",0))), f"{riesgo}%",
                    str(int(row.get("mover",0))), str(int(row.get("comprar",0)))]
            for v,w in zip(vals,widths): pdf.cell(w,6,v,1,0,"C")
            pdf.ln()
        pdf.set_text_color(0,0,0)
        buf = BytesIO(); pdf.output(buf); buf.seek(0)
        return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition":"attachment; filename=inventario.pdf"})
    except ImportError:
        return {"error":"fpdf2 no instalado"}
    except Exception as e:
        return await descargar_excel()

@app.post("/descargar")
async def descargar(): return await descargar_excel()