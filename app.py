import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import glob
import os
import sys
import hashlib
from io import BytesIO
import time

# =============================================================================
# CONFIGURACIÓN GLOBAL (de tu código original)
# =============================================================================

# En lugar de None, poner un límite muy alto pero razonable
Image.MAX_IMAGE_PIXELS = 1000000000  # 1 billón de píxeles

# Configuración fija (de tu código)
max_pixels = 2000000
resolucion_x = 600  # Fijo
densidad_tinta = 1.05  # g/ml

# =============================================================================
# SISTEMA DE AUTENTICACIÓN
# =============================================================================

def inicializar_sesion():
    """Inicializar variables de sesión"""
    if 'autenticado' not in st.session_state:
        st.session_state.autenticado = False
    if 'usuario_actual' not in st.session_state:
        st.session_state.usuario_actual = None
    if 'tipo_usuario' not in st.session_state:
        st.session_state.tipo_usuario = None
    if 'ultimos_resultados' not in st.session_state:
        st.session_state.ultimos_resultados = None

def verificar_usuarios_configurados():
    """Verificar que hay usuarios en secrets"""
    try:
        usuarios_permitidos = st.secrets.get("usuarios", {})
        return bool(usuarios_permitidos)
    except:
        return False

def mostrar_login():
    """Interfaz de login"""
    st.set_page_config(
        page_title="Acceso Simulador Tinta",
        page_icon="🔐",
        layout="centered"
    )
    
    st.title("🔐 Simulador de Consumo de Tinta")
    st.markdown("---")
    
    # Verificar si hay usuarios configurados
    if not verificar_usuarios_configurados():
        st.error("⚠️ Sistema no configurado. Contacta al administrador.")
        return
    
    usuarios_permitidos = st.secrets.get("usuarios", {})
    
    # Separar usuarios por rol
    usuarios_normales = [user for user, data in usuarios_permitidos.items() 
                        if data.get("rol") == "usuario"]
    usuarios_tecnicos = [user for user, data in usuarios_permitidos.items() 
                        if data.get("rol") == "tecnico"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Acceso Usuario")
        if usuarios_normales:
            usuario_user = st.selectbox(
                "Selecciona usuario:",
                options=usuarios_normales,
                key="user_select"
            )
            
            if usuario_user:
                password_user = st.text_input("Contraseña:", type="password", key="user_pass")
                if st.button("🔓 Acceder como Usuario", key="user_btn", use_container_width=True):
                    usuario_data = usuarios_permitidos.get(usuario_user, {})
                    if usuario_data.get("password") == password_user:
                        st.session_state.autenticado = True
                        st.session_state.usuario_actual = usuario_user
                        st.session_state.tipo_usuario = "usuario"
                        st.rerun()
                    else:
                        st.error("❌ Contraseña incorrecta")
        else:
            st.info("No hay usuarios normales configurados")
    
    with col2:
        st.subheader("🛠️ Acceso Técnico")
        if usuarios_tecnicos:
            tecnico_user = st.selectbox(
                "Selecciona técnico:",
                options=usuarios_tecnicos,
                key="tec_select"
            )
            
            if tecnico_user:
                password_tec = st.text_input("Contraseña:", type="password", key="tec_pass")
                if st.button("🔧 Acceder como Técnico", key="tec_btn", use_container_width=True):
                    usuario_data = usuarios_permitidos.get(tecnico_user, {})
                    if usuario_data.get("password") == password_tec:
                        st.session_state.autenticado = True
                        st.session_state.usuario_actual = tecnico_user
                        st.session_state.tipo_usuario = "tecnico"
                        st.rerun()
                    else:
                        st.error("❌ Contraseña incorrecta")
        else:
            st.info("No hay usuarios técnicos configurados")
    
    st.markdown("---")
    st.info("💡 **Nota:** Esta aplicación es de acceso restringido. Contacta al administrador para obtener credenciales.")

# =============================================================================
# CLASE PRINCIPAL - CÓDIGO ORIGINAL ADAPTADO
# =============================================================================

class CMYKRGConverterSimple:
    def __init__(self):
        self.modelo_600 = None
        self.modelo_1200 = None
        self.scaler_600 = None
        self.scaler_1200 = None
        self.modelo_actual = None
        self.scaler_actual = None
        
        # Obtener la carpeta donde está este script
        if getattr(sys, 'frozen', False):
            self.script_dir = os.path.dirname(sys.executable)
        else:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        st.info(f"📂 Carpeta del script: {self.script_dir}")
        st.info(f"🔐 Modo: {st.session_state.tipo_usuario.upper()}")
        
        # Cargar modelos al iniciar
        self.cargar_modelos()
    
    def cargar_modelos(self):
        """Cargar modelos específicos para 600 y 1200 DPI - CÓDIGO ORIGINAL"""
        try:
            st.info("🔍 BUSCANDO MODELOS ESPECÍFICOS POR RESOLUCIÓN...")
            
            # Buscar todos los archivos .pkl en la carpeta del script
            pkl_files = glob.glob(os.path.join(self.script_dir, "*.pkl"))
            
            if not pkl_files:
                st.error("❌ NO se encontraron archivos .pkl en la carpeta del script")
                return
            
            st.success(f"🎯 Archivos .pkl encontrados: {[os.path.basename(f) for f in pkl_files]}")
            
            # Buscar modelos específicos
            modelo_600_path = None
            modelo_1200_path = None
            
            for file_path in pkl_files:
                filename = os.path.basename(file_path).lower()
                
                # Buscar modelo 600 DPI
                if '600' in filename and '1200' not in filename:
                    modelo_600_path = file_path
                    st.success(f"✅ Encontrado modelo 600 DPI: {os.path.basename(file_path)}")
                
                # Buscar modelo 1200 DPI  
                elif '1200' in filename and '600' not in filename:
                    modelo_1200_path = file_path
                    st.success(f"✅ Encontrado modelo 1200 DPI: {os.path.basename(file_path)}")
                
                # Si el nombre es genérico, intentar deducir por el nombre
                elif 'modelo' in filename:
                    if '600' in filename:
                        modelo_600_path = file_path
                        st.success(f"✅ Asignado como modelo 600 DPI: {os.path.basename(file_path)}")
                    elif '1200' in filename:
                        modelo_1200_path = file_path
                        st.success(f"✅ Asignado como modelo 1200 DPI: {os.path.basename(file_path)}")
            
            # Cargar modelo 600 DPI
            if modelo_600_path:
                try:
                    model_data = joblib.load(modelo_600_path)
                    self.modelo_600 = model_data['model']
                    self.scaler_600 = model_data['scaler']
                    st.success(f"✅ Modelo 600 DPI cargado exitosamente")
                except Exception as e:
                    st.error(f"❌ Error cargando modelo 600 DPI: {e}")
            else:
                st.warning("❌ Modelo 600 DPI no encontrado")
            
            # Cargar modelo 1200 DPI
            if modelo_1200_path:
                try:
                    model_data = joblib.load(modelo_1200_path)
                    self.modelo_1200 = model_data['model']
                    self.scaler_1200 = model_data['scaler']
                    st.success(f"✅ Modelo 1200 DPI cargado exitosamente")
                except Exception as e:
                    st.error(f"❌ Error cargando modelo 1200 DPI: {e}")
            else:
                st.warning("❌ Modelo 1200 DPI no encontrado")
            
            # Si solo hay un modelo y no se pudo determinar, cargarlo como universal
            if len(pkl_files) == 1 and (not self.modelo_600 or not self.modelo_1200):
                universal_model = pkl_files[0]
                st.info(f"🔄 Cargando modelo universal: {os.path.basename(universal_model)}")
                try:
                    model_data = joblib.load(universal_model)
                    if not self.modelo_600:
                        self.modelo_600 = model_data['model']
                        self.scaler_600 = model_data['scaler']
                        st.success("✅ Modelo universal asignado a 600 DPI")
                    if not self.modelo_1200:
                        self.modelo_1200 = model_data['model']
                        self.scaler_1200 = model_data['scaler']
                        st.success("✅ Modelo universal asignado a 1200 DPI")
                except Exception as e:
                    st.error(f"❌ Error cargando modelo universal: {e}")
                
        except Exception as e:
            st.error(f"❌ Error general: {e}")
    
    def actualizar_estado_modelo(self):
        """Actualizar el estado del modelo en la interfaz"""
        estado_600 = "✅" if self.modelo_600 else "❌"
        estado_1200 = "✅" if self.modelo_1200 else "❌"
        
        if self.modelo_600 and self.modelo_1200:
            return f"🚀 Sistema listo - Ambos esquemas cargados (600 DPI: {estado_600} | 1200 DPI: {estado_1200})"
        elif self.modelo_600 or self.modelo_1200:
            return f"⚠️ Sistema parcialmente configurado (600 DPI: {estado_600} | 1200 DPI: {estado_1200})"
        else:
            return "❌ Sistema no configurado - Cargue los esquemas primero"
    
    def cambiar_resolucion(self, resolucion):
        """Cambiar modelo según resolución seleccionada"""
        if resolucion == "600" and self.modelo_600 is not None:
            self.modelo_actual = self.modelo_600
            self.scaler_actual = self.scaler_600
            st.success(f"🔧 Modelo 600 DPI activado")
        elif resolucion == "1200" and self.modelo_1200 is not None:
            self.modelo_actual = self.modelo_1200
            self.scaler_actual = self.scaler_1200
            st.success(f"🔧 Modelo 1200 DPI activado")
        else:
            self.modelo_actual = None
            st.error(f"❌ No hay modelo disponible para {resolucion} DPI")
    
    def detectar_dpi_real(self, img):
        """Detectar DPI de metadatos - CÓDIGO ORIGINAL"""
        try:
            dpi_x, dpi_y = img.info.get('dpi', (72, 72))
            dpi_promedio = (dpi_x + dpi_y) / 2
            if dpi_promedio <= 1:
                return 300
            else:
                return dpi_promedio
        except:
            return 300
    
    def optimizar_imagen(self, img_array):
        """Optimizar imagen a 2,000,000 píxeles máximo (resize) - CÓDIGO ORIGINAL"""
        height, width = img_array.shape[:2]
        total_pixels = height * width
        
        if total_pixels <= max_pixels:
            return img_array, False
        
        scale_factor = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        img_pil = Image.fromarray(img_array)
        img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return np.array(img_resized), True
    
    def aplicar_ingenieria_caracteristicas(self, X):
        """EXACTAMENTE LA MISMA ingeniería de características que en la versión original"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Características básicas de color
        intensity = X.mean(axis=1).reshape(-1, 1)
        saturation = (X.max(axis=1) - X.min(axis=1)).reshape(-1, 1)
        
        # Evitar división por cero en dominancia
        sum_rgb = X.sum(axis=1) + 1e-8
        dominance_r = (X[:, 0] / sum_rgb).reshape(-1, 1)
        dominance_g = (X[:, 1] / sum_rgb).reshape(-1, 1)
        dominance_b = (X[:, 2] / sum_rgb).reshape(-1, 1)
        
        # Características adicionales
        red_channel = X[:, 0].reshape(-1, 1)
        green_channel = X[:, 1].reshape(-1, 1)
        blue_channel = X[:, 2].reshape(-1, 1)
        
        # Luminosidad (fórmula estándar)
        luminance = (0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel).reshape(-1, 1)
        
        # Diferencia entre canales (en lugar de ratios problemáticos)
        rg_diff = (red_channel - green_channel).reshape(-1, 1)
        rb_diff = (red_channel - blue_channel).reshape(-1, 1)
        gb_diff = (green_channel - blue_channel).reshape(-1, 1)
        
        # Combinar características (sin ratios problemáticos)
        X_enhanced = np.hstack([
            X,                    # Características originales (3)
            intensity,           # Intensidad (1)
            saturation,          # Saturación (1)
            luminance,           # Luminosidad (1)
            dominance_r,         # Dominancia de rojo (1)
            dominance_g,         # Dominancia de verde (1)
            dominance_b,         # Dominancia de azul (1)
            rg_diff,             # Diferencia R-G (1)
            rb_diff,             # Diferencia R-B (1)  
            gb_diff,             # Diferencia G-B (1)
            red_channel**2,      # Términos cuadráticos (3)
            green_channel**2,
            blue_channel**2,
            np.sqrt(np.maximum(red_channel, 0)),    # Raíces cuadradas (3)
            np.sqrt(np.maximum(green_channel, 0)),
            np.sqrt(np.maximum(blue_channel, 0)),
            red_channel * green_channel,  # Interacciones (3)
            red_channel * blue_channel,
            green_channel * blue_channel
        ])
        
        # Limpiar cualquier valor problemático
        X_enhanced = np.nan_to_num(X_enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        
        st.info(f"🔧 Ingeniería de características: {X.shape[1]} → {X_enhanced.shape[1]} características")
        
        return X_enhanced
    
    def calcular_consumo_fisico_original(self, cmykrg_predictions, img_shape, resolucion_y, image_path):
        """Calcular consumo físico de tinta - MISMOS CÁLCULOS QUE v0.9"""
        try:
            st.info("🔍 Iniciando cálculo de consumo físico (MÉTODO v0.9)...")
            
            densidad_tinta = 1.05  # g/ml
            dpi_x = float(resolucion_x)  # 600 DPI fijo en X
            dpi_y = float(resolucion_y)
            
            # Obtener dimensiones reales de la imagen (igual que v0.9)
            with Image.open(image_path) as img:
                width_orig, height_orig = img.size
                dpi_real = self.detectar_dpi_real(img)
            
            # Calcular dimensiones reales en cm (igual que v0.9)
            ancho_cm = (width_orig / dpi_real) * 2.54
            alto_cm = (height_orig / dpi_real) * 2.54
            area_m2 = (ancho_cm * alto_cm) / 10000.0

            st.info(f"📐 Dimensiones imagen: {width_orig} x {height_orig} píxeles")
            st.info(f"📏 Dimensiones físicas: {ancho_cm:.1f} x {alto_cm:.1f} cm")
            st.info(f"📊 Área: {area_m2:.6f} m²")
            st.info(f"🎯 DPI real: {dpi_real}, DPI impresión: {dpi_x}x{dpi_y}")
            
            # FÓRMULA EXACTA v0.9 para puntos por m²
            puntos_por_m2 = (dpi_x / 2.54) * (dpi_y / 2.54) * 10000
            st.info(f"🔢 Puntos por m²: {puntos_por_m2:,.0f}")
            
            # Mismo volumen por punto que v0.9
            vol_por_punto_ml = 15e-9
            vol_max_ml_m2 = puntos_por_m2 * vol_por_punto_ml
            st.info(f"💧 Volumen máximo por m²: {vol_max_ml_m2:.6f} ml")
            
            # Mismo cálculo de coberturas que v0.9
            coberturas = np.mean(cmykrg_predictions / 100.0, axis=0)
            st.info(f"🎨 Coberturas promedio: {coberturas}")
            
            # Mismos factores de cabezal que v0.9
            factores_cabezal = {
                'Cian': 2.0, 'Magenta': 2.0, 'Amarillo': 2.0,
                'Negro': 2.0, 'Rojo': 1.0, 'Verde': 1.0
            }
            
            canales = ['Cian', 'Magenta', 'Amarillo', 'Negro', 'Rojo', 'Verde']
            consumo_total_g_m2 = 0
            consumos_detallados = {}
            
            # MISMO CÁLCULO POR CANAL que v0.9
            for i, canal in enumerate(canales):
                factor = factores_cabezal[canal]
                cobertura = coberturas[i]
                
                # Fórmula idéntica a v0.9
                vol_ml_m2 = cobertura * vol_max_ml_m2 * factor
                masa_g_m2 = vol_ml_m2 * densidad_tinta
                vol_ml_total = vol_ml_m2 * area_m2
                masa_g_total = masa_g_m2 * area_m2
                
                consumo_total_g_m2 += masa_g_m2
                
                consumos_detallados[canal] = {
                    'cobertura_promedio': cobertura * 100,
                    'volumen_ml_m2': vol_ml_m2,
                    'masa_g_m2': masa_g_m2,
                    'ml_total': vol_ml_total,
                    'g_total': masa_g_total,
                    'factores_cabezal': factor
                }
                
                st.info(f"  {canal}: {cobertura*100:.1f}% -> {masa_g_m2:.4f} g/m²")

            consumo_total_ml = consumo_total_g_m2 * area_m2 / densidad_tinta
            consumo_total_g = consumo_total_g_m2 * area_m2
            
            st.success(f"✅ Consumo TOTAL: {consumo_total_g_m2:.4f} g/m²")
            
            return {
                'total_g_m2': consumo_total_g_m2,
                'total_ml': consumo_total_ml,
                'total_g': consumo_total_g,
                'area_m2': area_m2,
                'resolucion': f"{dpi_x}x{dpi_y} DPI",
                'consumos_detallados': consumos_detallados,
                'dimensiones': f"{ancho_cm:.1f}x{alto_cm:.1f} cm",
                'dpi_real': dpi_real
            }

        except Exception as e:
            st.error(f"❌ ERROR en calcular_consumo_fisico_original: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def procesar_imagen_completo(self, uploaded_file, resolucion_y):
        """Procesamiento completo de la imagen - CÓDIGO ORIGINAL ADAPTADO"""
        try:
            # Guardar archivo temporalmente
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Cargando imagen...")
            progress_bar.progress(10)
            
            with Image.open("temp_image.jpg") as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                status_text.text("Optimizando imagen...")
                progress_bar.progress(30)
                
                img_array = np.array(img)
                img_optimized, was_optimized = self.optimizar_imagen(img_array)
                
                if was_optimized:
                    st.warning("⚠️ Imagen optimizada por tamaño")
                
                status_text.text("Preparando datos...")
                progress_bar.progress(50)
                
                pixels = img_optimized.reshape(-1, 3)
                
                status_text.text("Aplicando ingeniería de características...")
                progress_bar.progress(60)
                
                pixels_enhanced = self.aplicar_ingenieria_caracteristicas(pixels)
                
                status_text.text("Escalando datos...")
                progress_bar.progress(70)
                
                pixels_scaled = self.scaler_actual.transform(pixels_enhanced)
                
                status_text.text("Realizando descomposición por color...")
                progress_bar.progress(80)
                
                cmykrg_predictions = self.modelo_actual.predict(pixels_scaled)
                cmykrg_predictions = np.clip(cmykrg_predictions, 0, 100)
                
                status_text.text("Calculando consumo...")
                progress_bar.progress(90)
                
                resultados = self.calcular_consumo_fisico_original(
                    cmykrg_predictions, 
                    img_optimized.shape, 
                    resolucion_y,
                    "temp_image.jpg"
                )
                
                progress_bar.progress(100)
                status_text.text("Completado!")
                
                # Limpiar archivo temporal
                try:
                    os.remove("temp_image.jpg")
                except:
                    pass
                
                return resultados
                
        except Exception as e:
            st.error(f"❌ Error en procesamiento: {str(e)}")
            # Limpiar archivo temporal en caso de error
            try:
                os.remove("temp_image.jpg")
            except:
                pass
            return None

    def cargar_modelo_manual(self, resolucion, uploaded_model):
        """Cargar un modelo manualmente - SOLO TÉCNICOS"""
        try:
            # Guardar archivo temporalmente
            with open("temp_model.pkl", "wb") as f:
                f.write(uploaded_model.getbuffer())
            
            model_data = joblib.load("temp_model.pkl")
            filename = uploaded_model.name
            
            if resolucion == "600":
                self.modelo_600 = model_data['model']
                self.scaler_600 = model_data['scaler']
                st.success(f"✅ Modelo 600 DPI cargado: {filename}")
            else:
                self.modelo_1200 = model_data['model']
                self.scaler_1200 = model_data['scaler']
                st.success(f"✅ Modelo 1200 DPI cargado: {filename}")
            
            # Limpiar archivo temporal
            os.remove("temp_model.pkl")
            
        except Exception as e:
            st.error(f"❌ No se pudo cargar el modelo: {str(e)}")
            try:
                os.remove("temp_model.pkl")
            except:
                pass

# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================

    def mostrar_interfaz_principal(self):
        """Interfaz principal de la aplicación"""
        st.set_page_config(
            page_title="Simulador Consumo Tinta",
            page_icon="🖨️",
            layout="wide"
        )
        
        # Header con información de usuario
        col_title, col_user = st.columns([3, 1])
        
        with col_title:
            if st.session_state.tipo_usuario == "tecnico":
                st.title("🖨️ Simulador de Consumo - MODO TÉCNICO 🔧")
            else:
                st.title("🖨️ Simulador de Consumo - MODO USUARIO 👤")
        
        with col_user:
            st.write(f"**Usuario:** {st.session_state.usuario_actual}")
            if st.button("🚪 Cerrar Sesión"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        st.markdown("---")
        
        # Pestañas principales
        tab1, tab2 = st.tabs(["⚙️ Configuración y Cálculo", "📊 Resultados"])
        
        with tab1:
            self.mostrar_configuracion()
        
        with tab2:
            self.mostrar_resultados()
    
    def mostrar_configuracion(self):
        """Pestaña de configuración"""
        st.header("Configuración del Análisis")
        
        # Estado del sistema
        estado_modelo = self.actualizar_estado_modelo()
        st.info(estado_modelo)
        
        # Upload de imagen
        uploaded_file = st.file_uploader(
            "📁 Subir imagen RGB",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
            help="Formatos soportados: JPG, PNG, BMP, TIFF"
        )
        
        # Configuración de resolución
        col_res, col_btn = st.columns([1, 2])
        
        with col_res:
            resolucion = st.selectbox(
                "🎯 Resolución Y (DPI)",
                options=["600", "1200"],
                index=0,
                key="resolucion_select"
            )
            # Actualizar modelo según resolución
            self.cambiar_resolucion(resolucion)
        
        # Información de la imagen
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    st.image(image, caption="Vista previa", use_column_width=True)
                
                with col_info:
                    st.subheader("📊 Información de la Imagen")
                    
                    filename = uploaded_file.name
                    width, height = image.size
                    total_pixels = width * height
                    dpi_real = self.detectar_dpi_real(image)
                    ancho_cm = (width / dpi_real) * 2.54
                    alto_cm = (height / dpi_real) * 2.54
                    area_m2 = (ancho_cm * alto_cm) / 10000.0
                    
                    info_data = {
                        "Archivo": filename,
                        "Tamaño": f"{width} × {height} píxeles",
                        "Píxeles totales": f"{total_pixels:,}",
                        "DPI detectado": f"{dpi_real:.0f}",
                        "Dimensiones físicas": f"{ancho_cm:.1f} × {alto_cm:.1f} cm",
                        "Área de impresión": f"{area_m2:.4f} m²",
                        "Modo": image.mode
                    }
                    
                    for key, value in info_data.items():
                        st.write(f"**{key}:** {value}")
                
                # Botón de cálculo
                col_btn1, col_btn2 = st.columns([2, 1])
                
                with col_btn1:
                    if st.button("🎯 CALCULAR CONSUMO DE TINTA", 
                               type="primary", 
                               use_container_width=True,
                               disabled=(self.modelo_actual is None)):
                        
                        if self.modelo_actual is None:
                            st.error("❌ No hay modelo cargado para la resolución seleccionada")
                        else:
                            with st.spinner("Procesando imagen..."):
                                resultados = self.procesar_imagen_completo(uploaded_file, resolucion)
                                st.session_state.ultimos_resultados = resultados
                                st.success("✅ Cálculo completado! Ve a la pestaña 'Resultados'")
                        
            except Exception as e:
                st.error(f"❌ Error procesando imagen: {e}")
        
        # Funciones técnicas (solo para técnicos)
        if st.session_state.tipo_usuario == "tecnico":
            st.markdown("---")
            st.subheader("🛠️ Funciones Técnicas")
            
            col_tec1, col_tec2, col_tec3 = st.columns(3)
            
            with col_tec1:
                if st.button("🔄 Recargar Modelos Automáticamente", use_container_width=True):
                    self.cargar_modelos()
                    st.rerun()
            
            with col_tec2:
                st.write("**Cargar Modelo Manual:**")
                modelo_file = st.file_uploader("Subir modelo .pkl", type=['pkl'], key="model_upload")
                modelo_res = st.selectbox("Para resolución:", ["600", "1200"])
                if modelo_file and st.button("📥 Cargar Modelo", use_container_width=True):
                    self.cargar_modelo_manual(modelo_res, modelo_file)
                    st.rerun()
            
            with col_tec3:
                if st.button("📊 Ver Info del Sistema", use_container_width=True):
                    st.write(f"**Directorio:** {self.script_dir}")
                    st.write(f"**Modelo 600 cargado:** {self.modelo_600 is not None}")
                    st.write(f"**Modelo 1200 cargado:** {self.modelo_1200 is not None}")
                    st.write(f"**Modelo actual:** {self.modelo_actual is not None}")
    
    def mostrar_resultados(self):
        """Mostrar resultados del cálculo"""
        st.header("📊 Resultados del Consumo")
        
        if st.session_state.ultimos_resultados is None:
            st.info("ℹ️ Ejecuta un cálculo en la pestaña 'Configuración' para ver resultados aquí")
            return
        
        resultados = st.session_state.ultimos_resultados
        
        if not resultados:
            st.error("❌ No hay resultados válidos para mostrar")
            return
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Consumo por m²",
                f"{resultados['total_g_m2']:.2f} g/m²"
            )
        
        with col2:
            st.metric(
                "Consumo Total", 
                f"{resultados['total_g']:.2f} g"
            )
        
        with col3:
            st.metric(
                "Volumen Total",
                f"{resultados['total_ml']:.2f} ml"
            )
        
        with col4:
            st.metric(
                "Área de Impresión",
                f"{resultados['area_m2']:.4f} m²"
            )
        
        # Información detallada
        with st.expander("📋 Detalles del Análisis", expanded=True):
            col_det1, col_det2 = st.columns(2)
            
            with col_det1:
                st.write("**Configuración:**")
                st.write(f"- Resolución: {resultados['resolucion']}")
                st.write(f"- DPI real detectado: {resultados.get('dpi_real', 'N/A')}")
                st.write(f"- Dimensiones: {resultados['dimensiones']}")
                st.write(f"- Método: Estimación CMYK Doble + RG Simple")
            
            with col_det2:
                st.write("**Especificaciones:**")
                st.write(f"- Densidad tinta: {densidad_tinta} g/ml")
                st.write(f"- Resolución X fija: {resolucion_x} DPI")
                st.write(f"- Configuración: GS4 5-10-15pl")
        
        # Consumo por tinta (detalle para técnicos)
        if st.session_state.tipo_usuario == "tecnico" and 'consumos_detallados' in resultados:
            st.subheader("🎨 Consumo Detallado por Tinta")
            
            # Crear dataframe para mejor visualización
            tintas_data = []
            for tinta, datos in resultados['consumos_detallados'].items():
                tintas_data.append({
                    'Tinta': tinta,
                    'Cobertura (%)': f"{datos['cobertura_promedio']:.1f}%",
                    'Consumo (g/m²)': f"{datos['masa_g_m2']:.4f}",
                    'Volumen (ml/m²)': f"{datos['volumen_ml_m2']:.6f}",
                    'Total (g)': f"{datos['g_total']:.4f}",
                    'Total (ml)': f"{datos['ml_total']:.4f}",
                    'Tipo Cabezal': "Doble" if datos['factores_cabezal'] == 2.0 else "Simple"
                })
            
            df_tintas = pd.DataFrame(tintas_data)
            st.dataframe(df_tintas, use_container_width=True)
            
            # Gráfico de consumos
            st.subheader("📈 Distribución de Consumo por Tinta")
            consumos_g = [float(datos['masa_g_m2']) for datos in resultados['consumos_detallados'].values()]
            tintas = list(resultados['consumos_detallados'].keys())
            
            chart_data = pd.DataFrame({
                'Tinta': tintas,
                'Consumo (g/m²)': consumos_g
            })
            
            st.bar_chart(chart_data.set_index('Tinta'))

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    inicializar_sesion()
    
    if not st.session_state.autenticado:
        mostrar_login()
    else:
        app = CMYKRGConverterSimple()
        app.mostrar_interfaz_principal()

if __name__ == "__main__":
    main()