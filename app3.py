import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER"] = "false"
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import glob
import sys
import time
from io import BytesIO
import datetime
# === NUEVOS IMPORTS PARA ANALYTICS ===
import requests
import json

# =============================================================================
# SISTEMA DE ANALYTICS CON GOOGLE SHEETS
# =============================================================================

class StreamlitAnalytics:
    def __init__(self):
        self.webhook_url = st.secrets.get("https://script.google.com/macros/s/AKfycbx_uTia_9Y4j9olGQVlqNGPv73yBgyA-lKLDAoWehEVAligGChW7ct1QcGlNgvuYWHb/exec", "")
        self.enabled = bool(self.webhook_url)
    
    def track_event(self, event_type, **metadata):
        """Registrar evento en Google Sheets"""
        if not self.enabled:
            return True
            
        try:
            payload = {
                'event_type': event_type,
                'timestamp': datetime.datetime.now().isoformat(),
                'usuario': st.session_state.get('usuario_actual', 'unknown'),
                'tipo_usuario': st.session_state.get('tipo_usuario', 'unknown'),
                **metadata
            }
            
            response = requests.post(
                self.webhook_url, 
                json=payload, 
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            # Fallar silenciosamente para no interrumpir la app
            return False
    
    def track_file_upload(self, uploaded_file, resolucion):
        """Registrar subida de archivo"""
        return self.track_event(
            'file_upload',
            filename=uploaded_file.name,
            file_size=uploaded_file.size,
            file_type=uploaded_file.type or uploaded_file.name.split('.')[-1],
            resolucion=resolucion
        )
    
    def track_calculation(self, uploaded_file, resultados, exito=True):
        """Registrar cálculo completado"""
        return self.track_event(
            'calculation_complete' if exito else 'calculation_error',
            filename=uploaded_file.name,
            file_size=uploaded_file.size,
            consumo_total=resultados.get('total_g_m2') if resultados else None,
            area_m2=resultados.get('area_m2') if resultados else None,
            exito=exito
        )

# Inicializar analytics global
analytics = StreamlitAnalytics()

# =============================================================================
# CONFIGURACIÓN GLOBAL - MANTENER ORIGINAL
# =============================================================================

Image.MAX_IMAGE_PIXELS = 1000000000  # 1 billón de píxeles (ORIGINAL)

# Configuración fija (ORIGINAL)
max_pixels = 2000000  # 2 millones (ORIGINAL)
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
    if 'historial_procesamientos' not in st.session_state:
        st.session_state.historial_procesamientos = []
    if 'estadisticas_uso' not in st.session_state:
        st.session_state.estadisticas_uso = {
            'total_archivos': 0,
            'archivos_exitosos': 0,
            'archivos_fallidos': 0,
            'primero_uso': None,
            'ultimo_uso': None
        }

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
    
    if not verificar_usuarios_configurados():
        st.error("⚠️ Sistema no configurado. Contacta al administrador.")
        return
    
    usuarios_permitidos = st.secrets.get("usuarios", {})
    
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
                if st.button("🔓 Acceder como Usuario", key="user_btn", width='stretch'):
                    usuario_data = usuarios_permitidos.get(usuario_user, {})
                    if usuario_data.get("password") == password_user:
                        st.session_state.autenticado = True
                        st.session_state.usuario_actual = usuario_user
                        st.session_state.tipo_usuario = "usuario"
                        if not st.session_state.estadisticas_uso['primero_uso']:
                            st.session_state.estadisticas_uso['primero_uso'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # === NUEVO: Trackear login ===
                        analytics.track_event('user_login', usuario=usuario_user, tipo_usuario="usuario")
                        
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
                if st.button("🔧 Acceder como Técnico", key="tec_btn", width='stretch'):
                    usuario_data = usuarios_permitidos.get(tecnico_user, {})
                    if usuario_data.get("password") == password_tec:
                        st.session_state.autenticado = True
                        st.session_state.usuario_actual = tecnico_user
                        st.session_state.tipo_usuario = "tecnico"
                        if not st.session_state.estadisticas_uso['primero_uso']:
                            st.session_state.estadisticas_uso['primero_uso'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # === NUEVO: Trackear login técnico ===
                        analytics.track_event('user_login', usuario=tecnico_user, tipo_usuario="tecnico")
                        
                        st.rerun()
                    else:
                        st.error("❌ Contraseña incorrecta")
        else:
            st.info("No hay usuarios técnicos configurados")
    
    st.markdown("---")
    st.info("💡 **Nota:** Esta aplicación es de acceso restringido. Contacta al administrador para obtener credenciales.")

# =============================================================================
# CLASE PRINCIPAL - VERSIÓN CON PROCESAMIENTO POR LOTES PERO RESULTADOS ORIGINALES
# =============================================================================

class CMYKRGConverterSimple:
    def __init__(self):
        self.modelo_600 = None
        self.modelo_1200 = None
        self.scaler_600 = None
        self.scaler_1200 = None
        self.modelo_actual = None
        self.scaler_actual = None
        
        if getattr(sys, 'frozen', False):
            self.script_dir = os.path.dirname(sys.executable)
        else:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Solo mostrar info técnica para técnicos
        if st.session_state.tipo_usuario == "tecnico":
            st.info(f"📂 Carpeta del script: {self.script_dir}")
            st.info(f"🔐 Modo: {st.session_state.tipo_usuario.upper()}")
        
        self.cargar_modelos()
    
    def cargar_modelos(self):
        """Cargar modelos específicos para 600 y 1200 DPI"""
        try:
            if st.session_state.tipo_usuario == "tecnico":
                st.info("🔍 BUSCANDO MODELOS ESPECÍFICOS POR RESOLUCIÓN...")
            
            pkl_files = glob.glob(os.path.join(self.script_dir, "*.pkl"))
            
            if not pkl_files:
                st.error("❌ NO se encontraron archivos .pkl en la carpeta del script")
                return
            
            if st.session_state.tipo_usuario == "tecnico":
                st.success(f"🎯 Archivos .pkl encontrados: {[os.path.basename(f) for f in pkl_files]}")
            
            modelo_600_path = None
            modelo_1200_path = None
            
            for file_path in pkl_files:
                filename = os.path.basename(file_path).lower()
                
                if '600' in filename and '1200' not in filename:
                    modelo_600_path = file_path
                    if st.session_state.tipo_usuario == "tecnico":
                        st.success(f"✅ Encontrado modelo 600 DPI: {os.path.basename(file_path)}")
                
                elif '1200' in filename and '600' not in filename:
                    modelo_1200_path = file_path
                    if st.session_state.tipo_usuario == "tecnico":
                        st.success(f"✅ Encontrado modelo 1200 DPI: {os.path.basename(file_path)}")
                
                elif 'modelo' in filename:
                    if '600' in filename:
                        modelo_600_path = file_path
                        if st.session_state.tipo_usuario == "tecnico":
                            st.success(f"✅ Asignado como modelo 600 DPI: {os.path.basename(file_path)}")
                    elif '1200' in filename:
                        modelo_1200_path = file_path
                        if st.session_state.tipo_usuario == "tecnico":
                            st.success(f"✅ Asignado como modelo 1200 DPI: {os.path.basename(file_path)}")
            
            if modelo_600_path:
                try:
                    model_data = joblib.load(modelo_600_path)
                    self.modelo_600 = model_data['model']
                    self.scaler_600 = model_data['scaler']
                    if st.session_state.tipo_usuario == "tecnico":
                        st.success(f"✅ Modelo 600 DPI cargado exitosamente")
                except Exception as e:
                    if st.session_state.tipo_usuario == "tecnico":
                        st.error(f"❌ Error cargando modelo 600 DPI: {e}")
            else:
                if st.session_state.tipo_usuario == "tecnico":
                    st.warning("❌ Modelo 600 DPI no encontrado")
            
            if modelo_1200_path:
                try:
                    model_data = joblib.load(modelo_1200_path)
                    self.modelo_1200 = model_data['model']
                    self.scaler_1200 = model_data['scaler']
                    if st.session_state.tipo_usuario == "tecnico":
                        st.success(f"✅ Modelo 1200 DPI cargado exitosamente")
                except Exception as e:
                    if st.session_state.tipo_usuario == "tecnico":
                        st.error(f"❌ Error cargando modelo 1200 DPI: {e}")
            else:
                if st.session_state.tipo_usuario == "tecnico":
                    st.warning("❌ Modelo 1200 DPI no encontrado")
            
            if len(pkl_files) == 1 and (not self.modelo_600 or not self.modelo_1200):
                universal_model = pkl_files[0]
                if st.session_state.tipo_usuario == "tecnico":
                    st.info(f"🔄 Cargando modelo universal: {os.path.basename(universal_model)}")
                try:
                    model_data = joblib.load(universal_model)
                    if not self.modelo_600:
                        self.modelo_600 = model_data['model']
                        self.scaler_600 = model_data['scaler']
                        if st.session_state.tipo_usuario == "tecnico":
                            st.success("✅ Modelo universal asignado a 600 DPI")
                    if not self.modelo_1200:
                        self.modelo_1200 = model_data['model']
                        self.scaler_1200 = model_data['scaler']
                        if st.session_state.tipo_usuario == "tecnico":
                            st.success("✅ Modelo universal asignado a 1200 DPI")
                except Exception as e:
                    if st.session_state.tipo_usuario == "tecnico":
                        st.error(f"❌ Error cargando modelo universal: {e}")
                
        except Exception as e:
            if st.session_state.tipo_usuario == "tecnico":
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
            if st.session_state.tipo_usuario == "tecnico":
                st.success(f"🔧 Modelo 600 DPI activado")
        elif resolucion == "1200" and self.modelo_1200 is not None:
            self.modelo_actual = self.modelo_1200
            self.scaler_actual = self.scaler_1200
            if st.session_state.tipo_usuario == "tecnico":
                st.success(f"🔧 Modelo 1200 DPI activado")
        else:
            self.modelo_actual = None
            if st.session_state.tipo_usuario == "tecnico":
                st.error(f"❌ No hay modelo disponible para {resolucion} DPI")
    
    def detectar_dpi_real(self, img):
        """Detectar DPI de metadatos"""
        try:
            dpi_x, dpi_y = img.info.get('dpi', (None, None))
            if dpi_x is None or dpi_y is None:
                raise ValueError("No se encontraron metadatos DPI en la imagen")
            # Convertir IFDRational a float si es necesario
            dpi_x = float(dpi_x) if hasattr(dpi_x, 'denominator') else dpi_x
            dpi_y = float(dpi_y) if hasattr(dpi_y, 'denominator') else dpi_y
            dpi_promedio = (dpi_x + dpi_y) / 2
            if dpi_promedio <= 1:
                raise ValueError("DPI inválido detectado en los metadatos de la imagen")
            return dpi_promedio
        except Exception as e:
            raise ValueError(f"No se pudo detectar el DPI de la imagen: {str(e)}")
    
    def optimizar_imagen(self, img_array):
        """Optimizar imagen a 2,000,000 píxeles máximo (resize) - ORIGINAL"""
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
        
        intensity = X.mean(axis=1).reshape(-1, 1)
        saturation = (X.max(axis=1) - X.min(axis=1)).reshape(-1, 1)
        
        sum_rgb = X.sum(axis=1) + 1e-8
        dominance_r = (X[:, 0] / sum_rgb).reshape(-1, 1)
        dominance_g = (X[:, 1] / sum_rgb).reshape(-1, 1)
        dominance_b = (X[:, 2] / sum_rgb).reshape(-1, 1)
        
        red_channel = X[:, 0].reshape(-1, 1)
        green_channel = X[:, 1].reshape(-1, 1)
        blue_channel = X[:, 2].reshape(-1, 1)
        
        luminance = (0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel).reshape(-1, 1)
        
        rg_diff = (red_channel - green_channel).reshape(-1, 1)
        rb_diff = (red_channel - blue_channel).reshape(-1, 1)
        gb_diff = (green_channel - blue_channel).reshape(-1, 1)
        
        X_enhanced = np.hstack([
            X, intensity, saturation, luminance,
            dominance_r, dominance_g, dominance_b,
            rg_diff, rb_diff, gb_diff,
            red_channel**2, green_channel**2, blue_channel**2,
            np.sqrt(np.maximum(red_channel, 0)),
            np.sqrt(np.maximum(green_channel, 0)),
            np.sqrt(np.maximum(blue_channel, 0)),
            red_channel * green_channel,
            red_channel * blue_channel,
            green_channel * blue_channel
        ])
        
        X_enhanced = np.nan_to_num(X_enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        
        if st.session_state.tipo_usuario == "tecnico":
            st.info(f"🔧 Ingeniería de características: {X.shape[1]} → {X_enhanced.shape[1]} características")
        
        return X_enhanced
    
    def agregar_log_procesamiento(self, uploaded_file, resolucion, resultados, exito=True, error_msg=None):
        """Agregar entrada al historial de procesamientos"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'usuario': st.session_state.usuario_actual,
            'tipo_usuario': st.session_state.tipo_usuario,
            'archivo_nombre': uploaded_file.name,
            'archivo_tamaño': uploaded_file.size,
            'archivo_tipo': uploaded_file.type,
            'resolucion': resolucion,
            'exito': exito,
            'error_msg': error_msg,
            'consumo_total': resultados['total_g_m2'] if resultados and exito else None,
            'area_m2': resultados['area_m2'] if resultados and exito else None
        }
        
        st.session_state.historial_procesamientos.insert(0, log_entry)
        
        stats = st.session_state.estadisticas_uso
        stats['total_archivos'] += 1
        stats['ultimo_uso'] = timestamp
        
        if exito:
            stats['archivos_exitosos'] += 1
        else:
            stats['archivos_fallidos'] += 1
        
        if len(st.session_state.historial_procesamientos) > 50:
            st.session_state.historial_procesamientos = st.session_state.historial_procesamientos[:50]
        
        # === NUEVO: Trackear en analytics también ===
        if exito and resultados:
            analytics.track_event(
                'processing_success',
                filename=uploaded_file.name,
                consumo_total=resultados['total_g_m2'],
                area_m2=resultados['area_m2']
            )
        else:
            analytics.track_event(
                'processing_error', 
                filename=uploaded_file.name,
                error_msg=error_msg
            )
    
    def mostrar_historial_procesamientos(self):
        """Mostrar historial de archivos procesados - SOLO TÉCNICOS"""
        if st.session_state.tipo_usuario != "tecnico":
            st.info("🔒 Esta función solo está disponible para usuarios técnicos")
            return
            
        if not st.session_state.historial_procesamientos:
            st.info("📝 Aún no se han procesado archivos en esta sesión")
            return
        
        st.subheader("📋 Historial de Procesamientos (Esta Sesión)")
        
        historial_data = []
        for log in st.session_state.historial_procesamientos[:20]:
            historial_data.append({
                'Fecha/Hora': log['timestamp'],
                'Archivo': log['archivo_nombre'],
                'Usuario': log['usuario'],
                'Resolución': log['resolucion'],
                'Estado': '✅ Éxito' if log['exito'] else '❌ Error',
                'Consumo (g/m²)': f"{log['consumo_total']:.2f}" if log['consumo_total'] else 'N/A',
                'Tamaño': f"{log['archivo_tamaño']:,} bytes"
            })
        
        if historial_data:
            df_historial = pd.DataFrame(historial_data)
            st.dataframe(df_historial, width='stretch')
        
        stats = st.session_state.estadisticas_uso
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Archivos", stats['total_archivos'])
        with col2:
            st.metric("Procesamientos Exitosos", stats['archivos_exitosos'])
        with col3:
            st.metric("Procesamientos Fallidos", stats['archivos_fallidos'])
        with col4:
            tasa_exito = (stats['archivos_exitosos'] / stats['total_archivos'] * 100) if stats['total_archivos'] > 0 else 0
            st.metric("Tasa de Éxito", f"{tasa_exito:.1f}%")
    
    def calcular_consumo_fisico_original(self, cmykrg_predictions, img_shape, resolucion_y, image, filename):
        """Calcular consumo físico de tinta - VERSIÓN ORIGINAL EXACTA"""
        try:
            if st.session_state.tipo_usuario == "tecnico":
                st.info("🔍 Iniciando cálculo de consumo físico (MÉTODO v0.9)...")
            
            densidad_tinta = 1.05
            dpi_x = float(resolucion_x)
            dpi_y = float(resolucion_y)
            
            width_orig, height_orig = image.size
            dpi_real = self.detectar_dpi_real(image)
            
            ancho_cm = (width_orig / dpi_real) * 2.54
            alto_cm = (height_orig / dpi_real) * 2.54
            area_m2 = (ancho_cm * alto_cm) / 10000.0

            if st.session_state.tipo_usuario == "tecnico":
                st.info(f"📐 Archivo: {filename}")
                st.info(f"📏 Dimensiones: {width_orig} x {height_orig} píxeles")
                st.info(f"📐 Dimensiones físicas: {ancho_cm:.1f} x {alto_cm:.1f} cm")
                st.info(f"📊 Área: {area_m2:.6f} m²")
                st.info(f"🎯 DPI real: {dpi_real}, DPI impresión: {dpi_x}x{dpi_y}")
            
            puntos_por_m2 = (dpi_x / 2.54) * (dpi_y / 2.54) * 10000
            
            if st.session_state.tipo_usuario == "tecnico":
                st.info(f"🔢 Puntos por m²: {puntos_por_m2:,.0f}")
            
            vol_por_punto_ml = 20e-9
            vol_max_ml_m2 = puntos_por_m2 * vol_por_punto_ml
            
            if st.session_state.tipo_usuario == "tecnico":
                st.info(f"💧 Volumen máximo por m²: {vol_max_ml_m2:.6f} ml")
            
            coberturas = np.mean(cmykrg_predictions / 100.0, axis=0)
            
            if st.session_state.tipo_usuario == "tecnico":
                st.info(f"🎨 Coberturas promedio: {coberturas}")
            
            factores_cabezal = {
                'Cian': 2.25, 'Magenta': 2.25, 'Amarillo': 2.25,
                'Negro': 2.25, 'Rojo': 1.15, 'Verde': 1.15
            }
            
            canales = ['Cian', 'Magenta', 'Amarillo', 'Negro', 'Rojo', 'Verde']
            consumo_total_g_m2 = 0
            consumos_detallados = {}
            
            for i, canal in enumerate(canales):
                factor = factores_cabezal[canal]
                cobertura = coberturas[i]
                
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
                
                if st.session_state.tipo_usuario == "tecnico":
                    st.info(f"  {canal}: {cobertura*100:.1f}% -> {masa_g_m2:.4f} g/m²")

            consumo_total_ml = consumo_total_g_m2 * area_m2 / densidad_tinta
            consumo_total_g = consumo_total_g_m2 * area_m2
            
            if st.session_state.tipo_usuario == "tecnico":
                st.success(f"✅ Consumo TOTAL: {consumo_total_g_m2:.4f} g/m²")
            
            return {
                'total_g_m2': consumo_total_g_m2,
                'total_ml': consumo_total_ml,
                'total_g': consumo_total_g,
                'area_m2': area_m2,
                'resolucion': f"{dpi_x}x{dpi_y} DPI",
                'consumos_detallados': consumos_detallados,
                'dimensiones': f"{ancho_cm:.1f}x{alto_cm:.1f} cm",
                'dpi_real': dpi_real,
                'archivo_procesado': filename
            }

        except ValueError as e:
            st.error(f"❌ Error en cálculo: {str(e)}")
            self.agregar_log_procesamiento(image, resolucion_y, None, exito=False, error_msg=str(e))
            return None
        except Exception as e:
            st.error(f"❌ ERROR en cálculo: {str(e)}")
            import traceback
            traceback.print_exc()
            self.agregar_log_procesamiento(image, resolucion_y, None, exito=False, error_msg=str(e))
            return None

    def procesar_imagen_por_lotes(self, uploaded_file, resolucion_y, batch_size=10000):
        """Procesamiento por LOTES para evitar problemas de memoria - PERO RESULTADOS ORIGINALES"""
        try:
            # === NUEVO: Trackear inicio de upload ===
            analytics.track_file_upload(uploaded_file, resolucion_y)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Cargando imagen...")
            progress_bar.progress(10)
            
            image = Image.open(uploaded_file)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            status_text.text("Optimizando imagen...")
            progress_bar.progress(30)
            
            img_array = np.array(image)
            img_optimized, was_optimized = self.optimizar_imagen(img_array)
            
            if was_optimized and st.session_state.tipo_usuario == "tecnico":
                st.warning("⚠️ Imagen optimizada por tamaño")
            
            status_text.text("Preparando procesamiento por lotes...")
            progress_bar.progress(40)
            
            # Obtener dimensiones
            height, width = img_optimized.shape[:2]
            total_pixels = height * width
            
            # Procesar por lotes
            batch_predictions = []
            num_batches = (total_pixels + batch_size - 1) // batch_size
            
            if st.session_state.tipo_usuario == "tecnico":
                st.info(f"🔧 Procesando {total_pixels:,} píxeles en {num_batches} lotes de {batch_size}")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_pixels)
                
                # Progreso
                progress = 40 + (batch_idx / num_batches) * 50
                status_text.text(f"Procesando lote {batch_idx + 1}/{num_batches}...")
                progress_bar.progress(int(progress))
                
                # Extraer lote actual
                batch_pixels = img_optimized.reshape(-1, 3)[start_idx:end_idx]
                
                # Aplicar ingeniería de características al lote (MISMO MÉTODO ORIGINAL)
                batch_enhanced = self.aplicar_ingenieria_caracteristicas(batch_pixels)
                
                # Escalar y predecir
                batch_scaled = self.scaler_actual.transform(batch_enhanced)
                batch_pred = self.modelo_actual.predict(batch_scaled)
                
                batch_predictions.append(batch_pred)
                
                # Liberar memoria periódicamente
                if batch_idx % 5 == 0:  # Cada 5 lotes
                    import gc
                    gc.collect()
            
            # Combinar todas las predicciones
            status_text.text("Combinando resultados...")
            progress_bar.progress(95)
            
            cmykrg_predictions = np.vstack(batch_predictions)
            cmykrg_predictions = np.clip(cmykrg_predictions, 0, 100)
            
            # Calcular consumo (MISMO MÉTODO ORIGINAL)
            resultados = self.calcular_consumo_fisico_original(
                cmykrg_predictions, 
                img_optimized.shape, 
                resolucion_y,
                image,
                uploaded_file.name
            )
            
            progress_bar.progress(100)
            status_text.text("Completado!")
            
            self.agregar_log_procesamiento(uploaded_file, resolucion_y, resultados, exito=True)
            
            # === NUEVO: Trackear cálculo exitoso ===
            analytics.track_calculation(uploaded_file, resultados, exito=True)
            
            # Limpiar memoria
            del img_array, img_optimized, batch_predictions, cmykrg_predictions
            import gc
            gc.collect()
            
            return resultados
        
        except Exception as e:
            error_msg = f"Error en procesamiento por lotes: {str(e)}"
            st.error(f"❌ {error_msg}")
            
            if st.session_state.tipo_usuario == "tecnico":
                import traceback
                st.error(f"🔧 Detalles: {traceback.format_exc()}")
                
            self.agregar_log_procesamiento(uploaded_file, resolucion_y, None, exito=False, error_msg=error_msg)
            
            # === NUEVO: Trackear error ===
            analytics.track_calculation(uploaded_file, None, exito=False)
            
            return None

    def cargar_modelo_manual(self, resolucion, uploaded_model):
        """Cargar un modelo manualmente - SOLO TÉCNICOS"""
        try:
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
            
            os.remove("temp_model.pkl")
            
        except Exception as e:
            st.error(f"❌ No se pudo cargar el modelo: {str(e)}")
            try:
                os.remove("temp_model.pkl")
            except:
                pass

# =============================================================================
# INTERFAZ STREAMLIT - ACTUALIZADA A width='stretch'
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
                st.title("🖨️ Simulador de Consumo de Tinta")
        
        with col_user:
            st.write(f"**Usuario:** {st.session_state.usuario_actual}")
            if st.button("🚪 Cerrar Sesión", width='stretch'):
                # === NUEVO: Trackear logout ===
                analytics.track_event('user_logout', usuario=st.session_state.usuario_actual)
                
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        st.markdown("---")
        
        # Pestañas diferentes según tipo de usuario
        if st.session_state.tipo_usuario == "tecnico":
            tab1, tab2, tab3 = st.tabs(["⚙️ Configuración y Cálculo", "📊 Resultados", "📋 Historial"])
        else:
            tab1, tab2 = st.tabs(["📁 Subir y Calcular", "📊 Resultados"])
        
        with tab1:
            self.mostrar_configuracion()
        
        with tab2:
            self.mostrar_resultados()
            
        if st.session_state.tipo_usuario == "tecnico":
            with tab3:
                self.mostrar_historial_procesamientos()
    
    def mostrar_configuracion(self):
        """Pestaña de configuración"""
        
        if st.session_state.tipo_usuario == "tecnico":
            st.header("Configuración del Análisis")
            estado_modelo = self.actualizar_estado_modelo()
            st.info(estado_modelo)
        else:
            st.header("📁 Subir Imagen y Calcular")
            st.info("💡 Sube una imagen RGB para calcular el consumo de tinta")
        
        # Upload de imagen (común para ambos)
        uploaded_file = st.file_uploader(
            "📁 Subir imagen RGB",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
            help="Formatos soportados: JPG, PNG, BMP, TIFF"
        )
        
        # Configuración de resolución
        if st.session_state.tipo_usuario == "tecnico":
            col_res, col_btn = st.columns([1, 2])
        else:
            col_res, col_btn = st.columns([1, 1])
        
        with col_res:
            resolucion = st.selectbox(
                "🎯 Resolución (DPI)",
                options=["600", "1200"],
                index=0,
                key="resolucion_select"
            )
            self.cambiar_resolucion(resolucion)
        
        # Información de la imagen
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                
                # Verificar DPI antes de mostrar información
                dpi_real = self.detectar_dpi_real(image)
                
                if st.session_state.tipo_usuario == "tecnico":
                    col_img, col_info = st.columns([1, 2])
                else:
                    col_img, col_info = st.columns([1, 1])
                
                with col_img:
                    st.image(image, caption="Vista previa", width='stretch')
                
                with col_info:
                    if st.session_state.tipo_usuario == "tecnico":
                        st.subheader("📊 Información de la Imagen")
                    else:
                        st.subheader("📋 Información de la Imagen")
                    
                    filename = uploaded_file.name
                    width, height = image.size
                    total_pixels = width * height
                    ancho_cm = (width / dpi_real) * 2.54
                    alto_cm = (height / dpi_real) * 2.54
                    area_m2 = (ancho_cm * alto_cm) / 10000.0
                    
                    # Información simplificada para usuarios
                    if st.session_state.tipo_usuario == "tecnico":
                        info_data = {
                            "Archivo": filename,
                            "Tamaño archivo": f"{uploaded_file.size:,} bytes",
                            "Tamaño imagen": f"{width} × {height} píxeles",
                            "Píxeles totales": f"{total_pixels:,}",
                            "DPI detectado": f"{dpi_real:.0f}",
                            "Dimensiones físicas": f"{ancho_cm:.1f} × {alto_cm:.1f} cm",
                            "Área de impresión": f"{area_m2:.4f} m²",
                            "Modo": image.mode
                        }
                    else:
                        info_data = {
                            "Archivo": filename,
                            "Tamaño": f"{width} × {height} píxeles",
                            "Dimensiones": f"{ancho_cm:.1f} × {alto_cm:.1f} cm",
                            "Área": f"{area_m2:.4f} m²"
                        }
                    
                    for key, value in info_data.items():
                        st.write(f"**{key}:** {value}")
                
                # Botón de cálculo
                if st.session_state.tipo_usuario == "tecnico":
                    col_btn1, col_btn2 = st.columns([2, 1])
                else:
                    col_btn1, col_btn2 = st.columns([1, 1])
                
                with col_btn1:
                    btn_text = "🎯 CALCULAR CONSUMO DE TINTA" if st.session_state.tipo_usuario == "tecnico" else "🎯 CALCULAR CONSUMO"
                    if st.button(btn_text, 
                               type="primary", 
                               width='stretch',
                               disabled=(self.modelo_actual is None)):
                        
                        if self.modelo_actual is None:
                            st.error("❌ No hay modelo cargado para la resolución seleccionada")
                        else:
                            with st.spinner("Procesando imagen (modo optimizado)..."):
                                # USAR LA VERSIÓN POR LOTES
                                resultados = self.procesar_imagen_por_lotes(uploaded_file, resolucion, batch_size=10000)
                                st.session_state.ultimos_resultados = resultados
                                if resultados:
                                    st.success(f"✅ {uploaded_file.name} procesado correctamente! Ve a la pestaña 'Resultados'")
                                else:
                                    st.error(f"❌ Error procesando {uploaded_file.name}")
                        
            except ValueError as e:
                st.error(f"❌ Error procesando imagen: {str(e)}")
                self.agregar_log_procesamiento(uploaded_file, resolucion, None, exito=False, error_msg=str(e))
            except Exception as e:
                st.error(f"❌ Error procesando imagen: {e}")
                self.agregar_log_procesamiento(uploaded_file, resolucion, None, exito=False, error_msg=str(e))
        
        # Funciones técnicas (solo para técnicos)
        if st.session_state.tipo_usuario == "tecnico":
            st.markdown("---")
            st.subheader("🛠️ Funciones Técnicas")
            
            col_tec1, col_tec2, col_tec3 = st.columns(3)
            
            with col_tec1:
                if st.button("🔄 Recargar Modelos Automáticamente", width='stretch'):
                    self.cargar_modelos()
                    st.rerun()
            
            with col_tec2:
                st.write("**Cargar Modelo Manual:**")
                modelo_file = st.file_uploader("Subir modelo .pkl", type=['pkl'], key="model_upload")
                modelo_res = st.selectbox("Para resolución:", ["600", "1200"])
                if modelo_file and st.button("📥 Cargar Modelo", width='stretch'):
                    self.cargar_modelo_manual(modelo_res, modelo_file)
                    st.rerun()
            
            with col_tec3:
                if st.button("📊 Ver Info del Sistema", width='stretch'):
                    st.write(f"**Directorio:** {self.script_dir}")
                    st.write(f"**Modelo 600 cargado:** {self.modelo_600 is not None}")
                    st.write(f"**Modelo 1200 cargado:** {self.modelo_1200 is not None}")
                    st.write(f"**Modelo actual:** {self.modelo_actual is not None}")
                    st.write(f"**Archivos en sesión:** {len(st.session_state.historial_procesamientos)}")
    
    def mostrar_resultados(self):
        """Mostrar resultados del cálculo"""
        st.header("📊 Resultados del Consumo")
        
        if st.session_state.ultimos_resultados is None:
            st.info("ℹ️ Ejecuta un cálculo en la pestaña anterior para ver resultados aquí")
            return
        
        resultados = st.session_state.ultimos_resultados
        
        if not resultados:
            st.error("❌ No hay resultados válidos para mostrar")
            return
        
        # Métricas principales (comunes para ambos)
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
        
        # Información del archivo procesado
        st.info(f"📁 **Archivo procesado:** {resultados.get('archivo_procesado', 'N/A')}")
        
        # Información detallada SOLO para técnicos
        if st.session_state.tipo_usuario == "tecnico":
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
        
        # Consumo por tinta SOLO para técnicos
        if st.session_state.tipo_usuario == "tecnico" and 'consumos_detallados' in resultados:
            st.subheader("🎨 Consumo Detallado por Tinta")
            
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
            st.dataframe(df_tintas, width='stretch')
            
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
