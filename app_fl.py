import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import glob
import os
import sys
import time
from io import BytesIO
import datetime
import scipy.ndimage

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
                        st.rerun()
                    else:
                        st.error("❌ Contraseña incorrecta")
        else:
            st.info("No hay usuarios técnicos configurados")
    
    st.markdown("---")
    st.info("💡 **Nota:** Esta aplicación es de acceso restringido. Contacta al administrador para obtener credenciales.")

# =============================================================================
# CLASE PRINCIPAL - VERSIÓN CORREGIDA
# =============================================================================

class CMYKRGConverterCompleto:
    def __init__(self):
        self.modelo_600 = None
        self.modelo_1200 = None
        self.scaler_600 = None
        self.scaler_1200 = None
        self.modelo_actual = None
        self.scaler_actual = None
        
        # CONFIGURACIÓN DE TAMAÑOS DE GOTA
        self.tamanos_gota = {
            'pequena': 6.3,    # pl - para detalles finos
            'mediana': 12.6,   # pl - cobertura media
            'grande': 18.9     # pl - áreas sólidas
        }
        
        # ✅ DISTRIBUCIONES POR TIPO DE TRABAJO
        self.distribuciones = {
            'fotografia': {
                'baja': [0.70, 0.25, 0.05],    # < 20% cobertura
                'media': [0.40, 0.45, 0.15],   # 20-50% cobertura  
                'alta': [0.20, 0.50, 0.30]     # > 50% cobertura
            },
            'comercial': {
                'baja': [0.50, 0.40, 0.10],
                'media': [0.30, 0.50, 0.20], 
                'alta': [0.15, 0.45, 0.40]
            },
            'industrial': {
                'baja': [0.30, 0.50, 0.20],
                'media': [0.20, 0.40, 0.40],
                'alta': [0.10, 0.30, 0.60]
            }
        }
        
        # ✅ TIPO DE TRABAJO POR DEFECTO
        self.tipo_trabajo_actual = 'comercial'
        
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

    # =============================================================================
    # MÉTODOS DE FILTRADO DE BLANCOS - VERSIÓN OPTIMIZADA
    # =============================================================================
    
    def aplicar_filtro_blancos_tolerancia(self, img_array, cmykrg_predictions, tolerancia=5):
        """Filtra blancos puros y casi blancos - VERSIÓN OPTIMIZADA"""
        height, width = img_array.shape[:2]
        total_pixels_img = height * width
        total_pixels_pred = len(cmykrg_predictions)
        
        # Si las dimensiones coinciden, aplicar filtro
        if total_pixels_img == total_pixels_pred:
            img_flat = img_array.reshape(-1, 3)
            
            # ✅ VERSIÓN OPTIMIZADA: Usar condiciones combinadas
            # Filtrar píxeles donde TODOS los canales son 254 o 255
            blancos_mask = np.all((img_flat == 254) | (img_flat == 255), axis=1)
            
            if np.any(blancos_mask):
                # Aplicar filtro - poner cobertura 0% en píxeles blancos y casi blancos
                cmykrg_predictions[blancos_mask] = 0
                
                # Solo mostrar info si es técnico
                if st.session_state.tipo_usuario == "tecnico":
                    píxeles_blancos = np.sum(blancos_mask)
                    porcentaje_blancos = (píxeles_blancos / len(blancos_mask)) * 100
                    
                    # Contar blancos puros separadamente
                    blancos_puros = np.all(img_flat == 255, axis=1)
                    píxeles_puros = np.sum(blancos_puros)
                    píxeles_casi_blancos = píxeles_blancos - píxeles_puros
                    
                    st.success(f"✅ Filtro blancos aplicado:")
                    st.success(f"   - Píxeles blancos puros (255,255,255): {píxeles_puros:,}")
                    st.success(f"   - Píxeles casi blancos (combinaciones 254/255): {píxeles_casi_blancos:,}")
                    st.success(f"   - Total filtrado: {píxeles_blancos:,} píxeles ({porcentaje_blancos:.1f}%)")
        
        return cmykrg_predictions

    # =============================================================================
    # MÉTODOS DE DITHERING - CORREGIDOS
    # =============================================================================
    
    def aplicar_dithering_floyd_steinberg(self, cmykrg_predictions, img_shape):
        """Aplicar dithering Floyd-Steinberg a las predicciones CMYKRG"""
        height, width = img_shape[:2]
        canales = cmykrg_predictions.shape[1]
        
        # Reformatear a imagen 2D por canal
        predicciones_2d = cmykrg_predictions.reshape(height, width, canales)
        resultado = np.zeros_like(predicciones_2d)
        
        # Aplicar dithering canal por canal
        for canal in range(canales):
            canal_data = predicciones_2d[:, :, canal].astype(np.float32)
            canal_ditherado = self._dither_canal_floyd_steinberg(canal_data)
            resultado[:, :, canal] = canal_ditherado
        
        return resultado.reshape(-1, canales)
    
    def _dither_canal_floyd_steinberg(self, canal_data):
        """Aplicar dithering Floyd-Steinberg a un solo canal"""
        img = canal_data.copy()
        h, w = img.shape
        
        # Umbral para decidir si imprimir o no (50%)
        umbral = 50.0
        
        for y in range(h):
            for x in range(w):
                old_pixel = img[y, x]
                
                # Decidir si imprimir este punto
                new_pixel = 100.0 if old_pixel >= umbral else 0.0
                img[y, x] = new_pixel
                
                # Calcular error
                error = old_pixel - new_pixel
                
                # Difundir error a vecinos (Floyd-Steinberg)
                if x + 1 < w:
                    img[y, x + 1] += error * 7/16
                if y + 1 < h:
                    if x - 1 >= 0:
                        img[y + 1, x - 1] += error * 3/16
                    img[y + 1, x] += error * 5/16
                    if x + 1 < w:
                        img[y + 1, x + 1] += error * 1/16
        
        return np.clip(img, 0, 100)

    # =============================================================================
    # MÉTODO PRINCIPAL CORREGIDO - CON 3 TIPOS DE DISTRIBUCIÓN
    # =============================================================================
    
    def get_distribucion_gotas(self, cobertura_canal):
        """Obtener distribución según tipo de trabajo actual y cobertura"""
        # ✅ CORRECCIÓN: Verificar que el tipo de trabajo existe
        if self.tipo_trabajo_actual not in self.distribuciones:
            st.warning(f"⚠️ Tipo de trabajo '{self.tipo_trabajo_actual}' no encontrado, usando 'comercial'")
            self.tipo_trabajo_actual = 'comercial'
        
        if cobertura_canal < 0.20:
            categoria = 'baja'
        elif cobertura_canal < 0.50:
            categoria = 'media'
        else:
            categoria = 'alta'
        
        distribucion = self.distribuciones[self.tipo_trabajo_actual][categoria]
        
        # ✅ DIAGNÓSTICO: Mostrar qué distribución se está usando
        if st.session_state.tipo_usuario == "tecnico":
            st.info(f"🎯 Distribución: {self.tipo_trabajo_actual.upper()} - {categoria.upper()} -> P:{distribucion[0]*100:.0f}%, M:{distribucion[1]*100:.0f}%, G:{distribucion[2]*100:.0f}%")
        
        return distribucion
    
    def calcular_consumo_con_modelo_completo(self, cmykrg_predictions, img_shape, resolucion_y, image, filename):
    """VERSIÓN CORREGIDA con diagnóstico mejorado"""
    
    # ✅ DIAGNÓSTICO MEJORADO - Verificar PREDICCIONES REALES
    def verificar_predicciones_detalladas(cmykrg_predictions, img_array, sample_size=10):
        """Diagnóstico detallado de las predicciones"""
        total_pixels = len(cmykrg_predictions)
        
        # Estadísticas básicas
        cobertura_promedio = np.mean(cmykrg_predictions)
        cobertura_maxima = np.max(cmykrg_predictions)
        cobertura_minima = np.min(cmykrg_predictions)
        
        st.info(f"📊 ESTADÍSTICAS PREDICCIONES:")
        st.info(f"  - Cobertura promedio: {cobertura_promedio:.4f}%")
        st.info(f"  - Cobertura máxima: {cobertura_maxima:.2f}%")
        st.info(f"  - Cobertura mínima: {cobertura_minima:.2f}%")
        
        # Contar píxeles con diferentes niveles de cobertura
        pixeles_cero = np.sum(cmykrg_predictions == 0)
        pixeles_baja = np.sum((cmykrg_predictions > 0) & (cmykrg_predictions <= 1))
        pixeles_media = np.sum((cmykrg_predictions > 1) & (cmykrg_predictions <= 10))
        pixeles_alta = np.sum(cmykrg_predictions > 10)
        
        st.info(f"🔍 DISTRIBUCIÓN DE COBERTURA:")
        st.info(f"  - Píxeles con 0%: {pixeles_cero:,} ({(pixeles_cero/total_pixels)*100:.1f}%)")
        st.info(f"  - Píxeles 0-1%: {pixeles_baja:,} ({(pixeles_baja/total_pixels)*100:.1f}%)")
        st.info(f"  - Píxeles 1-10%: {pixeles_media:,} ({(pixeles_media/total_pixels)*100:.1f}%)")
        st.info(f"  - Píxeles >10%: {pixeles_alta:,} ({(pixeles_alta/total_pixels)*100:.1f}%)")
        
        # Muestra de píxeles con diferentes niveles de cobertura
        st.info("🎯 MUESTRA DE PÍXELES CON COBERTURA:")
        img_flat = img_array.reshape(-1, 3)
        
        # Buscar píxeles con cobertura > 0
        pixeles_con_cobertura = np.where(np.any(cmykrg_predictions > 0, axis=1))[0]
        
        if len(pixeles_con_cobertura) > 0:
            st.success(f"✅ Se encontraron {len(pixeles_con_cobertura)} píxeles con cobertura > 0%")
            # Mostrar primeros píxeles con cobertura
            for i in range(min(5, len(pixeles_con_cobertura))):
                idx = pixeles_con_cobertura[i]
                rgb = img_flat[idx]
                pred = cmykrg_predictions[idx]
                st.info(f"  Pixel {idx}: RGB{rgb} -> CMYKRG{pred}")
        else:
            st.error("❌ TODOS los píxeles tienen cobertura 0% - REVISAR MODELO")
            
            # Mostrar primeros píxeles aunque sean 0 para diagnóstico
            st.info("🔍 Primeros píxeles (todos 0%):")
            for i in range(min(5, total_pixels)):
                rgb = img_flat[i]
                pred = cmykrg_predictions[i]
                st.info(f"  Pixel {i}: RGB{rgb} -> CMYKRG{pred}")

    # ✅ DIAGNÓSTICO COMPLETO
    if st.session_state.tipo_usuario == "tecnico":
        verificar_predicciones_detalladas(cmykrg_predictions, np.array(image))

    # Verificar que hay predicciones válidas
    if np.all(cmykrg_predictions == 0):
        st.error("❌ No se puede calcular: todas las predicciones son 0")
        return None

    # 1. CALCULAR PUNTOS DE IMPRESIÓN POR m²
    dpi_x = float(resolucion_x)
    dpi_y = float(resolucion_y)
    puntos_por_pulgada2 = dpi_x * dpi_y
    puntos_por_m2 = puntos_por_pulgada2 * (10000 / (2.54 * 2.54))
    
    if st.session_state.tipo_usuario == "tecnico":
        st.info(f"🎯 Resolución: {dpi_x}x{dpi_y} DPI")
        st.info(f"🎯 Puntos por m²: {puntos_por_m2:,.0f}")
        st.info(f"🎯 Tipo de trabajo: {self.tipo_trabajo_actual.upper()}")
    
    # 2. USAR LAS PREDICCIONES POR CANAL del modelo entrenado
    canales = ['Cian', 'Magenta', 'Amarillo', 'Negro', 'Rojo', 'Verde']
    factores_cabezal = [2.25, 2.25, 2.25, 2.25, 1.15, 1.15]
    
    # Calcular cobertura promedio POR CANAL
    coberturas_por_canal = np.mean(cmykrg_predictions / 100.0, axis=0)
    
    if st.session_state.tipo_usuario == "tecnico":
        coberturas_info = [f"{canal}: {coberturas_por_canal[i]*100:.1f}%" for i, canal in enumerate(canales)]
        st.info(f"🎨 Coberturas por canal: {', '.join(coberturas_info)}")
    
    consumo_total_g_m2 = 0
    consumo_total_ml = 0
    consumos_detallados = {}
    
    for i, canal in enumerate(canales):
        cobertura_canal = coberturas_por_canal[i]
        factor_cabezal = factores_cabezal[i]
        
        # 3. CALCULAR PUNTOS A IMPRIMIR por m² para ESTE CANAL
        puntos_a_imprimir_por_m2 = puntos_por_m2 * cobertura_canal
        
        # 4. ✅ USAR DISTRIBUCIÓN SEGÚN TIPO DE TRABAJO
        distribucion = self.get_distribucion_gotas(cobertura_canal)
        puntos_pequenos = puntos_a_imprimir_por_m2 * distribucion[0]
        puntos_medianos = puntos_a_imprimir_por_m2 * distribucion[1]
        puntos_grandes = puntos_a_imprimir_por_m2 * distribucion[2]
        
        # 5. CALCULAR VOLUMEN para este canal
        volumen_pl_por_m2 = (puntos_pequenos * self.tamanos_gota['pequena'] + 
                            puntos_medianos * self.tamanos_gota['mediana'] + 
                            puntos_grandes * self.tamanos_gota['grande'])
        volumen_ml_por_m2 = volumen_pl_por_m2 * 1e-9
        
        # 6. APLICAR FACTOR DE CABEZAL específico
        volumen_canal_ml_m2 = volumen_ml_por_m2 * factor_cabezal
        masa_canal_g_m2 = volumen_canal_ml_m2 * densidad_tinta
        
        consumo_total_g_m2 += masa_canal_g_m2
        
        # Guardar detalles por canal
        total_puntos_canal = puntos_pequenos + puntos_medianos + puntos_grandes
        if total_puntos_canal > 0:
            dist_pequena = (puntos_pequenos / total_puntos_canal) * 100
            dist_mediana = (puntos_medianos / total_puntos_canal) * 100
            dist_grande = (puntos_grandes / total_puntos_canal) * 100
        else:
            dist_pequena = dist_mediana = dist_grande = 0
        
        consumos_detallados[canal] = {
            'cobertura_promedio': cobertura_canal * 100,
            'puntos_por_m2': puntos_a_imprimir_por_m2,
            'volumen_ml_m2': volumen_canal_ml_m2,
            'masa_g_m2': masa_canal_g_m2,
            'distribucion_gotas': f"P:{dist_pequena:.1f}%, M:{dist_mediana:.1f}%, G:{dist_grande:.1f}%",
            'puntos_pequenos': int(puntos_pequenos),
            'puntos_medianos': int(puntos_medianos),
            'puntos_grandes': int(puntos_grandes),
            'factor_cabezal': factor_cabezal,
            'tipo_distribucion': self.tipo_trabajo_actual
        }
    
    # 7. CALCULAR PARA EL ÁREA ESPECÍFICA
    width_orig, height_orig = image.size
    dpi_real = self.detectar_dpi_real(image)
    ancho_cm = (width_orig / dpi_real) * 2.54
    alto_cm = (height_orig / dpi_real) * 2.54
    area_m2 = (ancho_cm * alto_cm) / 10000.0
    
    consumo_total_g = consumo_total_g_m2 * area_m2
    consumo_total_ml = consumo_total_g / densidad_tinta
    
    if st.session_state.tipo_usuario == "tecnico":
        st.info(f"📊 Área calculada: {area_m2:.6f} m²")
        st.info(f"📈 Consumo total: {consumo_total_g_m2:.4f} g/m²")
    
    return {
        'total_g_m2': consumo_total_g_m2,
        'total_ml': consumo_total_ml,
        'total_g': consumo_total_g,
        'area_m2': area_m2,
        'resolucion': f"{dpi_x}x{dpi_y} DPI",
        'consumos_detallados': consumos_detallados,
        'dimensiones': f"{ancho_cm:.1f}x{alto_cm:.1f} cm",
        'dpi_real': dpi_real,
        'archivo_procesado': filename,
        'metodo': 'MODELO_COMPLETO_CORREGIDO',
        'tipo_trabajo': self.tipo_trabajo_actual,
        'tamanos_gota_utilizados': f"{self.tamanos_gota['pequena']}pl, {self.tamanos_gota['mediana']}pl, {self.tamanos_gota['grande']}pl",
        'puntos_por_m2': puntos_por_m2
    }

    # =============================================================================
    # MÉTODOS DE PROCESAMIENTO PRINCIPAL
    # =============================================================================
    
    def procesar_imagen_completa(self, uploaded_file, resolucion_y, batch_size=10000):
    """Procesamiento completo con diagnóstico detallado del modelo"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Cargando imagen...")
        progress_bar.progress(5)
        
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        status_text.text("Optimizando imagen...")
        progress_bar.progress(15)
        
        img_array = np.array(image)
        img_optimized, was_optimized = self.optimizar_imagen(img_array)
        
        if was_optimized and st.session_state.tipo_usuario == "tecnico":
            st.warning("⚠️ Imagen optimizada por tamaño")
        
        # ✅ DIAGNÓSTICO: VERIFICAR MODELO Y SCALER
        if self.modelo_actual is None:
            st.error("❌ ERROR: Modelo actual es None")
            return None
            
        if self.scaler_actual is None:
            st.error("❌ ERROR: Scaler actual es None")
            return None
        
        if st.session_state.tipo_usuario == "tecnico":
            st.info(f"🔧 Modelo cargado: {type(self.modelo_actual).__name__}")
            st.info(f"🔧 Scaler cargado: {type(self.scaler_actual).__name__}")
        
        status_text.text("Procesando por lotes...")
        progress_bar.progress(25)
        
        # Procesamiento por lotes con diagnóstico
        height, width = img_optimized.shape[:2]
        total_pixels = height * width
        batch_predictions = []
        num_batches = (total_pixels + batch_size - 1) // batch_size
        
        if st.session_state.tipo_usuario == "tecnico":
            st.info(f"🔧 Procesando {total_pixels:,} píxeles en {num_batches} lotes de {batch_size}")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_pixels)
            
            progress = 25 + (batch_idx / num_batches) * 40
            status_text.text(f"Procesando lote {batch_idx + 1}/{num_batches}...")
            progress_bar.progress(int(progress))
            
            batch_pixels = img_optimized.reshape(-1, 3)[start_idx:end_idx]
            
            # ✅ DIAGNÓSTICO: VERIFICAR DATOS DE ENTRADA
            if st.session_state.tipo_usuario == "tecnico" and batch_idx == 0:
                st.info("🔍 DIAGNÓSTICO - Primer lote:")
                st.info(f"  - Batch pixels shape: {batch_pixels.shape}")
                st.info(f"  - Batch pixels range: [{batch_pixels.min()}, {batch_pixels.max()}]")
                st.info(f"  - Batch pixels muestra: {batch_pixels[0]}")
            
            batch_enhanced = self.aplicar_ingenieria_caracteristicas(batch_pixels)
            
            # ✅ DIAGNÓSTICO: VERIFICAR INGENIERÍA DE CARACTERÍSTICAS
            if st.session_state.tipo_usuario == "tecnico" and batch_idx == 0:
                st.info(f"  - Características mejoradas shape: {batch_enhanced.shape}")
                st.info(f"  - Características mejoradas range: [{batch_enhanced.min():.2f}, {batch_enhanced.max():.2f}]")
            
            # ✅ DIAGNÓSTICO: VERIFICAR SCALER
            try:
                batch_scaled = self.scaler_actual.transform(batch_enhanced)
                
                if st.session_state.tipo_usuario == "tecnico" and batch_idx == 0:
                    st.info(f"  - Datos escalados shape: {batch_scaled.shape}")
                    st.info(f"  - Datos escalados range: [{batch_scaled.min():.2f}, {batch_scaled.max():.2f}]")
                    st.info(f"  - Datos escalados muestra: {batch_scaled[0][:5]}...")  # Primeros 5 valores
                    
            except Exception as e:
                st.error(f"❌ ERROR en scaler.transform(): {e}")
                if st.session_state.tipo_usuario == "tecnico":
                    st.error(f"🔧 Scaler info: {self.scaler_actual}")
                    st.error(f"🔧 Data stats - min: {batch_enhanced.min()}, max: {batch_enhanced.max()}, mean: {batch_enhanced.mean()}")
                return None
            
            # ✅ DIAGNÓSTICO: VERIFICAR PREDICCIÓN DEL MODELO
            try:
                batch_pred = self.modelo_actual.predict(batch_scaled)
                
                if st.session_state.tipo_usuario == "tecnico" and batch_idx == 0:
                    st.info(f"  - Predicciones shape: {batch_pred.shape}")
                    st.info(f"  - Predicciones range: [{batch_pred.min():.2f}, {batch_pred.max():.2f}]")
                    st.info(f"  - Predicciones muestra: {batch_pred[0]}")
                    
            except Exception as e:
                st.error(f"❌ ERROR en model.predict(): {e}")
                if st.session_state.tipo_usuario == "tecnico":
                    st.error(f"🔧 Modelo info: {type(self.modelo_actual).__name__}")
                    st.error(f"🔧 Input shape para predict: {batch_scaled.shape}")
                return None
            
            batch_predictions.append(batch_pred)
        
        status_text.text("Combinando predicciones...")
        progress_bar.progress(70)
        
        cmykrg_predictions = np.vstack(batch_predictions)
        cmykrg_predictions = np.clip(cmykrg_predictions, 0, 100)
        
        # ✅ DIAGNÓSTICO COMPLETO DE LAS PREDICCIONES FINALES
        if st.session_state.tipo_usuario == "tecnico":
            self.diagnosticar_predicciones(cmykrg_predictions, img_array)
        
        # VERIFICAR SI TODAS LAS PREDICCIONES SON CERO
        if np.all(cmykrg_predictions == 0):
            st.error("🚨 PROBLEMA CRÍTICO: Todas las predicciones son 0")
            st.error("📋 Posibles causas:")
            st.error("  1. Modelo no está entrenado correctamente")
            st.error("  2. Scaler no compatible con los datos")
            st.error("  3. Problema en la ingeniería de características")
            st.error("  4. Modelo corrupto o incorrecto")
            return None
        
        # APLICAR FILTROS
        status_text.text("Aplicando optimizaciones...")
        progress_bar.progress(80)
        
        img_array_original = np.array(image)
        cmykrg_filtrado = self.aplicar_filtro_blancos_tolerancia(img_array_original, cmykrg_predictions.copy())
        cmykrg_optimizado = self.aplicar_dithering_floyd_steinberg(cmykrg_filtrado, img_optimized.shape)
        
        # CALCULAR CONSUMO
        status_text.text("Calculando consumo...")
        progress_bar.progress(90)
        
        resultados = self.calcular_consumo_con_modelo_completo(
            cmykrg_optimizado, 
            img_optimized.shape, 
            resolucion_y, 
            image, 
            uploaded_file.name
        )
        
        progress_bar.progress(100)
        status_text.text("Completado!")
        
        self.agregar_log_procesamiento(uploaded_file, resolucion_y, resultados, exito=True)
        
        return resultados
        
    except Exception as e:
        error_msg = f"Error en procesamiento completo: {str(e)}"
        st.error(f"❌ {error_msg}")
        
        if st.session_state.tipo_usuario == "tecnico":
            import traceback
            st.error(f"🔧 Detalles completos: {traceback.format_exc()}")
            
        self.agregar_log_procesamiento(uploaded_file, resolucion_y, None, exito=False, error_msg=error_msg)
        return None

    def diagnosticar_predicciones(self, cmykrg_predictions, img_array):
        """Diagnóstico detallado de las predicciones del modelo"""
        st.info("🔍 DIAGNÓSTICO COMPLETO DE PREDICCIONES:")
        
        total_pixels = len(cmykrg_predictions)
        
        # Estadísticas básicas
        st.info(f"📊 Estadísticas generales:")
        st.info(f"  - Total píxeles: {total_pixels:,}")
        st.info(f"  - Shape predicciones: {cmykrg_predictions.shape}")
        
        # Estadísticas por canal
        canales = ['Cian', 'Magenta', 'Amarillo', 'Negro', 'Rojo', 'Verde']
        for i, canal in enumerate(canales):
            canal_data = cmykrg_predictions[:, i]
            st.info(f"🎨 {canal}:")
            st.info(f"    - Min: {canal_data.min():.4f}")
            st.info(f"    - Max: {canal_data.max():.4f}")
            st.info(f"    - Mean: {canal_data.mean():.4f}")
            st.info(f"    - Std: {canal_data.std():.4f}")
            st.info(f"    - Píxeles > 0: {np.sum(canal_data > 0):,} ({(np.sum(canal_data > 0)/total_pixels)*100:.2f}%)")
        
        # Distribución de cobertura total
        cobertura_total = np.sum(cmykrg_predictions, axis=1)
        st.info(f"📈 Cobertura total por píxel:")
        st.info(f"    - Min: {cobertura_total.min():.4f}")
        st.info(f"    - Max: {cobertura_total.max():.4f}")
        st.info(f"    - Mean: {cobertura_total.mean():.4f}")
        
        # Contar píxeles por rangos de cobertura
        st.info(f"📋 Distribución de cobertura:")
        for umbral in [0, 0.1, 1, 5, 10, 50, 100]:
            if umbral == 0:
                count = np.sum(cobertura_total == 0)
                st.info(f"    - = 0%: {count:,} ({(count/total_pixels)*100:.2f}%)")
            else:
                count = np.sum(cobertura_total > umbral)
                st.info(f"    - > {umbral}%: {count:,} ({(count/total_pixels)*100:.2f}%)")
        
        # Muestra de píxeles con diferentes coberturas
        img_flat = img_array.reshape(-1, 3)
        
        # Píxeles con cobertura > 0
        pixeles_con_cobertura = np.where(cobertura_total > 0)[0]
        if len(pixeles_con_cobertura) > 0:
            st.success(f"✅ Píxeles con cobertura > 0: {len(pixeles_con_cobertura):,}")
            st.info("🔍 Muestra de píxeles CON cobertura:")
            for i in range(min(3, len(pixeles_con_cobertura))):
                idx = pixeles_con_cobertura[i]
                rgb = img_flat[idx]
                pred = cmykrg_predictions[idx]
                st.info(f"    Pixel {idx}: RGB{rgb} -> CMYKRG{pred}")
        else:
            st.error("❌ TODOS los píxeles tienen cobertura 0%")
            st.info("🔍 Muestra de píxeles (todos 0%):")
            for i in range(min(5, total_pixels)):
                rgb = img_flat[i]
                pred = cmykrg_predictions[i]
                st.info(f"    Pixel {i}: RGB{rgb} -> CMYKRG{pred}")
    
    # =============================================================================
    # MÉTODOS ORIGINALES (MANTENER)
    # =============================================================================
    
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
            'area_m2': resultados['area_m2'] if resultados and exito else None,
            'tipo_trabajo': resultados.get('tipo_trabajo', 'comercial') if resultados else None
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
                'Tipo Trabajo': log.get('tipo_trabajo', 'comercial'),
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
    # INTERFAZ CONFIGURACIÓN - SIMPLIFICADA Y CORREGIDA
    # =============================================================================
    
    def mostrar_configuracion_trabajo(self):
        """Interfaz simplificada para seleccionar tipo de trabajo"""
        if st.session_state.tipo_usuario != "tecnico":
            return
        
        st.markdown("---")
        st.subheader("🎯 Configuración de Tipo de Trabajo")
        
        # Seleccionar tipo de trabajo
        tipo_trabajo = st.selectbox(
            "Tipo de trabajo:",
            ["comercial", "fotografia", "industrial"],
            index=["comercial", "fotografia", "industrial"].index(self.tipo_trabajo_actual),
            help="Define la estrategia de distribución de gotas"
        )
        
        # Aplicar inmediatamente al cambiar
        if tipo_trabajo != self.tipo_trabajo_actual:
            self.tipo_trabajo_actual = tipo_trabajo
            st.success(f"✅ Tipo de trabajo actualizado: {tipo_trabajo.upper()}")
    
        # Mostrar configuración actual
        st.info(f"**Configuración actual:** {self.tipo_trabajo_actual.upper()}")

    def mostrar_configuracion_gotas(self):
        """Interfaz simplificada para configurar tamaños de gota"""
        if st.session_state.tipo_usuario != "tecnico":
            return
        
        st.markdown("---")
        st.subheader("💧 Configuración de Tamaños de Gota")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gota_pequena = st.number_input(
                "Gota pequeña (pl)", 
                value=self.tamanos_gota['pequena'],
                min_value=1.0, 
                max_value=50.0, 
                step=0.1
            )
        
        with col2:
            gota_mediana = st.number_input(
                "Gota mediana (pl)", 
                value=self.tamanos_gota['mediana'],
                min_value=1.0, 
                max_value=50.0, 
                step=0.1
            )
        
        with col3:
            gota_grande = st.number_input(
                "Gota grande (pl)", 
                value=self.tamanos_gota['grande'],
                min_value=1.0, 
                max_value=50.0, 
                step=0.1
            )
        
        # Botón para aplicar cambios
        if st.button("💾 Aplicar Configuración de Gotas"):
            self.tamanos_gota = {
                'pequena': gota_pequena,
                'mediana': gota_mediana,
                'grande': gota_grande
            }
            st.success("✅ Configuración de gotas aplicada")
        
        # Mostrar configuración actual
        st.info(f"**Configuración actual:** "
                f"Pequeña: {self.tamanos_gota['pequena']}pl, "
                f"Mediana: {self.tamanos_gota['mediana']}pl, "
                f"Grande: {self.tamanos_gota['grande']}pl")

# =============================================================================
# INTERFAZ STREAMLIT - ACTUALIZADA
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
                st.info(f"🔧 **Modo:** {self.tipo_trabajo_actual.upper()} (configuración por defecto)")
        
        with col_user:
            st.write(f"**Usuario:** {st.session_state.usuario_actual}")
            if st.button("🚪 Cerrar Sesión", width='stretch'):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        st.markdown("---")
        
        # Pestañas diferentes según tipo de usuario
        if st.session_state.tipo_usuario == "tecnico":
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚙️ Configuración y Cálculo", "📊 Resultados", "🎯 Tipo Trabajo", "💧 Tamaños Gota", "📋 Historial"])
        else:
            tab1, tab2 = st.tabs(["📁 Subir y Calcular", "📊 Resultados"])
        
        with tab1:
            self.mostrar_configuracion()
        
        with tab2:
            self.mostrar_resultados()
            
        if st.session_state.tipo_usuario == "tecnico":
            with tab3:
                self.mostrar_configuracion_trabajo()
            with tab4:
                self.mostrar_configuracion_gotas()
            with tab5:
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
                            with st.spinner("Procesando imagen..."):
                                resultados = self.procesar_imagen_completa(
                                    uploaded_file, 
                                    resolucion, 
                                    batch_size=10000
                                )
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
                    st.write(f"**Tipo trabajo actual:** {self.tipo_trabajo_actual}")
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
        st.success(f"🔧 **Tipo de trabajo:** {resultados.get('tipo_trabajo', 'N/A').upper()}")
        
        # Información detallada SOLO para técnicos
        if st.session_state.tipo_usuario == "tecnico" and 'consumos_detallados' in resultados:
            st.subheader("🎨 Consumo Detallado por Tinta")
            
            tintas_data = []
            for tinta, datos in resultados['consumos_detallados'].items():
                tintas_data.append({
                    'Tinta': tinta,
                    'Cobertura (%)': f"{datos['cobertura_promedio']:.1f}%",
                    'Consumo (g/m²)': f"{datos['masa_g_m2']:.4f}",
                    'Volumen (ml/m²)': f"{datos['volumen_ml_m2']:.6f}",
                    'Puntos/m²': f"{datos['puntos_por_m2']:,.0f}",
                    'Distribución Gotas': datos['distribucion_gotas'],
                    'Tipo Distribución': datos.get('tipo_distribucion', 'comercial')
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
            
            # Distribución de tamaños de gota
            st.subheader("💧 Distribución de Tamaños de Gota")
            
            datos_gotas = []
            for tinta, datos in resultados['consumos_detallados'].items():
                total_puntos = datos['puntos_pequenos'] + datos['puntos_medianos'] + datos['puntos_grandes']
                if total_puntos > 0:
                    porcentaje_pequenos = (datos['puntos_pequenos'] / total_puntos) * 100
                    porcentaje_medianos = (datos['puntos_medianos'] / total_puntos) * 100
                    porcentaje_grandes = (datos['puntos_grandes'] / total_puntos) * 100
                else:
                    porcentaje_pequenos = porcentaje_medianos = porcentaje_grandes = 0
                    
                datos_gotas.append({
                    'Tinta': tinta,
                    'Gotas Pequeñas': f"{datos['puntos_pequenos']:,.0f} ({porcentaje_pequenos:.1f}%)",
                    'Gotas Medianas': f"{datos['puntos_medianos']:,.0f} ({porcentaje_medianos:.1f}%)", 
                    'Gotas Grandes': f"{datos['puntos_grandes']:,.0f} ({porcentaje_grandes:.1f}%)"
                })
            
            df_gotas = pd.DataFrame(datos_gotas)
            st.dataframe(df_gotas, width='stretch')

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    inicializar_sesion()
    
    if not st.session_state.autenticado:
        mostrar_login()
    else:
        app = CMYKRGConverterCompleto()
        app.mostrar_interfaz_principal()

if __name__ == "__main__":
    main()
