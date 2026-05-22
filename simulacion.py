import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Configuración de la estructura de página
st.set_page_config(page_title="Simulador Termocinético", layout="wide")
st.title("Indagación Computacional: Ecuación de Arrhenius y Teoría de Colisiones")

# Definición del panel de control paramétrico
st.sidebar.header("Variables de Estado")
T = st.sidebar.slider("Temperatura (K)", min_value=250, max_value=800, value=300, step=10)
Ea = st.sidebar.slider("Energía de Activación (kJ/mol)", min_value=20, max_value=120, value=50, step=5)
A0 = st.sidebar.slider("Concentración Inicial [A]₀ (M)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Constantes físico-químicas del sistema
R = 8.314 # J/(mol·K)
Ea_J = Ea * 1000 # Conversión a J/mol
A_factor = 1e8 # Factor preexponencial de colisiones

# Motor de cálculo cinético
k = A_factor * np.exp(-Ea_J / (R * T))

# Matriz 1: Distribución de energías de Maxwell-Boltzmann
E_array = np.linspace(0, 150000, 500) # Espectro de 0 a 150 kJ/mol
f_E = (2 / np.sqrt(np.pi)) * (1 / (R * T))**1.5 * np.sqrt(E_array) * np.exp(-E_array / (R * T))
f_E = f_E / np.max(f_E) # Normalización visual del eje Y

# Segmentación matemática del área de reacción (E > Ea)
E_reaccion = E_array[E_array >= Ea_J]
f_E_reaccion = f_E[E_array >= Ea_J]

# Matriz 2: Integración cinética de primer orden
tiempos = np.linspace(0, 100, 200)
concentraciones = A0 * np.exp(-k * tiempos)

# Creación de la interfaz de doble panel
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución de Maxwell-Boltzmann")
    fig1 = go.Figure()
    
    # Curva general de moléculas
    fig1.add_trace(go.Scatter(x=E_array/1000, y=f_E, mode='lines', name='Población Molecular', line=dict(color='blue')))
    
    # Sombreado del área de choque efectivo
    fig1.add_trace(go.Scatter(x=E_reaccion/1000, y=f_E_reaccion, fill='tozeroy', mode='none', name='Moléculas con E > Ea', fillcolor='rgba(255, 0, 0, 0.5)'))
    
    # Límite paramétrico de la Energía de Activación
    fig1.add_vline(x=Ea, line_dash="dash", line_color="red", annotation_text="Ea")
    
    fig1.update_layout(xaxis_title="Energía Cinética (kJ/mol)", yaxis_title="Fracción de Moléculas (Normalizada)", height=500)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Evolución Temporal de la Concentración")
    fig2 = go.Figure()
    
    # Curva de desintegración
    fig2.add_trace(go.Scatter(x=tiempos, y=concentraciones, mode='lines', name='[A] vs t', line=dict(color='green', width=3)))
    
    # Monitor de datos en tiempo real
    fig2.add_annotation(x=50, y=A0*0.8, text=f"k = {k:.4e} s⁻¹", showarrow=False, font=dict(size=16, color="black"), bgcolor="lightgrey")
    
    fig2.update_layout(xaxis_title="Tiempo (s)", yaxis_title="Concentración [A] (M)", height=500)
    st.plotly_chart(fig2, use_container_width=True)

# Generación del resumen de estado
st.info(f"El sistema opera a {T} K. La fracción de moléculas con energía superior a la barrera de activación aumenta exponencialmente al desplazar la temperatura o reducir la Energía de Activación, alterando el valor empírico de la constante k.")