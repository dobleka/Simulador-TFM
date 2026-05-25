import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Celda Galvánica Paramétrica", layout="wide")
st.title("Electroquímica Dinámica: Pila Daniell con Ruido Instrumental")

# Panel lateral paramétrico
st.sidebar.header("Variables de Estado")
T = st.sidebar.slider("Temperatura (K)", 273, 373, 298, 1)
Zn_0 = st.sidebar.number_input("Concentración Zn²⁺ inicial (M)", 0.01, 2.00, 0.10, 0.05)
Cu_0 = st.sidebar.number_input("Concentración Cu²⁺ inicial (M)", 0.01, 2.00, 1.00, 0.05)
I_descarga = st.sidebar.slider("Corriente de descarga (A)", 0.01, 0.50, 0.10, 0.01)

# Constantes fisicoquímicas
R = 8.314 # J/(mol·K)
F = 96485 # C/mol
n = 2 # Electrones transferidos
E_std = 1.10 # V (Potencial estándar Zn/Cu)
Volumen = 1.0 # L

# Integración diferencial de la descarga (Leyes de Faraday)
def descarga_pila(t, y):
    Zn, Cu = y
    # La corriente constante agota el Cu2+ y genera Zn2+
    dZn = I_descarga / (n * F * Volumen)
    dCu = -I_descarga / (n * F * Volumen)
    return [dZn, dCu]

t_span = (0, 7200) # 2 horas de funcionamiento continuo
t_eval = np.linspace(0, 7200, 250)
sol = solve_ivp(descarga_pila, t_span, [Zn_0, Cu_0], t_eval=t_eval)

# Filtrado algorítmico para evitar el colapso logarítmico (concentraciones negativas)
Zn_t = np.clip(sol.y[0], 1e-6, None)
Cu_t = np.clip(sol.y[1], 1e-6, None)

# Nivel Termodinámico: Ecuación de Nernst con dispersión
Q_t = Zn_t / Cu_t
E_teorico = E_std - ((R * T) / (n * F)) * np.log(Q_t)

# Inyección de ruido estocástico (Simulación de interferencia en el voltímetro)
ruido_voltimetro = np.random.normal(0, 0.002, size=len(E_teorico))
E_experimental = E_teorico + ruido_voltimetro

# Renderizado vectorial interactivo
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_eval, y=E_experimental, mode='markers', 
                         marker=dict(size=4, color='orange', opacity=0.7), 
                         name='Voltaje Empírico (V)'))
fig.add_trace(go.Scatter(x=t_eval, y=E_teorico, mode='lines', 
                         line=dict(color='black', width=1, dash='dash'), 
                         name='Modelo Ideal'))
fig.update_layout(xaxis_title="Tiempo de operación (s)", yaxis_title="Fuerza Electromotriz (V)", height=500)

st.plotly_chart(fig, use_container_width=True)

# Exportación de la matriz de datos para su tratamiento analítico en aula
df = pd.DataFrame({
    "Tiempo (s)": t_eval, 
    "[Zn2+] (M)": Zn_t, 
    "[Cu2+] (M)": Cu_t, 
    "FEM Observada (V)": E_experimental
})

st.download_button(
    label="Extraer matriz de datos del voltímetro (CSV)", 
    data=df.to_csv(index=False).encode('utf-8'), 
    file_name="descarga_celda.csv", 
    mime="text/csv"
)

with st.expander("Mostrar el modelo analítico del sistema"):
    st.latex(r"E = E^\circ - \frac{RT}{nF} \ln Q")