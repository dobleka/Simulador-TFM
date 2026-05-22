import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# Configuración del entorno
st.set_page_config(page_title="Transitorio del Equilibrio", layout="wide")
st.title("Dinámica Transitoria del Equilibrio Químico (Proceso Haber-Bosch)")

# Panel paramétrico: Control de colisiones
st.sidebar.header("Parámetros Cinéticos")
kd = st.sidebar.slider("Constante Directa (kd)", 0.001, 0.100, 0.020, format="%.3f")
ki = st.sidebar.slider("Constante Inversa (ki)", 0.001, 0.100, 0.005, format="%.3f")

# Panel paramétrico: Estrés químico
st.sidebar.header("Perturbación (t = 50s)")
iny_N2 = st.sidebar.number_input("Inyección N2 (M)", 0.0, 5.0, 2.0, 0.5)
iny_H2 = st.sidebar.number_input("Inyección H2 (M)", 0.0, 5.0, 0.0, 0.5)
iny_NH3 = st.sidebar.number_input("Inyección NH3 (M)", 0.0, 5.0, 0.0, 0.5)

# Matriz Jacobiana y EDOs
def cinetica_haber(t, y):
    N2, H2, NH3 = y
    vd = kd * N2 * (H2**3)
    vi = ki * (NH3**2)
    dN2_dt = -vd + vi
    dH2_dt = -3*vd + 3*vi
    dNH3_dt = 2*vd - 2*vi
    return [dN2_dt, dH2_dt, dNH3_dt]

# Fase 1: Integración hasta el estado estacionario inicial
y0 = [2.0, 3.0, 0.0]
t_fase1 = np.linspace(0, 50, 200)
sol1 = solve_ivp(cinetica_haber, [0, 50], y0, t_eval=t_fase1, method='BDF')

# Vector de interrupción: Salto discreto de masa
y_pert = [sol1.y[0][-1] + iny_N2, 
          sol1.y[1][-1] + iny_H2, 
          sol1.y[2][-1] + iny_NH3]

# Fase 2: Integración de la relajación amortiguada
t_fase2 = np.linspace(50, 150, 400)
sol2 = solve_ivp(cinetica_haber, [50, 150], y_pert, t_eval=t_fase2, method='BDF')

# Ensamblaje de matrices temporales
t_total = np.concatenate((sol1.t, sol2.t))
N2_total = np.concatenate((sol1.y[0], sol2.y[0]))
H2_total = np.concatenate((sol1.y[1], sol2.y[1]))
NH3_total = np.concatenate((sol1.y[2], sol2.y[2]))

# Renderizado vectorial
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_total, y=N2_total, name="[N2]", line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=t_total, y=H2_total, name="[H2]", line=dict(color='red', width=3)))
fig.add_trace(go.Scatter(x=t_total, y=NH3_total, name="[NH3]", line=dict(color='green', width=3)))

fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="Estrés Químico")
fig.update_layout(xaxis_title="Tiempo (s)", yaxis_title="Concentración (M)", height=600)
st.plotly_chart(fig, use_container_width=True)