import streamlit as st
import numpy as np
import time
from fractions import Fraction

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Factorización LU para Matrices Simétricas",
    page_icon="🔢",
    layout="wide"
)

# --- Título y Descripción ---
st.title("Programa de Factorización LU para Matrices Simétricas")
st.markdown("""
Esta aplicación implementa un algoritmo para la descomposición de matrices simétricas. 
El programa analiza la matriz y determina el método de factorización adecuado.
""")

# --- Algoritmos de Factorización (Respetando el código base) ---

def crear_matriz_simetrica(n):
    """
    Genera una matriz simétrica de orden n con valores enteros aleatorios.
    Ahora devuelve la matriz como tipo float para consistencia.
    """
    matriz_temp = np.random.randint(-10, 11, size=(n, n))
    matriz_simetrica = matriz_temp + matriz_temp.T
    return matriz_simetrica.astype(float)

def factorizacion_lu_gauss(A):
    """
    Intenta la factorización A=LU usando eliminación Gaussiana simple.
    Si encuentra un pivote cero, retorna None, None.
    """
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.identity(n, dtype=float)
    for k in range(n - 1):
        if abs(U[k, k]) < 1e-9:
            return None, None
        for i in range(k + 1, n):
            multiplicador = U[i, k] / U[k, k]
            L[i, k] = multiplicador
            U[i, :] -= multiplicador * U[k, :]
    return L, U

def factorizacion_plu(A):
    """
    Realiza la factorización P.T*L*U con pivoteo.
    Modificado para manejar matrices singulares sin fallar.
    """
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.identity(n, dtype=float)
    P = np.identity(n, dtype=float)
    for k in range(n):
        # Lógica de pivoteo: busca la primera fila válida para intercambiar
        pivot_search_slice = U[k:, k]
        nonzero_indices = np.nonzero(np.abs(pivot_search_slice) > 1e-9)[0]

        if len(nonzero_indices) > 0:
            first_nonzero_relative_index = nonzero_indices[0]
            pivot_row_index = first_nonzero_relative_index + k
        else:
            pivot_row_index = k
            
        if pivot_row_index != k:
            U[[k, pivot_row_index]] = U[[pivot_row_index, k]]
            P[[k, pivot_row_index]] = P[[pivot_row_index, k]]
            if k > 0:
                L[[k, pivot_row_index], :k] = L[[pivot_row_index, k], :k]

        # --- LÓGICA CORREGIDA PARA MATRICES SINGULARES ---
        # Si el pivote es no nulo, se procede con la eliminación.
        # Si es nulo, se omite este paso y se continúa, permitiendo que la matriz U resultante tenga ceros en su diagonal.
        if abs(U[k, k]) > 1e-9:
            for i in range(k + 1, n):
                multiplicador = U[i, k] / U[k, k]
                L[i, k] = multiplicador
                U[i, k:] -= multiplicador * U[k, k:]
                U[i, k] = 0
                
    return P, L, U


def to_fraction_matrix(matrix):
    """
    Convierte una matriz de NumPy con floats a una con objetos Fraction para mayor exactitud.
    """
    return np.vectorize(lambda x: Fraction(x).limit_denominator(100000000))(matrix)

# --- Interfaz de Usuario ---

with st.sidebar:
    st.header("Controles de Entrada")
    n_size = st.slider("Orden de la Matriz (n)", min_value=4, max_value=10, value=4, step=1)
    
    input_method = st.radio(
        "Método de Creación de la Matriz A",
        ("Generar matriz simétrica aleatoria", "Ingresar matriz manualmente")
    )
    
    calcular = st.button("Calcular Factorización", type="primary", use_container_width=True)


if 'matriz_a' not in st.session_state or st.session_state.n_size != n_size:
    st.session_state.n_size = n_size
    st.session_state.matriz_a = crear_matriz_simetrica(n_size)

st.subheader("Paso 1: Definición de la Matriz de Entrada A")

if input_method == "Ingresar matriz manualmente":
    st.write(f"Ingrese los {n_size*n_size} elementos de la matriz A:")
    
    temp_A = np.zeros((n_size, n_size), dtype=float)
    grid_cols = st.columns(n_size)
    for i in range(n_size):
        for j in range(n_size):
            default_value = float(st.session_state.matriz_a[i, j])
            temp_A[i, j] = grid_cols[j].number_input(
                f"A[{i+1},{j+1}]", 
                value=default_value, 
                key=f"cell_{i}_{j}",
                label_visibility="collapsed"
            )
    st.session_state.matriz_a = temp_A
    st.write("**Matriz A Ingresada:**")
    st.dataframe(st.session_state.matriz_a, use_container_width=True)
else:
    if st.button("Generar Nueva Matriz Simétrica"):
        st.session_state.matriz_a = crear_matriz_simetrica(n_size)
    st.write("**Matriz A Aleatoria:**")
    st.dataframe(st.session_state.matriz_a, use_container_width=True)


if calcular:
    A = st.session_state.matriz_a
    st.divider()
    st.subheader("Paso 2: Análisis y Resultados de la Factorización")
    
    with st.spinner('Paso 1: Intentando resolver por el método directo A = LU...'):
        time.sleep(1)
        L, U = factorizacion_lu_gauss(A)

    if L is not None:
        st.success("ANÁLISIS DE RESULTADO: ÉXITO. La matriz pudo ser descompuesta directamente.")
        
        L_frac = to_fraction_matrix(L)
        U_frac = to_fraction_matrix(U)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Matriz L (Triangular Inferior):**")
            st.dataframe(L_frac.astype(str), use_container_width=True)
        with col2:
            st.write("**Matriz U (Triangular Superior):**")
            st.dataframe(U_frac.astype(str), use_container_width=True)
        
        st.subheader("Verificación (L x U)")
        verificacion = L @ U
        st.dataframe(verificacion, use_container_width=True)
    else:
        st.warning("ANÁLISIS DE RESULTADO: FALLO EN EL MÉTODO DIRECTO. Se ha encontrado un cero en la diagonal (pivote).")
        
        with st.spinner('Paso 2: Procediendo con el método general A = P.T*L*U...'):
            time.sleep(1)
            P, L_piv, U_piv = factorizacion_plu(A)

        # Se añade una comprobación para informar al usuario si la matriz es singular
        is_singular = np.any(np.abs(np.diag(U_piv)) < 1e-9)

        if is_singular:
            st.warning("ANÁLISIS DE RESULTADO: ÉXITO PARCIAL CON PIVOTEO. La matriz es singular.")
            st.markdown("La factorización se ha completado, pero la matriz `U` contiene ceros en su diagonal. Esto indica que la matriz original no tiene inversa y su determinante es cero.")
        else:
            st.info("ANÁLISIS DE RESULTADO: ÉXITO CON PIVOTEO. La matriz fue descompuesta utilizando pivoteo de filas.")
        
        P_frac = to_fraction_matrix(P)
        L_piv_frac = to_fraction_matrix(L_piv)
        U_piv_frac = to_fraction_matrix(U_piv)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Matriz de Permutación P:**")
            st.dataframe(P_frac.astype(str), use_container_width=True)
        with col2:
            st.write("**Matriz L (Triangular Inferior):**")
            st.dataframe(L_piv_frac.astype(str), use_container_width=True)
        with col3:
            st.write("**Matriz U (Triangular Superior):**")
            st.dataframe(U_piv_frac.astype(str), use_container_width=True)

        st.subheader("Verificación (P.T x L x U)")
        verificacion = P.T @ L_piv @ U_piv
        st.dataframe(verificacion, use_container_width=True)

