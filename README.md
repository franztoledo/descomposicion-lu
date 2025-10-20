# **Calculadora de Factorización LU/PLU para Matrices Simétricas**
Aplicación web interactiva desarrollada como proyecto para el curso de Álgebra Lineal. Esta herramienta implementa y visualiza el algoritmo de factorización LU, seleccionando de forma inteligente entre el método de descomposición directa (A = LU) y el método con pivoteo parcial (A = PᵀLU) cuando es necesario.
## **Descripción**
El objetivo de este proyecto es aplicar los conceptos teóricos del álgebra lineal en una solución computacional robusta. El programa permite al usuario ingresar una matriz simétrica (ya sea de forma manual o generada aleatoriamente) y realiza el análisis para descomponerla en sus matrices L (triangular inferior) y U (triangular superior), incluyendo la matriz de permutación P si es requerida.

La interfaz, desarrollada con Streamlit, busca ofrecer una experiencia de usuario clara e intuitiva, mostrando los resultados matemáticos de forma exacta mediante fracciones.
## **Características Principales**
- **Interfaz Gráfica Interactiva:** Desarrollada con Streamlit para una fácil manipulación y visualización de datos.
- **Doble Método de Factorización:**
  - Intenta primero la **factorización A = LU** por el método de eliminación gaussiana simple.
  - Si el método simple falla (debido a un pivote cero), automáticamente procede con la **factorización A = PᵀLU** usando pivoteo parcial, garantizando una solución para cualquier matriz no singular.
- **Entrada de Datos Flexible:** Permite generar matrices simétricas aleatorias o ingresar valores manualmente en una grilla intuitiva.
- **Resultados Exactos:** Muestra las matrices resultantes P, L y U utilizando fracciones para una precisión matemática total, evitando errores de redondeo.
- **Verificación Automática:** Calcula el producto de las matrices resultantes (L @ U o P.T @ L @ U) para comprobar que la factorización reconstruye la matriz original A.
- **Manejo de Casos Singulares:** Detecta y notifica al usuario si una matriz es singular.
## **Tecnologías Utilizadas**
- **Python 3:** Lenguaje principal de desarrollo.
- **Streamlit:** Para la creación de la interfaz web interactiva.
- **NumPy:** Para el manejo eficiente de matrices y operaciones numéricas.
## **Instalación y Ejecución Local**
Para ejecutar este proyecto en tu máquina local, sigue estos pasos:

1. **Clona el repositorio:**\
   git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)\
   cd tu-repositorio
1. **Crea un entorno virtual (recomendado):**\
   python -m venv venv\
   source venv/bin/activate  # En Windows: venv\Scripts\activate
1. Instala las dependencias:\
   Crea un archivo requirements.txt con el siguiente contenido:\
   streamlit\
   numpy\
   time\
   fractions\
\
   Luego, instala las librerías:\
   pip install -r requirements.txt
1. **Ejecuta la aplicación:**\
   streamlit run app.py\
\
   La aplicación se abrirá automáticamente en tu navegador web.
## **Autores**
- *(Tu Nombre Completo)*
- *(Nombre Completo del Integrante 2)*
- *(etc...)*
