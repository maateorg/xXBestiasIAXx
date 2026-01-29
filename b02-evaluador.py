from collections import deque
import os

# --- VARIABLES GLOBALES ---
NOMBRE_MAPA = "1.txt"
CANTIDAD_C = 2  # Cantidad esperada de transformadores 'C'

# --- 1. Leer mapas original y generado ---
ruta_entrada = f"entradas/{NOMBRE_MAPA}"
ruta_salida = f"salidas/{NOMBRE_MAPA}"

with open(ruta_entrada, "r") as f:
    mapa_original = [list(line.strip()) for line in f.readlines()]

with open(ruta_salida, "r") as f:
    mapa_generado = [list(line.strip()) for line in f.readlines()]

# --- 2. Verificar que O, X, E y T están en las mismas posiciones ---
# Comprueba que no se hayan eliminado ninguna 'E', 'T' ni 'O' del mapa original
def verificar_posiciones(mapa_orig, mapa_gen):
    filas_orig = len(mapa_orig)
    columnas_orig = len(mapa_orig[0]) if filas_orig > 0 else 0
    filas_gen = len(mapa_gen)
    columnas_gen = len(mapa_gen[0]) if filas_gen > 0 else 0
    
    # Verificar dimensiones
    if filas_orig != filas_gen or columnas_orig != columnas_gen:
        print(f"Error: Los mapas tienen dimensiones diferentes")
        return False
    
    # Verificar posiciones de O, X, E y T (elementos que no deben cambiar)
    elementos_fijos = ['O', 'X', 'E', 'T']
    for i in range(filas_orig):
        for j in range(columnas_orig):
            # Si en el original hay O, X, E o T, debe ser igual en el generado
            if mapa_orig[i][j] in elementos_fijos:
                if mapa_gen[i][j] != mapa_orig[i][j]:
                    print(f"Error en posición ({i},{j}): Original '{mapa_orig[i][j]}' vs Generado '{mapa_gen[i][j]}'")
                    return False
            # Si en el generado hay O, X, E o T, debe ser igual en el original
            if mapa_gen[i][j] in elementos_fijos:
                if mapa_orig[i][j] != mapa_gen[i][j]:
                    print(f"Error en posición ({i},{j}): Original '{mapa_orig[i][j]}' vs Generado '{mapa_gen[i][j]}'")
                    return False
    
    return True

# --- 3. Verificar cantidad de transformadores 'C' ---
def verificar_cantidad_transformadores(mapa, cantidad_esperada):
    transformadores = 0
    for fila in mapa:
        for celda in fila:
            if celda == 'C':
                transformadores += 1
    
    if transformadores != cantidad_esperada:
        print(f"Error: Se esperaban {cantidad_esperada} transformadores 'C', pero se encontraron {transformadores}")
        return False
    
    return True

# --- 3.1 Verificar que todos los 'C' se colocaron donde antes había '-' ---
def verificar_c_en_espacios_blancos(mapa_orig, mapa_gen):
    filas = len(mapa_orig)
    cols = len(mapa_orig[0]) if filas > 0 else 0
    
    for i in range(filas):
        for j in range(cols):
            if mapa_gen[i][j] == 'C':
                if mapa_orig[i][j] != '-':
                    print(f"Error: El transformador 'C' en posición ({i},{j}) no está en un espacio en blanco '-' (original: '{mapa_orig[i][j]}')")
                    return False
    
    return True

# --- 3.2 Verificar que cada 'C' tiene al menos una 'X' en las 8 celdas contiguas ---
def verificar_c_tiene_x_adyacente(mapa):
    filas = len(mapa)
    cols = len(mapa[0]) if filas > 0 else 0
    
    # Direcciones de las 8 casillas adyacentes (incluye diagonales)
    direcciones = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for i in range(filas):
        for j in range(cols):
            if mapa[i][j] == 'C':
                tiene_x_adyacente = False
                
                # Verificar las 8 casillas adyacentes
                for di, dj in direcciones:
                    ni, nj = i + di, j + dj
                    # Verificar que esté dentro de los límites
                    if 0 <= ni < filas and 0 <= nj < cols:
                        if mapa[ni][nj] == 'X':
                            tiene_x_adyacente = True
                            break
                
                if not tiene_x_adyacente:
                    print(f"Error: El transformador 'C' en posición ({i},{j}) no tiene ninguna 'X' (zona residencial) en las 8 celdas contiguas")
                    return False
    
    return True

# --- 3.3 Verificar que cada 'C' NO tiene ninguna 'E' en las 8 celdas contiguas ---
def verificar_c_sin_e_adyacente(mapa):
    filas = len(mapa)
    cols = len(mapa[0]) if filas > 0 else 0
    
    # Direcciones de las 8 casillas adyacentes (incluye diagonales)
    direcciones = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for i in range(filas):
        for j in range(cols):
            if mapa[i][j] == 'C':
                # Verificar las 8 casillas adyacentes
                for di, dj in direcciones:
                    ni, nj = i + di, j + dj
                    # Verificar que esté dentro de los límites
                    if 0 <= ni < filas and 0 <= nj < cols:
                        if mapa[ni][nj] == 'E':
                            print(f"Error: El transformador 'C' en posición ({i},{j}) tiene una 'E' (Subestación) adyacente en posición ({ni},{nj})")
                            return False
    
    return True

# --- 3.4 Verificar que todas las 'T' tienen al menos 2 'C' a distancia de 3 celdas ---
def verificar_t_tiene_cercanos(mapa):
    filas = len(mapa)
    cols = len(mapa[0]) if filas > 0 else 0
    
    torres = [(i, j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'T']
    transformadores = [(i, j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'C']
    
    for torre in torres:
        # Contar transformadores dentro de radio 3
        transformadores_cercanos = 0
        for transformador in transformadores:
            distancia = distancia_minima(mapa, torre, [transformador])
            if distancia <= 3:
                transformadores_cercanos += 1
        
        if transformadores_cercanos < 2:
            print(f"Error: La torre 'T' en posición {torre} tiene solo {transformadores_cercanos} transformadores 'C' dentro de un radio de 3 celdas (mínimo requerido: 2)")
            return False
    
    return True

# --- 4. Función para calcular distancia mínima ---
def distancia_minima(mapa, start, targets):
    filas, cols = len(mapa), len(mapa[0])
    visitado = [[False]*cols for _ in range(filas)]
    queue = deque([(start[0], start[1], 0)])
    
    while queue:
        i, j, d = queue.popleft()
        if (i, j) in targets:
            return d
        if visitado[i][j]:
            continue
        visitado[i][j] = True
        
        # Moverse a las 4 direcciones (arriba, abajo, izquierda, derecha)
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i + di, j + dj
            # Solo verificar que esté dentro de los límites
            # TODAS las casillas son transitables (-, X, O, C, T, E)
            if 0 <= ni < filas and 0 <= nj < cols:
                queue.append((ni, nj, d+1))
    
    return float('inf')

# --- 5. Función distancia_total ---
# Calcula la suma de distancias mínimas:
# - De cada 'O' (casa/hospital) a la 'C' (transformador) más cercana
# - De cada 'T' (torre/industria) a la 'C' (transformador) más cercana
# NOTA: Las 'X' (zonas residenciales) solo son transitables, NO se cuentan en el cálculo
def distancia_total(mapa):
    filas, cols = len(mapa), len(mapa[0])
    casas = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'O']
    torres = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'T']
    transformadores = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'C']
    
    total = 0
    # Distancia de cada casa O a la C más cercana
    for casa in casas:
        total += distancia_minima(mapa, casa, transformadores)
    # Distancia de cada torre T a la C más cercana
    for torre in torres:
        total += distancia_minima(mapa, torre, transformadores)
    return total

# --- 6. Ejecutar verificaciones y cálculo ---
if (verificar_posiciones(mapa_original, mapa_generado) and 
    verificar_cantidad_transformadores(mapa_generado, CANTIDAD_C) and
    verificar_c_en_espacios_blancos(mapa_original, mapa_generado) and
    verificar_c_tiene_x_adyacente(mapa_generado) and
    verificar_c_sin_e_adyacente(mapa_generado) and
    verificar_t_tiene_cercanos(mapa_generado)):
    total_pasos = distancia_total(mapa_generado)
    print(total_pasos)
else:
    print("Error: El mapa generado no cumple con los requisitos")