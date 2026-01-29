import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# --- VARIABLES GLOBALES ---
NOMBRE_MAPA = "1.txt"

# Rutas de los mapas
ruta_entrada = f"entradas/{NOMBRE_MAPA}"
ruta_salida = f"salidas/{NOMBRE_MAPA}"

# Rutas de las imágenes
imagenes = {
    'X': 'images/residencial.png',
    'O': 'images/hospital.png',
    'T': 'images/intrustria.png',
    'E': 'images/subestacion.png',
    'C': 'images/transformador.png',
    '-': None  # No se pinta nada
}

# --- Leer mapas ---
with open(ruta_entrada, "r") as f:
    mapa_entrada = [list(line.strip()) for line in f.readlines()]

with open(ruta_salida, "r") as f:
    mapa_salida = [list(line.strip()) for line in f.readlines()]

# Normalizar ancho de las filas
ancho_entrada = max(len(fila) for fila in mapa_entrada)
ancho_salida = max(len(fila) for fila in mapa_salida)
ancho = max(ancho_entrada, ancho_salida)

# Convertir lista a string, aplicar ljust, y convertir de vuelta a lista
mapa_entrada = [list(''.join(fila).ljust(ancho)) for fila in mapa_entrada]
mapa_salida = [list(''.join(fila).ljust(ancho)) for fila in mapa_salida]

filas = len(mapa_entrada)
cols = ancho

# --- Cargar imágenes y rotarlas 180 grados ---
imagenes_cargadas = {}
for simbolo, ruta in imagenes.items():
    if ruta and os.path.exists(ruta):
        img = mpimg.imread(ruta)
        # Rotar 180 grados (rotar 90 dos veces o usar flip)
        img_rotada = np.rot90(img, 2)  # rot90 dos veces = 180 grados
        imagenes_cargadas[simbolo] = img_rotada
    else:
        imagenes_cargadas[simbolo] = None

# --- Función para dibujar mapa ---
def dibujar_mapa(ax, mapa, titulo):
    # Dibujar fondo blanco para todo el mapa
    fondo = plt.Rectangle((-0.5, -0.5), cols, filas, 
                         fill=True, facecolor='white', zorder=0)
    ax.add_patch(fondo)
    
    for i in range(filas):
        for j in range(cols):
            simbolo = mapa[i][j]
            if simbolo in imagenes_cargadas and imagenes_cargadas[simbolo] is not None:
                # Dibujar imagen centrada en la celda
                img = imagenes_cargadas[simbolo]
                # Ajustar tamaño de la imagen (85% del tamaño de la celda)
                extent = [j - 0.425, j + 0.425, i - 0.425, i + 0.425]
                ax.imshow(img, extent=extent, aspect='auto', zorder=2)
            # Si es '-', no se dibuja nada (espacio en blanco)
            # Dibujar cuadrícula sutil para mejor visualización
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                               fill=False, edgecolor='lightgray', 
                               linewidth=0.3, zorder=1)
            ax.add_patch(rect)

# Crear carpeta para imágenes generadas
os.makedirs("imagenes_generadas", exist_ok=True)

# --- Crear y guardar mapa de entrada ---
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.set_xlim(-0.5, cols - 0.5)
ax1.set_ylim(-0.5, filas - 0.5)
ax1.set_aspect('equal')
ax1.invert_yaxis()  # Invertir para que (0,0) esté arriba-izquierda
ax1.axis('off')

dibujar_mapa(ax1, mapa_entrada, 'Entrada')

plt.tight_layout()

ruta_imagen_entrada = f"imagenes_generadas/{NOMBRE_MAPA.replace('.txt', '_entrada.png')}"
plt.savefig(ruta_imagen_entrada, dpi=150, bbox_inches='tight', pad_inches=0)
print(f"✅ Mapa de entrada guardado en: {ruta_imagen_entrada}")
plt.close(fig1)

# --- Crear y guardar mapa de salida ---
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
ax2.set_xlim(-0.5, cols - 0.5)
ax2.set_ylim(-0.5, filas - 0.5)
ax2.set_aspect('equal')
ax2.invert_yaxis()
ax2.axis('off')

dibujar_mapa(ax2, mapa_salida, 'Salida')

plt.tight_layout()

ruta_imagen_salida = f"imagenes_generadas/{NOMBRE_MAPA.replace('.txt', '_salida.png')}"
plt.savefig(ruta_imagen_salida, dpi=150, bbox_inches='tight', pad_inches=0)
print(f"✅ Mapa de salida guardado en: {ruta_imagen_salida}")
plt.close(fig2)

print("\n✅ Visualizaciones completadas!")

