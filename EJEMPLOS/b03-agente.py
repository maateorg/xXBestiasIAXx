from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step
import asyncio
from collections import deque
from llama_index.llms.google_genai import GoogleGenAI
import os
from datetime import datetime
import time

# --- VARIABLES GLOBALES ---
NOMBRE_MAPA = "1.txt"
CANTIDAD_C = 2

local_llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=""
)

Settings.llm = local_llm

# --- Eventos simplificados ---
class AnalisisEvent(Event):
    texto: str
    analisis: str

class SolucionEvent(Event):
    texto: str
    analisis: str
    salida: str
    intentos: int

# --- Workflow simplificado (3 steps) ---
class ProblemaFlow(Workflow):
    llm = local_llm
    max_intentos = 2

    # Step 1: Analizar problema completo (reglas + restricciones + formato)
    @step
    async def analizar_problema(self, ev: StartEvent) -> AnalisisEvent:
        prompt = f"""
Analiza este problema y extrae:
1. Las reglas que deben cumplirse
2. Las restricciones existentes
3. El formato de salida esperado

Enunciado:
{ev.query}

Responde de forma estructurada y concisa.
"""
        response = await self.llm.acomplete(prompt)
        return AnalisisEvent(texto=ev.query, analisis=response.text)

    # Step 2: Generar y validar solución (con reintentos)
    @step
    async def generar_solucion(self, ev: AnalisisEvent) -> SolucionEvent:
        mejor_solucion = None
        intento = 1
        
        while intento <= self.max_intentos:
            print(f"Intento {intento} de {self.max_intentos}...")
            
            prompt = f"""
Problema original:
{ev.texto}

Análisis del problema:
{ev.analisis}

{"GENERA" if intento == 1 else f"REINTENTO {intento}: Genera una NUEVA"} solución que cumpla TODAS las reglas.

IMPORTANTE:
- Devuelve SOLO el mapa completo con los {CANTIDAD_C} transformadores 'C' colocados
- No se puede colocar una 'C' si en las 8 celdas contiguas tiene una 'E' (Subestación)
- NO incluyas explicaciones, comentarios ni texto adicional
- Mantén las mismas dimensiones del mapa original
- Cada línea debe tener exactamente el mismo formato que el original
- NO modifiques ninguna 'O', 'X', 'E' ni 'T' del mapa original
- El transformador 'C' solo se puede colocar en un espacio en blanco '-'
- El 'C' debe tener en las 8 celdas contiguas (alrededor) al menos una 'X' (zona residencial)
- Para que el mapa sea válido, cada 'T' (Industria) debe tener al menos 2 'C' dentro de un radio de 3 celdas de distancia
"""
            
            respuesta = await self.llm.acomplete(prompt)
            solucion = respuesta.text.strip()
            
            # Validación simple
            prompt_validacion = f"""
Solución:
{solucion}

Análisis del problema:
{ev.analisis}

¿Cumple TODAS las reglas? Responde SOLO: SI o NO
"""
            validacion = await self.llm.acomplete(prompt_validacion)
            es_valida = "SI" in validacion.text.upper().strip()
            
            if mejor_solucion is None or es_valida:
                mejor_solucion = solucion
            
            print(f"Validación: {validacion.text.strip()}")
            
            if es_valida:
                print(f"✅ Solución válida en intento {intento}")
                return SolucionEvent(
                    texto=ev.texto,
                    analisis=ev.analisis,
                    salida=mejor_solucion,
                    intentos=intento
                )
            
            intento += 1
        
        print(f"⚠️ No se encontró solución válida. Usando mejor intento.")
        return SolucionEvent(
            texto=ev.texto,
            analisis=ev.analisis,
            salida=mejor_solucion,
            intentos=self.max_intentos
        )

    # Step 3: Finalizar
    @step
    async def finalizar(self, ev: SolucionEvent) -> StopEvent:
        return StopEvent(result=ev.salida)



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


# --- Ejecución ---
async def main():
    # Tiempo inicial
    tiempo_inicio = time.time()
    timestamp_inicio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕐 Inicio del programa: {timestamp_inicio}\n")
    
    ruta_entrada = f"entradas/{NOMBRE_MAPA}"
    
    with open(ruta_entrada, "r") as f:
        mapa = [list(line.strip()) for line in f.readlines()]
    
    os.makedirs("salidas", exist_ok=True)
    
    mapa_str = '\n'.join([''.join(fila) for fila in mapa])
    query = f"""
Tengo un mapa:
{mapa_str}

Este mapa es una representación de una ciudad donde las 'X' son zonas residenciales, 'O' son hospitales, 'T' son edificios industriales, 
'E' son subestaciones y '-' son espacios en blanco. Quiero colocar {CANTIDAD_C} transformadores 'C' para optimizar 
la distribución eléctrica.

Reglas para colocar {CANTIDAD_C} transformadores 'C':
1. El transformador 'C' solo se puede colocar en un espacio en blanco '-'
2. El 'C' debe tener en las 8 celdas contiguas (alrededor) al menos una 'X' (zona residencial)
3. No se puede colocar una 'C' si en las 8 celdas contiguas tiene una 'E' (Subestación)
4. Para que el mapa sea válido, cada 'T' (Industria) debe tener al menos 2 'C' dentro de un radio de 3 celdas de distancia

Objetivo: Minimizar la distancia entre cada 'O' (Hospital) y la 'C' más cercana, y la distancia entre cada 'T' (torre) y la 'C' más cercana.

No modifiques el mapa inicial más que para colocar las 'C', que ocuparán el lugar de '-'. Debe conservar las mismas dimensiones y elementos.

Devuélveme SOLO el mapa completo con los {CANTIDAD_C} transformadores 'C' colocados respetando estas reglas.
"""

    flow = ProblemaFlow(timeout=2000, verbose=True)
    result = await flow.run(query=query)

    resultado_texto = result
    print("\n" + "="*50)
    print("MAPA GENERADO POR LLM:")
    print("="*50)
    print(resultado_texto)
    print("="*50 + "\n")

    # --- Limpiar el texto: eliminar marcadores de código markdown ---
    def limpiar_texto_mapa(texto):
        """Elimina marcadores de código markdown (```) y limpia el texto"""
        texto = texto.strip()
        # Eliminar marcadores de código al inicio y final
        if texto.startswith('```'):
            # Encontrar la primera línea que no es ```
            lineas = texto.split('\n')
            inicio = 0
            for i, linea in enumerate(lineas):
                if linea.strip().startswith('```'):
                    inicio = i + 1
                    break
            texto = '\n'.join(lineas[inicio:])
        
        if texto.endswith('```'):
            # Eliminar la última línea si es ```
            lineas = texto.split('\n')
            if lineas and lineas[-1].strip() == '```':
                texto = '\n'.join(lineas[:-1])
            else:
                # Si termina con ``` en la misma línea
                texto = texto.rstrip('`').rstrip()
        
        # Eliminar cualquier ``` que quede en el texto
        texto = texto.replace('```', '')
        
        return texto.strip()

    resultado_texto_limpio = limpiar_texto_mapa(resultado_texto)

    ruta_salida = f"salidas/{NOMBRE_MAPA}"
    os.makedirs("salidas", exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(resultado_texto_limpio)
    
    print(f"✅ Mapa guardado en: {ruta_salida}\n")

    ancho = max(len(line) for line in resultado_texto_limpio.splitlines())
    mapa_llm = [list(line.ljust(ancho)) for line in resultado_texto_limpio.splitlines() if line.strip()]
    print(mapa_llm)

    total_pasos = distancia_total(mapa_llm)
    print(f"📊 Suma total de pasos: {total_pasos}\n")
    
    # Tiempo final
    tiempo_fin = time.time()
    timestamp_fin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tiempo_transcurrido = tiempo_fin - tiempo_inicio
    print(f"🕐 Fin del programa: {timestamp_fin}")
    print(f"⏱️  Tiempo transcurrido: {tiempo_transcurrido:.2f} segundos")


if __name__ == "__main__":
    asyncio.run(main())