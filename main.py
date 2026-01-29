# transformer_placement_agent.py
import os
from typing import List, Tuple, Dict, Optional, Any
import google.generativeai as genai
import json
import re
from dataclasses import dataclass
from enum import Enum

# Configuración de API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=GOOGLE_API_KEY)


class ToolResult:
    """Resultado de la ejecución de una herramienta"""
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data = data
    
    def __str__(self):
        return f"{'✓' if self.success else '✗'} {self.message}"


@dataclass
class AgentState:
    """Estado actual del agente"""
    grid: List[List[str]]
    transformers_placed: int
    target_transformers: int
    iteration: int
    history: List[Dict[str, Any]]
    
    def to_dict(self):
        return {
            "grid": [''.join(row) for row in self.grid],
            "transformers_placed": self.transformers_placed,
            "target_transformers": self.target_transformers,
            "remaining": self.target_transformers - self.transformers_placed
        }


class TransformerPlacementAgent:
    """
    Agente autónomo para colocación de transformadores usando ReAct pattern.
    El agente puede:
    - Razonar sobre el estado actual
    - Decidir qué herramienta usar
    - Ejecutar acciones
    - Aprender de sus errores
    - Replanificar cuando sea necesario
    """
    
    def __init__(self, grid: List[List[str]], n_transformers: int):
        self.initial_grid = [list(row) for row in grid]
        self.grid = [list(row) for row in grid]
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.n_transformers = n_transformers
        self.transformers_placed = 0
        self.history = []
        
        # Configurar el modelo LLM
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Definir herramientas disponibles
        self.tools = {
            "analyze_position": self.tool_analyze_position,
            "place_transformer": self.tool_place_transformer,
            "check_constraints": self.tool_check_constraints,
            "find_best_candidates": self.tool_find_best_candidates,
            "get_industry_coverage": self.tool_get_industry_coverage,
            "undo_last_placement": self.tool_undo_last_placement,
            "finish": self.tool_finish
        }
        
        print(f"\n🤖 AGENTE AUTÓNOMO INICIALIZADO")
        print(f"📊 Mapa: {self.rows}x{self.cols}")
        print(f"🔧 Objetivo: Colocar {self.n_transformers} transformadores")
        print(f"🛠️  Herramientas disponibles: {list(self.tools.keys())}\n")
    
    # ==================== HERRAMIENTAS DEL AGENTE ====================
    
    def tool_analyze_position(self, row: int, col: int) -> ToolResult:
        """Analiza una posición específica del grid"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return ToolResult(False, f"Posición ({row}, {col}) fuera de límites")
        
        cell = self.grid[row][col]
        neighbors = self.get_neighbors(row, col)
        
        analysis = {
            "position": (row, col),
            "current_value": cell,
            "is_available": cell == '-',
            "is_valid": self.is_valid_position(row, col),
            "neighbors": {
                "residential_X": sum(1 for r, c in neighbors if self.grid[r][c] == 'X'),
                "hospital_O": sum(1 for r, c in neighbors if self.grid[r][c] == 'O'),
                "industry_T": sum(1 for r, c in neighbors if self.grid[r][c] == 'T'),
                "substation_E": sum(1 for r, c in neighbors if self.grid[r][c] == 'E'),
            },
            "priority_score": self.calculate_position_score(row, col) if cell == '-' else 0
        }
        
        return ToolResult(True, f"Análisis completado para ({row}, {col})", analysis)
    
    def tool_place_transformer(self, row: int, col: int, reason: str = "") -> ToolResult:
        """Intenta colocar un transformador en la posición especificada"""
        if self.transformers_placed >= self.n_transformers:
            return ToolResult(False, "Ya se alcanzó el número objetivo de transformadores")
        
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return ToolResult(False, f"Posición ({row}, {col}) fuera de límites")
        
        if not self.is_valid_position(row, col):
            reasons = []
            if self.grid[row][col] != '-':
                reasons.append(f"celda no disponible (contiene '{self.grid[row][col]}')")
            
            neighbors = self.get_neighbors(row, col)
            has_residential = any(self.grid[r][c] == 'X' for r, c in neighbors)
            if not has_residential:
                reasons.append("no hay zonas residenciales 'X' adyacentes")
            
            has_substation = any(self.grid[r][c] == 'E' for r, c in neighbors)
            if has_substation:
                reasons.append("hay subestación 'E' adyacente")
            
            return ToolResult(False, f"Posición inválida: {', '.join(reasons)}")
        
        # Colocar el transformador
        self.grid[row][col] = 'C'
        self.transformers_placed += 1
        
        # Guardar en historial
        self.history.append({
            "action": "place",
            "position": (row, col),
            "reason": reason,
            "state": [list(row) for row in self.grid]
        })
        
        return ToolResult(
            True, 
            f"Transformador #{self.transformers_placed} colocado en ({row}, {col}). Razón: {reason}",
            {"position": (row, col), "total_placed": self.transformers_placed}
        )
    
    def tool_check_constraints(self) -> ToolResult:
        """Verifica todas las restricciones del problema"""
        industries = self.find_industries()
        violations = []
        satisfied = []
        
        for ind_r, ind_c in industries:
            count = self.count_transformers_near_industry((ind_r, ind_c))
            if count < 2:
                violations.append({
                    "position": (ind_r, ind_c),
                    "transformers": count,
                    "needed": 2 - count
                })
            else:
                satisfied.append({
                    "position": (ind_r, ind_c),
                    "transformers": count
                })
        
        data = {
            "total_industries": len(industries),
            "satisfied": satisfied,
            "violations": violations,
            "all_satisfied": len(violations) == 0
        }
        
        if len(violations) == 0:
            return ToolResult(True, "✓ Todas las restricciones satisfechas", data)
        else:
            return ToolResult(
                False, 
                f"✗ {len(violations)} industria(s) sin cobertura suficiente",
                data
            )
    
    def tool_find_best_candidates(self, top_n: int = 5) -> ToolResult:
        """Encuentra las mejores posiciones candidatas para colocar transformadores"""
        candidates = []
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_valid_position(r, c):
                    score = self.calculate_position_score(r, c)
                    neighbors = self.get_neighbors(r, c)
                    
                    candidates.append({
                        "position": (r, c),
                        "score": score,
                        "neighbors": {
                            "X": sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'X'),
                            "O": sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'O'),
                            "T": sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'T'),
                        }
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:top_n]
        
        return ToolResult(
            True,
            f"Encontradas {len(candidates)} posiciones válidas, mostrando top {top_n}",
            {"candidates": top_candidates, "total": len(candidates)}
        )
    
    def tool_get_industry_coverage(self) -> ToolResult:
        """Obtiene el estado de cobertura de todas las industrias"""
        industries = self.find_industries()
        coverage = []
        
        for ind_r, ind_c in industries:
            count = self.count_transformers_near_industry((ind_r, ind_c))
            nearby_positions = self.get_cells_within_radius(ind_r, ind_c, 3)
            available = sum(1 for r, c in nearby_positions if self.is_valid_position(r, c))
            
            coverage.append({
                "position": (ind_r, ind_c),
                "current_transformers": count,
                "needs_more": count < 2,
                "available_positions": available
            })
        
        return ToolResult(
            True,
            f"Estado de cobertura de {len(industries)} industrias",
            {"industries": coverage}
        )
    
    def tool_undo_last_placement(self) -> ToolResult:
        """Deshace la última colocación de transformador"""
        if not self.history:
            return ToolResult(False, "No hay acciones para deshacer")
        
        last_action = self.history.pop()
        if last_action["action"] == "place":
            row, col = last_action["position"]
            self.grid[row][col] = '-'
            self.transformers_placed -= 1
            
            return ToolResult(
                True,
                f"Deshecha colocación en ({row}, {col})",
                {"position": (row, col)}
            )
        
        return ToolResult(False, "La última acción no fue una colocación")
    
    def tool_finish(self) -> ToolResult:
        """Marca el proceso como terminado"""
        return ToolResult(True, "Agente finalizado", {"final_grid": self.grid})
    
    # ==================== FUNCIONES AUXILIARES ====================
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Obtiene las 8 posiciones adyacentes"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Verifica si una posición es válida para colocar un transformador"""
        if self.grid[row][col] != '-':
            return False
        
        neighbors = self.get_neighbors(row, col)
        has_residential = any(self.grid[nr][nc] == 'X' for nr, nc in neighbors)
        if not has_residential:
            return False
        
        has_substation = any(self.grid[nr][nc] == 'E' for nr, nc in neighbors)
        if has_substation:
            return False
        
        return True
    
    def get_cells_within_radius(self, row: int, col: int, radius: int) -> List[Tuple[int, int]]:
        """Obtiene todas las celdas dentro de un radio dado (distancia Manhattan)"""
        cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if abs(r - row) + abs(c - col) <= radius:
                    cells.append((r, c))
        return cells
    
    def count_transformers_near_industry(self, industry_pos: Tuple[int, int]) -> int:
        """Cuenta transformadores cerca de una industria"""
        ir, ic = industry_pos
        cells_in_radius = self.get_cells_within_radius(ir, ic, 3)
        count = sum(1 for r, c in cells_in_radius if self.grid[r][c] == 'C')
        return count
    
    def find_industries(self) -> List[Tuple[int, int]]:
        """Encuentra todas las industrias en el mapa"""
        industries = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 'T':
                    industries.append((r, c))
        return industries
    
    def calculate_position_score(self, row: int, col: int) -> float:
        """Calcula la puntuación de una posición"""
        score = 0.0
        neighbors = self.get_neighbors(row, col)
        
        hospitals_nearby = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'O')
        score += hospitals_nearby * 10.0
        
        industries_nearby = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'T')
        score += industries_nearby * 5.0
        
        residential_nearby = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'X')
        score += residential_nearby * 2.0
        
        industries = self.find_industries()
        for ind_r, ind_c in industries:
            distance = abs(row - ind_r) + abs(col - ind_c)
            if distance <= 3:
                current_coverage = self.count_transformers_near_industry((ind_r, ind_c))
                if current_coverage < 2:
                    score += (3 - distance) * 3.0
        
        return score
    
    def get_current_state(self) -> AgentState:
        """Obtiene el estado actual del agente"""
        return AgentState(
            grid=self.grid,
            transformers_placed=self.transformers_placed,
            target_transformers=self.n_transformers,
            iteration=len(self.history),
            history=self.history
        )
    
    def format_grid(self) -> str:
        """Formatea el grid para mostrar al LLM"""
        return '\n'.join([''.join(row) for row in self.grid])
    
    # ==================== CORE DEL AGENTE (ReAct Loop) ====================
    
    def create_system_prompt(self) -> str:
        """Crea el prompt del sistema con las reglas y herramientas"""
        return f"""Eres un agente experto en optimización de redes eléctricas. Tu tarea es colocar transformadores en un grid siguiendo estrictas restricciones.

REGLAS DEL PROBLEMA:
1. Los transformadores 'C' solo se colocan en espacios disponibles '-'
2. Cada transformador DEBE tener al menos una zona residencial 'X' en sus 8 posiciones adyacentes
3. NO puede haber subestaciones 'E' en las 8 posiciones adyacentes al transformador
4. Cada industria 'T' DEBE tener al menos 2 transformadores 'C' dentro de un radio de 3 pasos (distancia Manhattan)

SÍMBOLOS:
- '-': Terreno disponible
- 'X': Zonas residenciales (prioridad normal)
- 'O': Hospitales (prioridad ALTA)
- 'T': Industrias (prioridad media, requieren 2+ transformadores cercanos)
- 'E': Subestaciones existentes (bloquean posiciones adyacentes)
- 'C': Transformadores ya colocados

HERRAMIENTAS DISPONIBLES:
{json.dumps(list(self.tools.keys()), indent=2)}

PROCESO (ReAct pattern):
1. THOUGHT: Piensa en el estado actual y qué hacer a continuación
2. ACTION: Decide qué herramienta usar y con qué parámetros
3. OBSERVATION: Observa el resultado de la herramienta
4. Repite hasta completar la tarea

FORMATO DE RESPUESTA:
Debes responder SIEMPRE en este formato JSON:
{{
  "thought": "tu razonamiento sobre qué hacer",
  "action": "nombre_de_herramienta",
  "parameters": {{"param1": valor1, "param2": valor2}}
}}

ESTRATEGIA RECOMENDADA:
1. Usa 'find_best_candidates' para identificar posiciones óptimas
2. Usa 'get_industry_coverage' para priorizar industrias sin cobertura
3. Usa 'analyze_position' antes de colocar para verificar
4. Usa 'place_transformer' cuando estés seguro
5. Usa 'check_constraints' periódicamente para validar
6. Usa 'undo_last_placement' si cometiste un error
7. Usa 'finish' cuando hayas terminado

IMPORTANTE:
- Prioriza industrias que no tienen cobertura suficiente
- Considera hospitales como segunda prioridad
- Verifica restricciones antes de cada colocación
- Si una estrategia no funciona, replantea el enfoque"""

    def parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parsea la respuesta del LLM"""
        try:
            # Intentar extraer JSON del texto
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except json.JSONDecodeError:
            return None
    
    def execute_action(self, action: str, parameters: Dict[str, Any]) -> ToolResult:
        """Ejecuta una acción (herramienta)"""
        tool = self.tools.get(action)
        if not tool:
            return ToolResult(False, f"Herramienta '{action}' no encontrada")
        
        try:
            return tool(**parameters)
        except Exception as e:
            return ToolResult(False, f"Error ejecutando '{action}': {str(e)}")
    
    def solve(self, max_iterations: int = 30, verbose: bool = True) -> List[List[str]]:
        """
        Bucle principal del agente (ReAct loop)
        """
        print("🚀 INICIANDO BUCLE DE RAZONAMIENTO DEL AGENTE\n")
        
        conversation_history = []
        system_prompt = self.create_system_prompt()
        
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"🔄 ITERACIÓN {iteration + 1}/{max_iterations}")
            print(f"📊 Transformadores: {self.transformers_placed}/{self.n_transformers}")
            print(f"{'='*70}\n")
            
            # Si ya terminamos, salir
            if self.transformers_placed >= self.n_transformers:
                print("✅ Objetivo alcanzado. Verificando restricciones finales...")
                result = self.tool_check_constraints()
                print(result)
                break
            
            # Crear el estado actual para el LLM
            state = self.get_current_state()
            state_json = state.to_dict()
            
            # Construir el prompt
            user_prompt = f"""
ESTADO ACTUAL:
Grid:
{self.format_grid()}

Transformadores colocados: {self.transformers_placed}/{self.n_transformers}
Transformadores restantes: {self.n_transformers - self.transformers_placed}

¿Qué acción tomar a continuación? Piensa paso a paso y decide.
"""
            
            # Llamar al LLM
            if verbose:
                print("🧠 AGENTE PENSANDO...")
            
            try:
                chat = self.model.start_chat(history=[
                    {"role": "user", "parts": [system_prompt]},
                    {"role": "model", "parts": ["Entendido. Estoy listo para ayudarte a colocar transformadores siguiendo las reglas."]}
                ] + conversation_history)
                
                response = chat.send_message(user_prompt)
                response_text = response.text
                
                if verbose:
                    print(f"\n💭 RESPUESTA DEL AGENTE:\n{response_text}\n")
                
                # Parsear la respuesta
                decision = self.parse_llm_response(response_text)
                
                if not decision or 'action' not in decision:
                    print("⚠️  No se pudo parsear la decisión del agente. Reintentando...")
                    continue
                
                thought = decision.get('thought', 'No especificado')
                action = decision.get('action')
                parameters = decision.get('parameters', {})
                
                print(f"💡 PENSAMIENTO: {thought}")
                print(f"🎯 ACCIÓN: {action}")
                print(f"⚙️  PARÁMETROS: {json.dumps(parameters, indent=2)}")
                
                # Ejecutar la acción
                result = self.execute_action(action, parameters)
                
                print(f"\n📤 RESULTADO: {result}")
                if result.data and verbose:
                    print(f"📊 DATOS: {json.dumps(result.data, indent=2)}")
                
                # Actualizar historial de conversación
                conversation_history.extend([
                    {"role": "user", "parts": [user_prompt]},
                    {"role": "model", "parts": [response_text]}
                ])
                
                # Si el agente decidió terminar
                if action == "finish":
                    print("\n🏁 El agente decidió terminar.")
                    break
                
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                continue
        
        # Verificación final
        print(f"\n{'='*70}")
        print("📋 VERIFICACIÓN FINAL")
        print(f"{'='*70}\n")
        
        final_check = self.tool_check_constraints()
        print(final_check)
        
        if final_check.data:
            violations = final_check.data.get('violations', [])
            if violations:
                print(f"\n⚠️  INDUSTRIAS SIN COBERTURA:")
                for v in violations:
                    print(f"  • Industria en {v['position']}: {v['transformers']}/2 transformadores")
        
        print(f"\n✅ PROCESO COMPLETADO")
        print(f"📊 Transformadores colocados: {self.transformers_placed}/{self.n_transformers}")
        
        return self.grid


# ==================== FUNCIONES AUXILIARES ====================

def parse_grid_from_file(filename: str) -> List[List[str]]:
    """Lee el mapa desde un archivo"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    grid = []
    for line in lines:
        line = line.strip()
        if line:
            grid.append(list(line))
    return grid


def print_grid(grid: List[List[str]], title: str = ""):
    """Imprime el mapa de forma legible"""
    if title:
        print(f"\n{title}")
    print("=" * 50)
    for row in grid:
        print(' '.join(row))
    print("=" * 50 + "\n")


def get_user_input():
    """Solicita al usuario el archivo y el número de transformadores"""
    print("="*70)
    print("🤖 AGENTE AUTÓNOMO DE COLOCACIÓN DE TRANSFORMADORES")
    print("🧠 Powered by Google Gemini 2.0 Flash (ReAct Pattern)")
    print("="*70)
    print()
    
    # Solicitar archivo
    while True:
        filename = input("📁 Ingrese la ruta del archivo .txt con el mapa: ").strip()
        
        if not filename:
            print("❌ Debe ingresar una ruta de archivo.\n")
            continue
            
        if not os.path.exists(filename):
            print(f"❌ El archivo '{filename}' no existe.\n")
            retry = input("¿Desea intentar con otro archivo? (s/n): ").strip().lower()
            if retry != 's':
                print("Saliendo del programa...")
                exit(0)
            continue
        
        try:
            grid = parse_grid_from_file(filename)
            if not grid or not grid[0]:
                print("❌ El archivo está vacío o tiene formato inválido.\n")
                continue
            print(f"✅ Archivo cargado correctamente ({len(grid)}x{len(grid[0])})\n")
            break
        except Exception as e:
            print(f"❌ Error al leer el archivo: {str(e)}\n")
            continue
    
    # Solicitar número de transformadores
    while True:
        try:
            n_transformers_str = input("🔧 Ingrese el número de transformadores a colocar: ").strip()
            n_transformers = int(n_transformers_str)
            
            if n_transformers <= 0:
                print("❌ El número debe ser mayor a 0.\n")
                continue
            
            print(f"✅ Objetivo: {n_transformers} transformadores\n")
            break
        except ValueError:
            print("❌ Debe ingresar un número válido.\n")
            continue
    
    return grid, n_transformers


def main():
    # Obtener datos del usuario
    grid, n_transformers = get_user_input()
    
    # Mostrar mapa inicial
    print_grid(grid, "📍 MAPA INICIAL")
    
    # Crear y ejecutar el agente
    agent = TransformerPlacementAgent(grid, n_transformers)
    solution = agent.solve(max_iterations=30, verbose=True)
    
    print_grid(solution, "🗺️  SOLUCIÓN FINAL")
    
    # Guardar resultado
    output_file = "solution_agent.txt"
    with open(output_file, 'w') as f:
        for row in solution:
            f.write(''.join(row) + '\n')
    
    print(f"💾 Solución guardada en: {output_file}")
    print("\n✨ ¡Proceso completado!")


if __name__ == "__main__":
    main()