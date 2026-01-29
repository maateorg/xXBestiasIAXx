"""
Agente Autónomo de Colocación de Transformadores
Implementación usando LlamaIndex + Google Gemini con patrón ReAct
"""

import os
import json
import re
import time
import asyncio
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
from datetime import datetime

# Importaciones de LlamaIndex y Google
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step

# ==================== CONFIGURACIÓN ====================


def initialize_llm(api_key: Optional[str] = None):
    """Inicializa el LLM con API key desde variable de entorno o parámetro"""
    
    llm = GoogleGenAI(
        model="models/gemini-2.5-flash",  # Modelo más reciente
        api_key=api_key,
        temperature=0.1,
        request_timeout=600.0
    )
    Settings.llm = llm
    return llm


# ==================== EVENTOS DEL WORKFLOW ====================

class LoopEvent(Event):
    """Evento para volver a iniciar el ciclo de pensamiento"""
    pass


class ThoughtEvent(Event):
    """Evento con el pensamiento del agente"""
    thought: str
    state: Dict[str, Any]


class ActionEvent(Event):
    """Evento con la acción decidida"""
    thought: str
    action: str
    parameters: Dict[str, Any]
    state: Dict[str, Any]


class ObservationEvent(Event):
    """Evento con el resultado de una acción"""
    thought: str
    action: str
    result: str
    result_data: Optional[Dict[str, Any]]
    state: Dict[str, Any]


# ==================== CLASES DE SOPORTE ====================

class ToolResult:
    """Resultado de la ejecución de una herramienta"""
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data = data
    
    def __str__(self):
        return f"{'✓' if self.success else '✗'} {self.message}"
    
    def to_dict(self):
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data
        }


@dataclass
class AgentState:
    """Estado del agente en un momento dado"""
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


# ==================== HERRAMIENTAS DEL AGENTE ====================

class TransformerTools:
    """Herramientas disponibles para el agente"""
    
    def __init__(self, grid: List[List[str]], n_transformers: int):
        self.grid = [list(row) for row in grid]
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.n_transformers = n_transformers
        self.transformers_placed = 0
        self.history = []
        self.legend = {
            'X': 'Casa',
            'O': 'Hospital',
            'T': 'Industria',
            'E': 'Estación eléctrica',
            'C': 'Transformador',
            '-': 'Vacío'
        }
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Obtiene los vecinos de una celda (8-conectado)"""
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
        # Dentro de límites
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        
        # Celda vacía
        if self.grid[row][col] != '-':
            return False
        
        neighbors = self.get_neighbors(row, col)
        
        # Debe tener al menos una casa vecina
        if not any(self.grid[nr][nc] == 'X' for nr, nc in neighbors):
            return False
        
        # No puede estar junto a estación eléctrica
        if any(self.grid[nr][nc] == 'E' for nr, nc in neighbors):
            return False
        
        return True
    
    def get_cells_within_radius(self, row: int, col: int, radius: int) -> List[Tuple[int, int]]:
        """Obtiene todas las celdas dentro de un radio Manhattan"""
        cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if abs(r - row) + abs(c - col) <= radius:
                    cells.append((r, c))
        return cells
    
    def count_transformers_near_industry(self, industry_pos: Tuple[int, int]) -> int:
        """Cuenta transformadores en radio 3 de una industria"""
        ir, ic = industry_pos
        cells = self.get_cells_within_radius(ir, ic, 3)
        return sum(1 for r, c in cells if self.grid[r][c] == 'C')
    
    def find_industries(self) -> List[Tuple[int, int]]:
        """Encuentra todas las industrias en el mapa"""
        return [(r, c) for r in range(self.rows) 
                for c in range(self.cols) 
                if self.grid[r][c] == 'T']
    
    def calculate_position_score(self, row: int, col: int) -> float:
        """Calcula un score heurístico para una posición"""
        if not self.is_valid_position(row, col):
            return -1000.0
        
        score = 0.0
        neighbors = self.get_neighbors(row, col)
        
        # Bonificaciones por vecinos
        score += sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'O') * 10.0  # Hospitales
        score += sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'T') * 5.0   # Industrias
        score += sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'X') * 2.0   # Casas
        
        # Bonificación por industrias desatendidas
        for ind in self.find_industries():
            dist = abs(row - ind[0]) + abs(col - ind[1])
            if dist <= 3:
                current_count = self.count_transformers_near_industry(ind)
                if current_count < 2:
                    # Mayor bonificación si la industria está más desatendida
                    score += (3 - dist) * (3 - current_count) * 3.0
        
        return score
    
    # ========== HERRAMIENTAS DISPONIBLES PARA EL AGENTE ==========
    
    def analyze_position(self, row: int, col: int) -> ToolResult:
        """Analiza una posición específica"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return ToolResult(False, f"Posición ({row}, {col}) fuera de límites")
        
        is_valid = self.is_valid_position(row, col)
        score = self.calculate_position_score(row, col)
        
        neighbors_info = {}
        for nr, nc in self.get_neighbors(row, col):
            cell_type = self.grid[nr][nc]
            neighbors_info[cell_type] = neighbors_info.get(cell_type, 0) + 1
        
        return ToolResult(
            success=is_valid,
            message=f"{'Válida' if is_valid else 'Inválida'} - Score: {score:.1f}",
            data={
                "valid": is_valid,
                "score": score,
                "neighbors": neighbors_info,
                "current_cell": self.grid[row][col]
            }
        )
    
    def place_transformer(self, row: int, col: int, reason: str = "") -> ToolResult:
        """Coloca un transformador en una posición"""
        if self.transformers_placed >= self.n_transformers:
            return ToolResult(False, f"Límite alcanzado ({self.n_transformers} transformadores)")
        
        if not self.is_valid_position(row, col):
            return ToolResult(False, f"Posición ({row}, {col}) inválida para transformador")
        
        self.grid[row][col] = 'C'
        self.transformers_placed += 1
        self.history.append({
            "action": "place",
            "position": (row, col),
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        return ToolResult(
            success=True,
            message=f"Transformador colocado en ({row}, {col})",
            data={
                "total_placed": self.transformers_placed,
                "remaining": self.n_transformers - self.transformers_placed,
                "reason": reason
            }
        )
    
    def check_constraints(self) -> ToolResult:
        """Verifica que se cumplan las restricciones de industrias"""
        violations = []
        for ind in self.find_industries():
            count = self.count_transformers_near_industry(ind)
            if count < 2:
                violations.append({
                    "position": ind,
                    "current_transformers": count,
                    "required": 2
                })
        
        if not violations:
            return ToolResult(
                success=True,
                message="✓ Todas las restricciones cumplidas",
                data={"violations": 0}
            )
        
        return ToolResult(
            success=False,
            message=f"✗ {len(violations)} industrias sin cobertura adecuada",
            data={"violations": violations}
        )
    
    def find_best_candidates(self, top_n: int = 5) -> ToolResult:
        """Encuentra las mejores posiciones candidatas"""
        candidates = []
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_valid_position(r, c):
                    neighbors = self.get_neighbors(r, c)
                    candidates.append({
                        "position": (r, c),
                        "score": self.calculate_position_score(r, c),
                        "neighbors": {
                            "X": sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'X'),
                            "O": sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'O'),
                            "T": sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'T')
                        }
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:top_n]
        
        return ToolResult(
            success=True,
            message=f"Encontrados {len(candidates)} candidatos (mostrando top {top_n})",
            data={"candidates": top_candidates, "total": len(candidates)}
        )
    
    def get_industry_coverage(self) -> ToolResult:
        """Obtiene la cobertura actual de cada industria"""
        coverage = []
        for ind in self.find_industries():
            count = self.count_transformers_near_industry(ind)
            coverage.append({
                "position": ind,
                "transformers": count,
                "satisfied": count >= 2
            })
        
        total_industries = len(coverage)
        satisfied = sum(1 for c in coverage if c["satisfied"])
        
        return ToolResult(
            success=True,
            message=f"Cobertura: {satisfied}/{total_industries} industrias satisfechas",
            data={"coverage": coverage, "satisfaction_rate": satisfied/total_industries if total_industries > 0 else 0}
        )
    
    # ========== UTILIDADES ==========
    
    def get_state(self) -> AgentState:
        """Obtiene el estado actual del agente"""
        return AgentState(
            grid=self.grid,
            transformers_placed=self.transformers_placed,
            target_transformers=self.n_transformers,
            iteration=len(self.history),
            history=self.history
        )
    
    def format_grid(self) -> str:
        """Formatea el grid como string"""
        return '\n'.join([''.join(row) for row in self.grid])
    
    def get_legend(self) -> str:
        """Retorna la leyenda del mapa"""
        return '\n'.join([f"{k}: {v}" for k, v in self.legend.items()])


# ==================== WORKFLOW REACT ====================

class TransformerAgentWorkflow(Workflow):
    """Workflow principal del agente usando patrón ReAct"""
    
    def __init__(self, tools: TransformerTools, max_iterations: int = 30, verbose: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools = tools
        self.max_iterations = max_iterations
        self.iteration = 0
        self.llm = Settings.llm
        self.verbose = verbose
        self.recent_actions = deque(maxlen=5)
        self.best_candidates = None
    
    def _log(self, message: str):
        """Log condicional basado en verbose"""
        if self.verbose:
            print(message)
    
    # ========== STEP 1: THINK ==========
    @step
    async def think(self, ev: Union[StartEvent, LoopEvent]) -> ThoughtEvent:
        """Paso de razonamiento: el agente piensa qué hacer"""
        self.iteration += 1
        self._log(f"\n{'='*60}\n🔄 ITERACIÓN {self.iteration}/{self.max_iterations}\n{'='*60}")
        
        state = self.tools.get_state().to_dict()
        
        # Si ya cumplimos el objetivo, verificar restricciones
        if self.tools.transformers_placed >= self.tools.n_transformers:
            self._log("🎯 Objetivo de transformadores alcanzado. Verificando restricciones...")
            return ThoughtEvent(
                thought='{"thought": "Objetivo alcanzado", "action": "check_constraints", "parameters": {}}',
                state=state
            )
        
        # Construir prompt mejorado
        prompt = self._build_prompt(state)
        
        self._log("🧠 Analizando situación y decidiendo acción...")
        
        try:
            response = await self.llm.acomplete(prompt)
            response_text = response.text.strip()
            self._log(f"💭 Decisión: {response_text[:150]}...")
            
            return ThoughtEvent(thought=response_text, state=state)
        
        except Exception as e:
            self._log(f"⚠️ Error en generación: {e}")
            # Fallback: usar find_best_candidates
            return ThoughtEvent(
                thought='{"thought": "Error en razonamiento, usando búsqueda de candidatos", "action": "find_best_candidates", "parameters": {"top_n": 5}}',
                state=state
            )
    
    def _build_prompt(self, state: Dict[str, Any]) -> str:
        """Construye un prompt estructurado y detallado para el LLM"""
        industries = self.tools.find_industries()
        industry_status = []
        for ind in industries:
            count = self.tools.count_transformers_near_industry(ind)
            industry_status.append(f"  - Industria en {ind}: {count}/2 transformadores")
        
        industry_info = '\n'.join(industry_status) if industry_status else "  - No hay industrias"
        
        prompt = f"""Eres un agente experto en colocación óptima de transformadores eléctricos.

**ESTADO ACTUAL:**
```
{self.tools.format_grid()}
```

**LEYENDA:**
- X: Casa (requiere transformador vecino)
- O: Hospital (alta prioridad)
- T: Industria (requiere 2+ transformadores en radio 3)
- E: Estación eléctrica (no puede tener transformador vecino)
- C: Transformador (ya colocado)
- -: Espacio vacío

**PROGRESO:**
- Transformadores colocados: {state['transformers_placed']}/{state['target_transformers']}
- Restantes: {state['remaining']}

**ESTADO INDUSTRIAS:**
{industry_info}

**RESTRICCIONES:**
1. El transformador debe ir en celda vacía (-)
2. Debe tener al menos una casa (X) vecina
3. NO puede estar junto a estación eléctrica (E)
4. Cada industria (T) necesita 2+ transformadores en radio Manhattan 3

**HERRAMIENTAS DISPONIBLES:**
- `analyze_position(row, col)`: Analiza una posición específica
- `place_transformer(row, col, reason)`: Coloca transformador con justificación
- `check_constraints()`: Verifica restricciones de industrias
- `find_best_candidates(top_n)`: Encuentra mejores posiciones (default top_n=5)
- `get_industry_coverage()`: Revisa cobertura de industrias

**TU TAREA:**
Responde ÚNICAMENTE con un objeto JSON válido con este formato exacto:
{{
    "thought": "tu razonamiento paso a paso sobre qué hacer y por qué",
    "action": "nombre_de_la_herramienta",
    "parameters": {{parámetros de la herramienta}}
}}

**ESTRATEGIA RECOMENDADA:**
1. Si quedan transformadores: usar find_best_candidates para ver opciones
2. Analizar las mejores opciones con analyze_position
3. Colocar con place_transformer incluyendo reason detallado
4. Priorizar industrias desatendidas y hospitales
5. Al terminar: usar check_constraints

Responde ahora con el JSON:"""
        
        
        
        strategy_hint = ""
        if state['transformers_placed'] == 0:
            strategy_hint = """
            **PRIMERA ACCIÓN OBLIGATORIA:**
            Usa find_best_candidates(top_n=5) para identificar las mejores posiciones disponibles.
            """
        elif state['transformers_placed'] < state['target_transformers']:
            strategy_hint = """
        **SIGUIENTE PASO:**
        1. Si ya conoces candidatos: usa analyze_position en la mejor opción
        2. Si analizaste una posición válida: usa place_transformer AHORA
        3. Si no tienes candidatos: usa find_best_candidates
        """
    
            prompt = f"""{prompt}

        {strategy_hint}

        Responde ahora con el JSON:"""
        return prompt
    
    # ========== STEP 2: ACT ==========
    @step
    async def act(self, ev: ThoughtEvent) -> ObservationEvent:
        """Paso de acción: ejecuta la decisión del agente"""
        decision = self._parse_llm_response(ev.thought)
        
        if not decision or 'action' not in decision:
            self._log("⚠️ Error al parsear respuesta. Usando acción por defecto.")
            decision = {
                "thought": "Fallback por error de parseo",
                "action": "find_best_candidates",
                "parameters": {"top_n": 5}
            }
        
        action = decision.get('action', 'find_best_candidates')
        params = decision.get('parameters', {})
        thought = decision.get('thought', 'Sin pensamiento')

        # Detección de loops
        if len(self.recent_actions) >= 3:
            if all(a == action for a in list(self.recent_actions)[-3:]):
                self._log(f"⚠️ Loop detectado con '{action}'. Forzando colocación.")
                if self.best_candidates and self.best_candidates.get('candidates'):
                    best_pos = self.best_candidates['candidates'][0]['position']
                    action = 'place_transformer'
                    params = {
                        'row': best_pos[0],
                        'col': best_pos[1],
                        'reason': 'Colocación forzada por loop detection'
                    }
        
        self.recent_actions.append(action)
        
        self._log(f"🎯 ACCIÓN: {action}")
        self._log(f"📋 Parámetros: {params}")
        
        # Ejecutar herramienta
        tool_result = self._execute_tool(action, params)
        
        # Guardar candidatos para uso futuro
        if action == 'find_best_candidates' and tool_result.success:
            self.best_candidates = tool_result.data
        
        self._log(f"📤 RESULTADO: {tool_result.message}")
        if tool_result.data and self.verbose:
            self._log(f"   Datos: {json.dumps(tool_result.data, indent=2)[:200]}...")
        
        return ObservationEvent(
            thought=thought,
            action=action,
            result=tool_result.message,
            result_data=tool_result.data,
            state=ev.state
        )
    
    # ========== STEP 3: OBSERVE ==========
    @step
    async def observe(self, ev: ObservationEvent) -> Union[LoopEvent, StopEvent]:
        """Paso de observación: decide si continuar o terminar"""
        
        # Condición de éxito: todos los transformadores colocados
        if self.tools.transformers_placed >= self.tools.n_transformers:
            self._log("\n✅ Objetivo de colocación cumplido. Verificando restricciones finales...")
            check_result = self.tools.check_constraints()
            
            success = check_result.success
            result = {
                "grid": self.tools.grid,
                "success": success,
                "transformers_placed": self.tools.transformers_placed,
                "constraints_satisfied": success,
                "iterations": self.iteration,
                "history": self.tools.history,
                "final_check": check_result.to_dict()
            }
            
            if success:
                self._log("🎉 ¡Todas las restricciones cumplidas!")
            else:
                self._log(f"⚠️ Restricciones no cumplidas: {check_result.message}")
            
            return StopEvent(result=result)
        
        # Condición de parada: máximo de iteraciones
        if self.iteration >= self.max_iterations:
            self._log(f"\n⏱️ Máximo de iteraciones alcanzado ({self.max_iterations})")
            return StopEvent(result={
                "grid": self.tools.grid,
                "success": False,
                "reason": "max_iterations",
                "transformers_placed": self.tools.transformers_placed,
                "target": self.tools.n_transformers,
                "iterations": self.iteration
            })
        
        # Continuar el ciclo
        self._log("🔄 Continuando con siguiente iteración...")
        return LoopEvent()
    
    # ========== UTILIDADES ==========
    
    def _parse_llm_response(self, text: str) -> Optional[Dict]:
        """Parse robusto de la respuesta del LLM"""
        try:
            # Limpiar markdown
            clean = text.replace("```json", "").replace("```", "").strip()
            
            # Buscar JSON en el texto
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                json_str = match.group()
                return json.loads(json_str)
            
            # Intentar parsear directamente
            return json.loads(clean)
        
        except json.JSONDecodeError as e:
            self._log(f"⚠️ Error JSON: {e}")
            return None
        except Exception as e:
            self._log(f"⚠️ Error inesperado en parseo: {e}")
            return None
    
    def _execute_tool(self, action: str, params: Dict) -> ToolResult:
        """Ejecuta una herramienta del agente"""
        tool_mapping = {
            "analyze_position": self.tools.analyze_position,
            "place_transformer": self.tools.place_transformer,
            "check_constraints": self.tools.check_constraints,
            "find_best_candidates": self.tools.find_best_candidates,
            "get_industry_coverage": self.tools.get_industry_coverage,
        }
        
        if action not in tool_mapping:
            return ToolResult(False, f"Herramienta desconocida: {action}")
        
        try:
            return tool_mapping[action](**params)
        except TypeError as e:
            return ToolResult(False, f"Parámetros incorrectos para {action}: {e}")
        except Exception as e:
            return ToolResult(False, f"Error ejecutando {action}: {e}")


# ==================== UTILIDADES ====================

def print_grid(grid: List[List[str]], title: str = "MAPA"):
    """Imprime el grid de forma visual"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)
    for i, row in enumerate(grid):
        print(f"{i:2d} | {''.join(row)}")
    print('='*50)


def parse_grid_file(filepath: str) -> List[List[str]]:
    """Lee y parsea un archivo de mapa"""
    with open(filepath, 'r') as file:
        return [list(line.strip()) for line in file if line.strip()]


def save_solution(grid: List[List[str]], filepath: str = "salidas/solucion.txt"):
    """Guarda la solución en un archivo"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for row in grid:
            f.write(''.join(row) + '\n')
    print(f"💾 Solución guardada en: {filepath}")


# ==================== MAIN ====================

async def main():
    """Función principal"""
    print("🤖 AGENTE AUTÓNOMO DE COLOCACIÓN DE TRANSFORMADORES")
    print("="*60)
    
    # Configurar API key
    try:        
        initialize_llm("AIzaSyDqNqEy0KilnH-hj3WUKdZ71I1YM55drSA")
        print("✓ LLM inicializado correctamente\n")
    
    except Exception as e:
        print(f"❌ Error configurando LLM: {e}")
        return
    
    # Solicitar nombre del archivo (sin ruta)
    filename = input("📁 Nombre del archivo del mapa: ").strip()
    
    # Construir ruta de entrada
    input_dir = "entradas"
    map_file = os.path.join(input_dir, filename)
    
    if not os.path.exists(map_file):
        print(f"❌ El archivo '{filename}' no existe en la carpeta '{input_dir}/'")
        return
    
    # Solicitar número de transformadores
    try:
        n_transformers = int(input("🔢 Número de transformadores: ").strip())
        if n_transformers <= 0:
            print("❌ El número debe ser positivo")
            return
    except ValueError:
        print("❌ Debe ingresar un número válido")
        return
    
    # Cargar mapa
    try:
        grid = parse_grid_file(map_file)
        print(f"\n✓ Mapa cargado: {len(grid)}x{len(grid[0]) if grid else 0}")
    except Exception as e:
        print(f"❌ Error leyendo mapa: {e}")
        return
    
    print_grid(grid, "MAPA INICIAL")
    
    # Crear herramientas y workflow
    tools = TransformerTools(grid, n_transformers)
    workflow = TransformerAgentWorkflow(
        tools=tools,
        max_iterations=30,
        verbose=True,
        timeout=120
    )
    
    print(f"\n🚀 Iniciando agente (máx {30} iteraciones)...\n")
    
    # Ejecutar workflow
    start_time = datetime.now()
    
    try:
        result = await workflow.run()
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
        return
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("🏁 EJECUCIÓN FINALIZADA")
    print("="*60)
    print(f"⏱️  Tiempo: {duration:.2f} segundos")
    print(f"🔄 Iteraciones: {workflow.iteration}")
    
    if isinstance(result, dict) and 'grid' in result:
        print_grid(result['grid'], "MAPA FINAL")
        
        print(f"\n📊 RESULTADOS:")
        print(f"   Transformadores colocados: {result.get('transformers_placed', '?')}/{n_transformers}")
        print(f"   Éxito: {'✅ SÍ' if result.get('success') else '❌ NO'}")
        
        if result.get('success'):
            print(f"   Estado: ✓ Todas las restricciones cumplidas")
        else:
            print(f"   Estado: ✗ {result.get('reason', 'Restricciones no cumplidas')}")
        
        # Guardar solución con el mismo nombre en la carpeta salidas
        try:
            output_file = os.path.join("salidas", filename)
            save_solution(result['grid'], output_file)
        except Exception as e:
            print(f"⚠️  Error guardando solución: {e}")
        
        # Guardar log detallado con nombre relacionado
        try:
            base_name = os.path.splitext(filename)[0]
            log_file = os.path.join("salidas", f"{base_name}_log.json")
            os.makedirs("salidas", exist_ok=True)
            with open(log_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"📝 Log guardado en: {log_file}")
        except Exception as e:
            print(f"⚠️  Error guardando log: {e}")


if __name__ == "__main__":
    asyncio.run(main())
