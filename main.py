"""
Agente Autónomo de Colocación de Transformadores - VERSIÓN HÍBRIDA
Implementación usando LlamaIndex + Google Gemini con patrón ReAct
HÍBRIDO: Heurística genera candidatos + LLM elige el mejor (RÁPIDO + INTELIGENTE)
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


class APIKeyRotator:
    def __init__(self, api_keys):
        if not api_keys:
            raise ValueError("Lista de API keys vacía")
        self.api_keys = api_keys
        self.index = 0

    @property
    def current(self):
        return self.api_keys[self.index]

    def rotate(self):
        self.index = (self.index + 1) % len(self.api_keys)
        print(f"🔑 Rotando API key → usando #{self.index + 1}/{len(self.api_keys)}")



def initialize_llm(rotator: APIKeyRotator):
    llm = GoogleGenAI(
        model="models/gemini-2.5-flash",
        api_key=rotator.current,
        temperature=0.0,
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
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        
        if self.grid[row][col] != '-':
            return False
        
        neighbors = self.get_neighbors(row, col)
        
        if not any(self.grid[nr][nc] == 'X' for nr, nc in neighbors):
            return False
        
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
        """Calcula un score heurístico mejorado para una posición"""
        if not self.is_valid_position(row, col):
            return -1000.0
        
        score = 0.0
        neighbors = self.get_neighbors(row, col)
        
        # Bonificaciones por vecinos importantes
        hospital_neighbors = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'O')
        industry_neighbors = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'T')
        house_neighbors = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'X')
        
        score += hospital_neighbors * 15.0  # Hospitales MUY importantes
        score += industry_neighbors * 12.0   # Industrias directas muy importantes
        score += house_neighbors * 3.0       # Casas importantes
        
        # NUEVA LÓGICA: Priorizar industrias más desatendidas
        industries = self.find_industries()
        for ind in industries:
            dist = abs(row - ind[0]) + abs(col - ind[1])
            if dist <= 3:
                current_count = self.count_transformers_near_industry(ind)
                # Bonificación MASIVA para industrias con 0 o 1 transformador
                if current_count == 0:
                    score += (4 - dist) * 25.0  # Crítico: 100 puntos si está muy cerca
                elif current_count == 1:
                    score += (4 - dist) * 15.0  # Muy importante: completar a 2
        
        return score
    
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
            message=f"Transformador #{self.transformers_placed} colocado en ({row}, {col})",
            data={
                "total_placed": self.transformers_placed,
                "remaining": self.n_transformers - self.transformers_placed,
                "reason": reason
            }
        )
    
    def place_multiple_transformers(self, positions: List[Dict[str, Any]]) -> ToolResult:
        """Coloca múltiples transformadores a la vez"""
        placed = []
        failed = []
        
        for pos_data in positions:
            row = pos_data.get('row')
            col = pos_data.get('col')
            reason = pos_data.get('reason', '')
            
            if row is None or col is None:
                failed.append({"position": "unknown", "reason": "Coordenadas faltantes"})
                continue
            
            result = self.place_transformer(row, col, reason)
            if result.success:
                placed.append((row, col))
            else:
                failed.append({"position": (row, col), "reason": result.message})
        
        if not placed:
            return ToolResult(
                success=False,
                message=f"No se pudo colocar ningún transformador. Errores: {len(failed)}",
                data={"placed": 0, "failed": failed}
            )
        
        return ToolResult(
            success=True,
            message=f"✓ {len(placed)} transformadores colocados. Fallos: {len(failed)}",
            data={
                "placed": len(placed),
                "positions": placed,
                "failed": failed,
                "total_placed": self.transformers_placed,
                "remaining": self.n_transformers - self.transformers_placed
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
    
    def find_best_candidates(self, top_n: int = 10) -> ToolResult:
        """Encuentra las mejores posiciones candidatas con mejor scoring"""
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
                            "T": sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'T')
                        }
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:top_n]
        
        return ToolResult(
            success=True,
            message=f"Encontrados {len(candidates)} candidatos (top {top_n} disponibles)",
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


# ==================== WORKFLOW REACT MEJORADO ====================

class TransformerAgentWorkflow(Workflow):
    """Workflow principal del agente usando patrón ReAct optimizado"""
    
    def __init__(self, tools: TransformerTools, key_rotator, max_iterations: int = 200, verbose: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools = tools
        self.key_rotator = key_rotator
        self.max_iterations = max_iterations
        self.iteration = 0
        self.llm = Settings.llm
        self.verbose = verbose
        self.recent_actions = deque(maxlen=5)
    
    def _log(self, message: str):
        """Log condicional basado en verbose"""
        if self.verbose:
            print(message)

    async def _with_key_rotation(self, coro):
        """Ejecuta una coroutine con rotación automática de API key si hay error de cuota"""
        try:
            return await coro
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                self._log("🚨 429 / quota detectado → rotando API key (SIN contar iteración)")
                self.key_rotator.rotate()
                initialize_llm(self.key_rotator)
                self.llm = Settings.llm  # Actualizar referencia
                # Decrementar iteración porque vamos a reintentar
                self.iteration -= 1
                return await coro
            raise
    
    @step
    async def think(self, ev: Union[StartEvent, LoopEvent]) -> ThoughtEvent:
        """Paso de razonamiento: HÍBRIDO (heurística + LLM)"""
        self.iteration += 1
        self._log(f"\n{'='*60}\n🔄 ITERACIÓN {self.iteration}/{self.max_iterations}\n{'='*60}")
        
        state = self.tools.get_state().to_dict()
        
        # Objetivo alcanzado - verificar restricciones
        if self.tools.transformers_placed >= self.tools.n_transformers:
            self._log("🎯 Objetivo alcanzado. Verificando restricciones...")
            return ThoughtEvent(
                thought='{"thought": "Verificando restricciones finales", "action": "check_constraints", "parameters": {}}',
                state=state
            )
        
        remaining = self.tools.n_transformers - self.tools.transformers_placed
        
        # PASO 1: Usar heurística para generar CANDIDATOS (RÁPIDO)
        # Si quedan muchos transformadores, generar más candidatos
        if remaining >= 500:
            num_candidates = min(30, remaining // 20)  # Más candidatos para batches grandes
        elif remaining >= 100:
            num_candidates = min(20, remaining // 10)
        elif remaining >= 20:
            num_candidates = min(15, remaining // 3)
        else:
            num_candidates = min(10, max(5, remaining // 2))
        
        self._log(f"⚡ Generando {num_candidates} candidatos con heurística...")
        candidates_result = self.tools.find_best_candidates(top_n=num_candidates)
        
        if not candidates_result.success or not candidates_result.data.get('candidates'):
            self._log("⚠️ No hay candidatos disponibles")
            return ThoughtEvent(
                thought='{"thought": "No hay posiciones válidas", "action": "check_constraints", "parameters": {}}',
                state=state
            )
        
        candidates = candidates_result.data['candidates']
        coverage = self.tools.get_industry_coverage()
        
        # Decidir si colocar múltiples transformadores - BATCHING ESCALADO
        batch_size = 1
        if remaining >= 500:
            batch_size = 20  # Batch grande para 500+
        elif remaining >= 100:
            batch_size = 10  # Batch medio para 100+
        elif remaining >= 20:
            batch_size = min(5, remaining // 4)  # Hasta 5 para 20+
        elif remaining >= 10:
            batch_size = min(3, remaining // 3)  # Hasta 3 para 10+
        
        # PASO 2: LLM elige entre los candidatos (RÁPIDO porque son pocos)
        candidates_str = "\n".join([
            f"{i+1}. Posición ({c['position'][0]}, {c['position'][1]}) - Score: {c['score']:.1f}\n"
            f"   Vecinos: {c['neighbors']['X']} casas, {c['neighbors']['O']} hospitales, {c['neighbors']['T']} industrias"
            for i, c in enumerate(candidates[:num_candidates])
        ])
        
        if batch_size > 1:
            prompt = f"""Eres un agente experto en colocar transformadores.

**ESTADO:**
- Colocados: {self.tools.transformers_placed}/{self.tools.n_transformers}
- Restantes: {remaining}

**COBERTURA INDUSTRIAS:**
{coverage.message}

**TOP {num_candidates} CANDIDATOS (pre-filtrados por heurística):**
{candidates_str}

**RESTRICCIÓN CRÍTICA:**
Cada industria (T) DEBE tener ≥2 transformadores en radio Manhattan ≤3.

**DECIDE:**
Quedan MUCHOS transformadores ({remaining}). Elige {batch_size} candidatos de la lista para colocar MÚLTIPLES transformadores a la vez.
Prioriza industrias con <2 transformadores y evita posiciones redundantes.

**RESPONDE JSON:**
{{
  "thought": "[Tu decisión breve]",
  "action": "place_multiple_transformers",
  "parameters": {{
    "positions": [
      {{"row": <fila>, "col": <columna>, "reason": "[razón]"}},
      {{"row": <fila>, "col": <columna>, "reason": "[razón]"}},
      ...
    ]
  }}
}}

SOLO JSON, sin más texto."""
        else:
            prompt = f"""Eres un agente experto en colocar transformadores.

**ESTADO:**
- Colocados: {self.tools.transformers_placed}/{self.tools.n_transformers}
- Restantes: {remaining}

**COBERTURA INDUSTRIAS:**
{coverage.message}

**TOP {num_candidates} CANDIDATOS (pre-filtrados por heurística):**
{candidates_str}

**RESTRICCIÓN CRÍTICA:**
Cada industria (T) DEBE tener ≥2 transformadores en radio Manhattan ≤3.

**DECIDE:**
Elige UNO de los candidatos. Prioriza industrias con <2 transformadores.

**RESPONDE JSON:**
{{
  "thought": "[Tu decisión breve]",
  "action": "place_transformer",
  "parameters": {{
    "row": <fila del candidato elegido>,
    "col": <columna del candidato elegido>,
    "reason": "[Por qué este candidato]"
  }}
}}

SOLO JSON, sin más texto."""
        
        try:
            self._log(f"🤖 LLM eligiendo {batch_size} transformador{'es' if batch_size > 1 else ''}...")
            response = await self._with_key_rotation(
                asyncio.to_thread(self.llm.complete, prompt)
            )
            thought = response.text.strip()
            self._log(f"💬 LLM decidió: {thought[:100]}...")
            
            return ThoughtEvent(
                thought=thought,
                state=state
            )
        except Exception as e:
            self._log(f"❌ Error LLM: {e} → usando mejor candidato")
            # Fallback: usar el mejor candidato heurístico
            best = candidates[0]
            return ThoughtEvent(
                thought=f'{{"thought": "Fallback: mejor candidato heurístico", "action": "place_transformer", "parameters": {{"row": {best["position"][0]}, "col": {best["position"][1]}, "reason": "Fallback heurístico"}}}}',
                state=state
            )
    
    @step
    async def act(self, ev: ThoughtEvent) -> ObservationEvent:
        """Paso de acción: ejecuta la decisión de la LLM"""
        decision = self._parse_llm_response(ev.thought)
        
        if not decision or 'action' not in decision:
            self._log("⚠️ Error al parsear respuesta de LLM.")
            # Buscar cualquier posición válida como fallback
            for r in range(self.tools.rows):
                for c in range(self.tools.cols):
                    if self.tools.is_valid_position(r, c):
                        decision = {
                            "thought": "Fallback por error de parseo",
                            "action": "place_transformer",
                            "parameters": {"row": r, "col": c, "reason": "Posición de fallback"}
                        }
                        break
                if decision and 'action' in decision:
                    break
            
            if not decision or 'action' not in decision:
                # Si no hay posiciones válidas, verificar restricciones
                decision = {
                    "thought": "No hay posiciones válidas",
                    "action": "check_constraints",
                    "parameters": {}
                }
        
        action = decision.get('action', 'check_constraints')
        params = decision.get('parameters', {})
        thought = decision.get('thought', 'Sin pensamiento')
        
        self.recent_actions.append(action)
        
        self._log(f"🎯 ACCIÓN: {action}")
        if params:
            self._log(f"📋 Parámetros: {params}")
        
        # Ejecutar herramienta
        tool_result = await self._with_key_rotation(
            asyncio.to_thread(self._execute_tool, action, params)
        )
        
        self._log(f"📤 RESULTADO: {tool_result.message}")
        
        return ObservationEvent(
            thought=thought,
            action=action,
            result=tool_result.message,
            result_data=tool_result.data,
            state=ev.state
        )
    
    @step
    async def observe(self, ev: ObservationEvent) -> Union[LoopEvent, StopEvent]:
        """Paso de observación: decide si continuar o terminar"""
        
        # Condición de éxito
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
        
        # Condición de parada por iteraciones
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
        
        # Continuar
        return LoopEvent()
    
    def _parse_llm_response(self, text: str) -> Optional[Dict]:
        """Parse robusto de la respuesta del LLM"""
        try:
            clean = text.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                json_str = match.group()
                return json.loads(json_str)
            return json.loads(clean)
        except Exception as e:
            self._log(f"⚠️ Error en parseo: {e}")
            return None
    
    def _execute_tool(self, action: str, params: Dict) -> ToolResult:
        """Ejecuta una herramienta del agente"""
        tool_mapping = {
            "place_transformer": self.tools.place_transformer,
            "place_multiple_transformers": self.tools.place_multiple_transformers,
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
    print("🤖 AGENTE AUTÓNOMO DE COLOCACIÓN DE TRANSFORMADORES v3.5")
    print("="*60)
    print("⚡ Versión HÍBRIDA: Heurística + LLM (rápido e inteligente)")
    print("="*60)
    
    # Configurar API key
    try:        
        api_keys = [
            
        ]

        key_rotator = APIKeyRotator(api_keys)
        initialize_llm(key_rotator)
        print("✓ LLM inicializado correctamente\n")
    except Exception as e:
        print(f"❌ Error configurando LLM: {e}")
        return
    
    # Solicitar nombre del archivo
    filename = input("📁 Nombre del archivo del mapa: ").strip()
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
        key_rotator=key_rotator,
        max_iterations=max(400, n_transformers * 3),
        verbose=True,
        timeout=600
    )
    
    print(f"\n🚀 Iniciando agente (máx {200} iteraciones)...\n")
    
    # Ejecutar workflow
    start_time = datetime.now()
    
    try:
        result = await workflow.run()
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
        import traceback
        traceback.print_exc()
        return
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("🏁 EJECUCIÓN FINALIZADA")
    print("="*60)
    print(f"⏱️  Tiempo: {duration:.2f} segundos")
    print(f"🔄 Iteraciones: {workflow.iteration}")
    print(f"⚡ Eficiencia: {(n_transformers / workflow.iteration * 100):.1f}% (colocaciones/iteración)")
    
    if isinstance(result, dict) and 'grid' in result:
        print_grid(result['grid'], "MAPA FINAL")
        
        print(f"\n📊 RESULTADOS:")
        print(f"   Transformadores colocados: {result.get('transformers_placed', '?')}/{n_transformers}")
        print(f"   Éxito: {'✅ SÍ' if result.get('success') else '❌ NO'}")
        
        if result.get('success'):
            print(f"   Estado: ✓ Todas las restricciones cumplidas")
        else:
            print(f"   Estado: ✗ {result.get('reason', 'Restricciones no cumplidas')}")
        
        # Guardar solución
        try:
            output_file = os.path.join("salidas", filename)
            save_solution(result['grid'], output_file)
        except Exception as e:
            print(f"⚠️  Error guardando solución: {e}")
        
        # Guardar log
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