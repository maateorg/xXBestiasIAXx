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

# --- CONFIGURACIÓN DE GEMINI ---
llm = GoogleGenAI(
    model="models/gemini-1.5-flash",  # Usamos 1.5 Flash por estabilidad
    api_key="PON_AQUI_TU_NUEVA_API_KEY",  # <--- ⚠️ PON TU NUEVA CLAVE AQUI
    temperature=0.1,
    request_timeout=60.0
)

Settings.llm = llm


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
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data = data
    
    def __str__(self):
        return f"{'✓' if self.success else '✗'} {self.message}"

@dataclass
class AgentState:
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
    def __init__(self, grid: List[List[str]], n_transformers: int):
        self.grid = [list(row) for row in grid]
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.n_transformers = n_transformers
        self.transformers_placed = 0
        self.history = []
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors
    
    def is_valid_position(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols): return False
        if self.grid[row][col] != '-': return False
        neighbors = self.get_neighbors(row, col)
        if not any(self.grid[nr][nc] == 'X' for nr, nc in neighbors): return False
        if any(self.grid[nr][nc] == 'E' for nr, nc in neighbors): return False
        return True
    
    def get_cells_within_radius(self, row: int, col: int, radius: int) -> List[Tuple[int, int]]:
        cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if abs(r - row) + abs(c - col) <= radius:
                    cells.append((r, c))
        return cells
    
    def count_transformers_near_industry(self, industry_pos: Tuple[int, int]) -> int:
        ir, ic = industry_pos
        cells = self.get_cells_within_radius(ir, ic, 3)
        return sum(1 for r, c in cells if self.grid[r][c] == 'C')
    
    def find_industries(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] == 'T']
    
    def calculate_position_score(self, row: int, col: int) -> float:
        score = 0.0
        neighbors = self.get_neighbors(row, col)
        score += sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'O') * 10.0
        score += sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'T') * 5.0
        score += sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'X') * 2.0
        
        for ind in self.find_industries():
            dist = abs(row - ind[0]) + abs(col - ind[1])
            if dist <= 3 and self.count_transformers_near_industry(ind) < 2:
                score += (3 - dist) * 3.0
        return score
    
    def analyze_position(self, row: int, col: int) -> ToolResult:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return ToolResult(False, "Fuera de límites")
        return ToolResult(True, "Análisis completado", {"score": self.calculate_position_score(row, col)})
    
    def place_transformer(self, row: int, col: int, reason: str = "") -> ToolResult:
        if self.transformers_placed >= self.n_transformers:
            return ToolResult(False, "Límite alcanzado")
        if not self.is_valid_position(row, col):
            return ToolResult(False, "Posición inválida")
            
        self.grid[row][col] = 'C'
        self.transformers_placed += 1
        self.history.append({"action": "place", "pos": (row, col), "reason": reason})
        return ToolResult(True, f"Colocado en ({row}, {col})", {"total": self.transformers_placed})
    
    def check_constraints(self) -> ToolResult:
        violations = []
        for ind in self.find_industries():
            count = self.count_transformers_near_industry(ind)
            if count < 2:
                violations.append({"pos": ind, "current": count})
        
        if not violations: return ToolResult(True, "Restricciones OK", {})
        return ToolResult(False, f"{len(violations)} violaciones", {"violations": violations})
    
    def find_best_candidates(self, top_n: int = 5) -> ToolResult:
        candidates = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_valid_position(r, c):
                    candidates.append({
                        "position": (r, c),
                        "score": self.calculate_position_score(r, c),
                        "neighbors": {
                            "X": sum(1 for nr, nc in self.get_neighbors(r, c) if self.grid[nr][nc] == 'X'),
                            "O": sum(1 for nr, nc in self.get_neighbors(r, c) if self.grid[nr][nc] == 'O'),
                            "T": sum(1 for nr, nc in self.get_neighbors(r, c) if self.grid[nr][nc] == 'T')
                        }
                    })
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return ToolResult(True, f"Top {top_n} encontrados", {"candidates": candidates[:top_n]})
    
    def get_industry_coverage(self) -> ToolResult:
        coverage = []
        for ind in self.find_industries():
            coverage.append({
                "pos": ind,
                "count": self.count_transformers_near_industry(ind)
            })
        return ToolResult(True, "Cobertura industrias", {"data": coverage})

    def get_state(self) -> AgentState:
        return AgentState(self.grid, self.transformers_placed, self.n_transformers, len(self.history), self.history)
    
    def format_grid(self) -> str:
        return '\n'.join([''.join(row) for row in self.grid])


# ==================== WORKFLOW REACT ====================

class TransformerAgentWorkflow(Workflow):
    def __init__(self, tools: TransformerTools, max_iterations: int = 30, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools = tools
        self.max_iterations = max_iterations
        self.iteration = 0
        self.llm = Settings.llm
    
    # 1. THINK: Acepta StartEvent O LoopEvent
    @step
    async def think(self, ev: Union[StartEvent, LoopEvent]) -> ThoughtEvent:
        self.iteration += 1
        print(f"\n{'='*50}\n🔄 ITERACIÓN {self.iteration}\n{'='*50}")
        
        state = self.tools.get_state().to_dict()
        
        if self.tools.transformers_placed >= self.tools.n_transformers:
            return ThoughtEvent(thought="Objetivo alcanzado. check_constraints", state=state)
        
        prompt = f"""
ESTADO:
{self.tools.format_grid()}
Colocados: {self.tools.transformers_placed}/{self.tools.n_transformers}

HERRAMIENTAS: analyze_position(row, col), place_transformer(row, col, reason), check_constraints(), find_best_candidates(top_n), get_industry_coverage()

Responde JSON: {{"thought": "...", "action": "...", "parameters": {{...}}}}
"""
        print("🧠 PENSANDO...")
        response = await self.llm.acomplete(prompt)
        print(f"💭 RESPUESTA: {response.text[:100]}...") # Print parcial para no saturar
        return ThoughtEvent(thought=response.text, state=state)
    
    # 2. ACT
    @step
    async def act(self, ev: ThoughtEvent) -> ObservationEvent:
        decision = self._parse_llm_response(ev.thought)
        
        if not decision or 'action' not in decision:
            # Si el agente dice que terminó en texto plano, verificamos
            if "Objetivo alcanzado" in ev.thought:
                decision = {"action": "check_constraints", "parameters": {}}
            else:
                print("⚠️ Fallo parseo JSON. Usando backup.")
                decision = {"thought": "Backup", "action": "find_best_candidates", "parameters": {"top_n": 5}}
        
        action = decision.get('action')
        print(f"🎯 ACCIÓN: {action}")
        
        tool_res = self._execute_tool(action, decision.get('parameters', {}))
        print(f"📤 RESULTADO: {tool_res}")
        
        return ObservationEvent(thought=decision.get('thought', ''), action=action, result=str(tool_res), result_data=tool_res.data, state=ev.state)
    
    # 3. OBSERVE: Devuelve LoopEvent (para ir a Think) o StopEvent
    @step
    async def observe(self, ev: ObservationEvent) -> Union[LoopEvent, StopEvent]:
        # Condición de parada: Éxito
        if self.tools.transformers_placed >= self.tools.n_transformers:
            print("\n✅ Objetivo cumplido. Verificando...")
            check = self.tools.check_constraints()
            return StopEvent(result={"grid": self.tools.grid, "success": check.success})
        
        # Condición de parada: Iteraciones máximas
        if self.iteration >= self.max_iterations:
            print("⚠️ Máximo iteraciones.")
            return StopEvent(result={"grid": self.tools.grid, "success": False})
        
        # SI NO HEMOS TERMINADO -> VOLVER A PENSAR (LoopEvent)
        return LoopEvent()

    def _parse_llm_response(self, text: str) -> Optional[Dict]:
        try:
            clean = text.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            return json.loads(match.group() if match else clean)
        except:
            return None

    def _execute_tool(self, action: str, params: Dict) -> ToolResult:
        mapping = {
            "analyze_position": self.tools.analyze_position,
            "place_transformer": self.tools.place_transformer,
            "check_constraints": self.tools.check_constraints,
            "find_best_candidates": self.tools.find_best_candidates,
            "get_industry_coverage": self.tools.get_industry_coverage,
        }
        if action not in mapping: return ToolResult(False, "Herramienta desconocida")
        try: return mapping[action](**params)
        except Exception as e: return ToolResult(False, str(e))

# ==================== MAIN ====================

def parse_grid(f):
    with open(f) as file: return [list(l.strip()) for l in file if l.strip()]

async def main():
    print("🤖 AGENTE INICIADO")
    f_path = input("Archivo mapa: ").strip()
    if not os.path.exists(f_path): return print("No existe archivo")
    
    n_trans = int(input("N Transformadores: "))
    grid = parse_grid(f_path)
    
    print_grid(grid, "INICIO")
    
    tools = TransformerTools(grid, n_trans)
    wf = TransformerAgentWorkflow(tools, timeout=120)
    
    res = await wf.run()
    
    print("\n🏁 FINALIZADO")
    if isinstance(res, dict) and 'grid' in res:
        print_grid(res['grid'], "FINAL")
        os.makedirs("salidas", exist_ok=True)
        with open("salidas/solucion.txt", "w") as f:
            for r in res['grid']: f.write("".join(r)+"\n")
        print("Guardado en salidas/solucion.txt")

if __name__ == "__main__":
    asyncio.run(main())