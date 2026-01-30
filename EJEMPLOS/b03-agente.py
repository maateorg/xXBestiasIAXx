"""
Agente Autónomo de Colocación de Transformadores - VERSIÓN MEJORADA v4.1
NUEVA CARACTERÍSTICA: Maneja mapas con transformadores YA colocados
Completa solo los transformadores faltantes para satisfacer restricciones
"""

import os
import json
import re
import asyncio
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
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
        model="models/gemini-2.5-flash"
        api_key=rotator.current,
        temperature=0.0,
        request_timeout=600.0
    )
    Settings.llm = llm
    return llm


# ==================== EVENTOS DEL WORKFLOW ====================

class LoopEvent(Event):
    pass


class ThoughtEvent(Event):
    thought: str
    state: Dict[str, Any]


class ObservationEvent(Event):
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
    
    def to_dict(self):
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data
        }


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


# ==================== HERRAMIENTAS MEJORADAS ====================

class TransformerTools:
    """Herramientas con soporte para transformadores pre-colocados"""
    
    def __init__(self, grid: List[List[str]], n_transformers: int):
        self.grid = [list(row) for row in grid]
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        
        # NUEVO: Contar transformadores ya colocados
        self.initial_transformers = self._count_existing_transformers()
        self.transformers_placed = self.initial_transformers
        self.n_transformers = n_transformers
        
        self.history = []
        self.legend = {
            'X': 'Casa',
            'O': 'Hospital',
            'T': 'Industria',
            'E': 'Estación eléctrica',
            'C': 'Transformador',
            '-': 'Vacío'
        }
        
        # Cache de industrias
        self.industries = self._find_industries()
        self.industry_coverage = {ind: self.count_transformers_near_industry(ind) for ind in self.industries}
    
    def _count_existing_transformers(self) -> int:
        """Cuenta transformadores que ya están en el mapa"""
        count = 0
        for row in self.grid:
            for cell in row:
                if cell == 'C':
                    count += 1
        return count
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
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
    
    def _find_industries(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.rows) 
                for c in range(self.cols) 
                if self.grid[r][c] == 'T']
    
    def get_unsatisfied_industries(self) -> List[Tuple[int, int]]:
        """PRIORIDAD: Industrias que NECESITAN transformadores"""
        unsatisfied = []
        for ind in self.industries:
            count = self.count_transformers_near_industry(ind)
            if count < 2:
                unsatisfied.append(ind)
        return unsatisfied
    
    def calculate_position_score_v3(self, row: int, col: int) -> float:
        """Score MEJORADO v3 con énfasis en industrias críticas"""
        if not self.is_valid_position(row, col):
            return -1000.0
        
        score = 0.0
        neighbors = self.get_neighbors(row, col)
        
        # Bonificaciones básicas
        hospital_neighbors = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'O')
        industry_neighbors = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'T')
        house_neighbors = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'X')
        
        score += hospital_neighbors * 10.0
        score += industry_neighbors * 8.0
        score += house_neighbors * 2.0
        
        # ESTRATEGIA CRÍTICA: Máxima prioridad a industrias con 0 o 1
        unsatisfied = self.get_unsatisfied_industries()
        
        # Calcular cuántas industrias insatisfechas cubre esta posición
        industries_covered = 0
        max_priority_score = 0.0
        
        for ind in unsatisfied:
            dist = abs(row - ind[0]) + abs(col - ind[1])
            if dist <= 3:
                industries_covered += 1
                current_count = self.count_transformers_near_industry(ind)
                
                # BONIFICACIÓN EXPONENCIAL para industrias más críticas
                if current_count == 0:
                    # Industria con 0 transformadores = MÁXIMA PRIORIDAD
                    priority = (4 - dist) * 100.0  # Hasta 400 puntos
                elif current_count == 1:
                    # Industria con 1 transformador = ALTA PRIORIDAD
                    priority = (4 - dist) * 50.0   # Hasta 200 puntos
                else:
                    priority = 0.0
                
                max_priority_score = max(max_priority_score, priority)
        
        score += max_priority_score
        
        # BONIFICACIÓN EXTRA si cubre múltiples industrias insatisfechas
        if industries_covered > 1:
            score += industries_covered * 25.0
        
        # Penalización por estar cerca de industrias ya satisfechas
        satisfied = [ind for ind in self.industries if ind not in unsatisfied]
        for ind in satisfied:
            dist = abs(row - ind[0]) + abs(col - ind[1])
            if dist <= 3:
                score -= 3.0
        
        return score
    
    def find_positions_for_critical_industries(self, top_n: int = 20) -> ToolResult:
        """Encuentra posiciones para industrias CRÍTICAS (0 o 1 transformador)"""
        unsatisfied = self.get_unsatisfied_industries()
        
        if not unsatisfied:
            return ToolResult(
                success=True,
                message="Todas las industrias están satisfechas",
                data={"candidates": [], "unsatisfied_industries": 0}
            )
        
        # Separar por criticidad
        critical = []  # 0 transformadores
        urgent = []    # 1 transformador
        
        for ind in unsatisfied:
            count = self.count_transformers_near_industry(ind)
            if count == 0:
                critical.append(ind)
            else:
                urgent.append(ind)
        
        print(f"   🔴 CRÍTICAS (0 transf): {len(critical)}")
        print(f"   🟡 URGENTES (1 transf): {len(urgent)}")
        
        # Buscar posiciones priorizando críticas primero
        all_candidates = []
        
        # Primero, críticas
        for ind in critical:
            ir, ic = ind
            for r in range(max(0, ir-3), min(self.rows, ir+4)):
                for c in range(max(0, ic-3), min(self.cols, ic+4)):
                    dist = abs(r - ir) + abs(c - ic)
                    if dist <= 3 and self.is_valid_position(r, c):
                        score = self.calculate_position_score_v3(r, c)
                        all_candidates.append({
                            "position": (r, c),
                            "score": score,
                            "distance": dist,
                            "industry": ind,
                            "current_count": 0,
                            "priority": "CRÍTICA"
                        })
        
        # Luego, urgentes
        for ind in urgent:
            ir, ic = ind
            for r in range(max(0, ir-3), min(self.rows, ir+4)):
                for c in range(max(0, ic-3), min(self.cols, ic+4)):
                    dist = abs(r - ir) + abs(c - ic)
                    if dist <= 3 and self.is_valid_position(r, c):
                        score = self.calculate_position_score_v3(r, c)
                        all_candidates.append({
                            "position": (r, c),
                            "score": score,
                            "distance": dist,
                            "industry": ind,
                            "current_count": 1,
                            "priority": "URGENTE"
                        })
        
        # Eliminar duplicados manteniendo el mejor score
        position_best = {}
        for cand in all_candidates:
            pos = cand['position']
            if pos not in position_best or cand['score'] > position_best[pos]['score']:
                position_best[pos] = cand
        
        unique_candidates = list(position_best.values())
        unique_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return ToolResult(
            success=True,
            message=f"Encontradas {len(unique_candidates[:top_n])} posiciones para {len(unsatisfied)} industrias",
            data={
                "candidates": unique_candidates[:top_n],
                "unsatisfied_industries": len(unsatisfied),
                "critical_industries": len(critical),
                "urgent_industries": len(urgent)
            }
        )
    
    def place_transformer(self, row: int, col: int, reason: str = "") -> ToolResult:
        if self.transformers_placed >= self.n_transformers:
            return ToolResult(False, f"Límite alcanzado ({self.n_transformers} transformadores)")
        
        if not self.is_valid_position(row, col):
            return ToolResult(False, f"Posición ({row}, {col}) inválida")
        
        self.grid[row][col] = 'C'
        self.transformers_placed += 1
        
        # Actualizar cobertura
        for ind in self.industries:
            dist = abs(row - ind[0]) + abs(col - ind[1])
            if dist <= 3:
                self.industry_coverage[ind] += 1
        
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
                "unsatisfied_industries": len(self.get_unsatisfied_industries())
            }
        )
    
    def place_multiple_transformers(self, positions: List[Dict[str, Any]]) -> ToolResult:
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
                message=f"No se pudo colocar ningún transformador",
                data={"placed": 0, "failed": failed}
            )
        
        return ToolResult(
            success=True,
            message=f"✓ {len(placed)} transformadores colocados",
            data={
                "placed": len(placed),
                "positions": placed,
                "failed": failed,
                "total_placed": self.transformers_placed,
                "remaining": self.n_transformers - self.transformers_placed,
                "unsatisfied_industries": len(self.get_unsatisfied_industries())
            }
        )
    
    def check_constraints(self) -> ToolResult:
        violations = []
        for ind in self.industries:
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
            message=f"✗ {len(violations)} industrias sin cobertura",
            data={"violations": violations}
        )
    
    def get_state(self) -> AgentState:
        return AgentState(
            grid=self.grid,
            transformers_placed=self.transformers_placed,
            target_transformers=self.n_transformers,
            iteration=len(self.history),
            history=self.history
        )


# ==================== WORKFLOW MEJORADO ====================

class ImprovedTransformerWorkflow(Workflow):
    
    def __init__(self, tools: TransformerTools, key_rotator, max_iterations: int = 200, verbose: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools = tools
        self.key_rotator = key_rotator
        self.max_iterations = max_iterations
        self.iteration = 0
        self.llm = Settings.llm
        self.verbose = verbose
        self.consecutive_failures = 0
    
    def _log(self, message: str):
        if self.verbose:
            print(message)

    async def _with_key_rotation(self, coro):
        try:
            return await coro
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                self._log("🚨 429 detectado → rotando key")
                self.key_rotator.rotate()
                initialize_llm(self.key_rotator)
                self.llm = Settings.llm
                self.iteration -= 1
                return await coro
            raise
    
    @step
    async def think(self, ev: Union[StartEvent, LoopEvent]) -> ThoughtEvent:
        self.iteration += 1
        self._log(f"\n{'='*60}\n🔄 ITERACIÓN {self.iteration}/{self.max_iterations}\n{'='*60}")
        
        state = self.tools.get_state().to_dict()
        
        if self.tools.transformers_placed >= self.tools.n_transformers:
            self._log("🎯 Objetivo alcanzado")
            return ThoughtEvent(
                thought='{"thought": "Verificando", "action": "check_constraints", "parameters": {}}',
                state=state
            )
        
        remaining = self.tools.n_transformers - self.tools.transformers_placed
        unsatisfied = self.tools.get_unsatisfied_industries()
        
        self._log(f"📊 {self.tools.transformers_placed}/{self.tools.n_transformers} | Insatisfechas: {len(unsatisfied)}/{len(self.tools.industries)}")
        
        if not unsatisfied:
            self._log("✅ Todas las industrias satisfechas")
            return ThoughtEvent(
                thought='{"thought": "Completado", "action": "check_constraints", "parameters": {}}',
                state=state
            )
        
        # Buscar posiciones para industrias críticas
        self._log(f"⚡ Buscando posiciones para {len(unsatisfied)} industrias...")
        candidates_result = self.tools.find_positions_for_critical_industries(top_n=15)
        
        if not candidates_result.success or not candidates_result.data.get('candidates'):
            self._log("⚠️ Sin candidatos")
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                return ThoughtEvent(
                    thought='{"thought": "Sin opciones", "action": "check_constraints", "parameters": {}}',
                    state=state
                )
            return ThoughtEvent(
                thought='{"thought": "Reintento", "action": "find_positions_for_critical_industries", "parameters": {"top_n": 10}}',
                state=state
            )
        
        self.consecutive_failures = 0
        candidates = candidates_result.data['candidates']
        
        # Batch size adaptativo
        batch_size = 1
        critical_count = candidates_result.data.get('critical_industries', 0)
        
        if critical_count > 0 and remaining >= 2:
            # Si hay industrias críticas (0 transf), priorizar colocar 2 de inmediato
            batch_size = min(2, remaining)
        elif remaining >= 10:
            batch_size = min(3, remaining // 3)
        elif remaining >= 5:
            batch_size = 2
        
        # Preparar prompt
        candidates_str = "\n".join([
            f"{i+1}. ({c['position'][0]},{c['position'][1]}) Score:{c['score']:.0f} "
            f"[{c['priority']}] Ind:{c['industry']} Dist:{c['distance']}"
            for i, c in enumerate(candidates[:10])
        ])
        
        unsatisfied_str = "\n".join([
            f"  - {ind}: {self.tools.count_transformers_near_industry(ind)}/2"
            for ind in unsatisfied[:5]
        ])
        
        if batch_size > 1:
            prompt = f"""URGENTE: Colocar transformadores para satisfacer restricciones.

ESTADO:
- Colocados: {self.tools.transformers_placed}/{self.tools.n_transformers}
- Restantes: {remaining}
- Industrias INSATISFECHAS: {len(unsatisfied)}

INDUSTRIAS CRÍTICAS:
{unsatisfied_str}

CANDIDATOS (ordenados por prioridad):
{candidates_str}

RESTRICCIÓN: Cada industria necesita ≥2 transformadores en radio ≤3.

DECISIÓN: Elige {batch_size} posiciones de la lista.

RESPONDE SOLO JSON:
{{
  "thought": "[estrategia]",
  "action": "place_multiple_transformers",
  "parameters": {{
    "positions": [
      {{"row": X, "col": Y, "reason": "Ind (A,B)"}},
      ...
    ]
  }}
}}"""
        else:
            prompt = f"""URGENTE: Colocar transformador para satisfacer restricciones.

ESTADO:
- Colocados: {self.tools.transformers_placed}/{self.tools.n_transformers}
- Restantes: {remaining}
- Industrias INSATISFECHAS: {len(unsatisfied)}

INDUSTRIAS CRÍTICAS:
{unsatisfied_str}

CANDIDATOS:
{candidates_str}

DECISIÓN: Elige UNO.

RESPONDE SOLO JSON:
{{
  "thought": "[razón]",
  "action": "place_transformer",
  "parameters": {{"row": X, "col": Y, "reason": "[razón]"}}
}}"""
        
        try:
            self._log(f"🤖 LLM decidiendo...")
            response = await self._with_key_rotation(
                asyncio.to_thread(self.llm.complete, prompt)
            )
            thought = response.text.strip()
            return ThoughtEvent(thought=thought, state=state)
        except Exception as e:
            self._log(f"❌ Error LLM: {e} → fallback")
            best = candidates[0]
            return ThoughtEvent(
                thought=f'{{"thought": "Fallback", "action": "place_transformer", "parameters": {{"row": {best["position"][0]}, "col": {best["position"][1]}, "reason": "Fallback"}}}}',
                state=state
            )
    
    @step
    async def act(self, ev: ThoughtEvent) -> ObservationEvent:
        decision = self._parse_llm_response(ev.thought)
        
        if not decision or 'action' not in decision:
            self._log("⚠️ Error parseando")
            for r in range(self.tools.rows):
                for c in range(self.tools.cols):
                    if self.tools.is_valid_position(r, c):
                        decision = {
                            "thought": "Fallback",
                            "action": "place_transformer",
                            "parameters": {"row": r, "col": c, "reason": "Fallback"}
                        }
                        break
                if decision and 'action' in decision:
                    break
            
            if not decision or 'action' not in decision:
                decision = {
                    "thought": "Sin opciones",
                    "action": "check_constraints",
                    "parameters": {}
                }
        
        action = decision.get('action', 'check_constraints')
        params = decision.get('parameters', {})
        thought = decision.get('thought', 'Sin pensamiento')
        
        self._log(f"🎯 ACCIÓN: {action}")
        
        tool_result = await self._with_key_rotation(
            asyncio.to_thread(self._execute_tool, action, params)
        )
        
        self._log(f"📤 {tool_result.message}")
        
        return ObservationEvent(
            thought=thought,
            action=action,
            result=tool_result.message,
            result_data=tool_result.data,
            state=ev.state
        )
    
    @step
    async def observe(self, ev: ObservationEvent) -> Union[LoopEvent, StopEvent]:
        
        if self.tools.transformers_placed >= self.tools.n_transformers:
            self._log("\n✅ Objetivo cumplido")
            check_result = self.tools.check_constraints()
            
            result = {
                "grid": self.tools.grid,
                "success": check_result.success,
                "transformers_placed": self.tools.transformers_placed,
                "initial_transformers": self.tools.initial_transformers,
                "new_transformers": self.tools.transformers_placed - self.tools.initial_transformers,
                "constraints_satisfied": check_result.success,
                "iterations": self.iteration,
                "history": self.tools.history,
                "final_check": check_result.to_dict()
            }
            
            if check_result.success:
                self._log("🎉 Todas las restricciones cumplidas")
            else:
                self._log(f"⚠️ {check_result.message}")
            
            return StopEvent(result=result)
        
        if self.iteration >= self.max_iterations:
            self._log(f"\n⏱️ Max iteraciones ({self.max_iterations})")
            return StopEvent(result={
                "grid": self.tools.grid,
                "success": False,
                "reason": "max_iterations",
                "transformers_placed": self.tools.transformers_placed,
                "target": self.tools.n_transformers,
                "iterations": self.iteration
            })
        
        return LoopEvent()
    
    def _parse_llm_response(self, text: str) -> Optional[Dict]:
        try:
            clean = text.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(clean)
        except:
            return None
    
    def _execute_tool(self, action: str, params: Dict) -> ToolResult:
        tool_mapping = {
            "place_transformer": self.tools.place_transformer,
            "place_multiple_transformers": self.tools.place_multiple_transformers,
            "check_constraints": self.tools.check_constraints,
            "find_positions_for_critical_industries": self.tools.find_positions_for_critical_industries,
        }
        
        if action not in tool_mapping:
            return ToolResult(False, f"Herramienta desconocida: {action}")
        
        try:
            return tool_mapping[action](**params)
        except Exception as e:
            return ToolResult(False, f"Error: {e}")


# ==================== UTILIDADES ====================

def print_grid(grid: List[List[str]], title: str = "MAPA"):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)
    for i, row in enumerate(grid):
        print(f"{i:2d} | {''.join(row)}")
    print('='*50)


def parse_grid_file(filepath: str) -> List[List[str]]:
    with open(filepath, 'r') as file:
        return [list(line.rstrip('\n')) for line in file]


def save_solution(grid: List[List[str]], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for row in grid:
            f.write(''.join(row) + '\n')
    print(f"💾 Guardado: {filepath}")


# ==================== MAIN ====================

async def main():
    print("🤖 AGENTE MEJORADO v4.1 - Completa Transformadores Faltantes")
    print("="*60)
    
    api_keys = [
        "AIzaSyDC63z3vOh92SV8480DOT3VtpudchTUB6o",
            "AIzaSyD9CaSXRAL-N5U8Gnx8sP-htW537uVm-4",
            "AIzaSyAYEdt7jxPlti2WVaBcgaID_2cXfK-v4Aw",
            "AIzaSyDwIK9uP1uuCxUjQhTOcyteOeUDShZfstU",
            "AIzaSyAXwDVyMhKS65MzWsscn9RnvsuE9kV9ars",
            "AIzaSyB0XfHor7g1wwisemTvubRJl6-bzAqpiYU",
            "AIzaSyDUILlsCqUeXC8_SUOtt0i3GA62uKsxmYY",
            "AIzaSyB5jJ23C3DBSCid33TQtJCwbcepYF1uha8",
            "AIzaSyDckZ2I1J8e6PyZWP3I1Dssoi-6BJ0gl_s",
            "AIzaSyCJtyhGS3iFp1jI7g9ZQV6Vztsh6C64_7I",
            "AIzaSyAVEBVp7R5IgVId0HBMNUJDTuYpRPUKWk8",
            "AIzaSyAZuRfiZfWdcWcW9XxT3pslUGfc6Svp1hw",
            "AIzaSyAvQb6smQ-nBgxoDrmwAi_fsWBpGXsISCA",
            "AIzaSyDF742oqAbzp_2kwT37Fkl7BnzziXY10Bs",
            "AIzaSyAyudSmUtONdqMl4gqiuUgVfQxeEcNthrI",
            "AIzaSyBfz3yCjug072zYPfayvpIeEKkJDtQwGOM",
            "AIzaSyBc9meShmHUNEzc_zy_T5tuVwhPjwzZE-M",
            "AIzaSyAGmFwK4Mn0W0IlvtBfP6qmxS8UVZijGc4",
            "AIzaSyDraXlWqmHi9SOMTU-jaBop2tOd7nVnfec",
            "AIzaSyArcctGMu4s4xWAEIAgT2Hi8B7-es3KtXo"
    ]
    
    if not api_keys or not api_keys[0]:
        print("❌ Configura API keys")
        return
    
    try:
        key_rotator = APIKeyRotator(api_keys)
        initialize_llm(key_rotator)
        print("✓ LLM listo\n")
    except Exception as e:
        print(f"❌ Error LLM: {e}")
        return
    
    filename = input("📁 Archivo: ").strip()
    map_file = os.path.join("entradas", filename)
    
    if not os.path.exists(map_file):
        print(f"❌ No encontrado: {map_file}")
        return
    
    try:
        n_transformers = int(input("🔢 Número total de transformadores objetivo: ").strip())
        if n_transformers <= 0:
            print("❌ Debe ser positivo")
            return
    except ValueError:
        print("❌ Número inválido")
        return
    
    try:
        grid = parse_grid_file(map_file)
        print(f"\n✓ Mapa: {len(grid)}x{max(len(row) for row in grid)}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    tools = TransformerTools(grid, n_transformers)
    
    print(f"\n📊 Análisis:")
    print(f"   Industrias: {len(tools.industries)}")
    print(f"   Transformadores ya colocados: {tools.initial_transformers}")
    print(f"   Objetivo total: {n_transformers}")
    print(f"   Por colocar: {n_transformers - tools.initial_transformers}")
    
    # Verificar estado inicial
    unsatisfied = tools.get_unsatisfied_industries()
    print(f"   Industrias insatisfechas: {len(unsatisfied)}/{len(tools.industries)}")
    
    if len(unsatisfied) == 0:
        print("\n✅ ¡Todas las industrias ya están satisfechas!")
        print("No se necesita colocar más transformadores.")
        return
    
    if n_transformers < tools.initial_transformers:
        print(f"\n❌ ERROR: Ya hay {tools.initial_transformers} transformadores")
        print(f"   El objetivo ({n_transformers}) es menor.")
        return
    
    workflow = ImprovedTransformerWorkflow(
        tools=tools,
        key_rotator=key_rotator,
        max_iterations=max(100, (n_transformers - tools.initial_transformers) * 4),
        verbose=True,
        timeout=600
    )
    
    print(f"\n🚀 Iniciando...\n")
    
    start_time = datetime.now()
    
    try:
        result = await workflow.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("🏁 FINALIZADO")
    print("="*60)
    print(f"⏱️  {duration:.2f}s")
    print(f"🔄 {workflow.iteration} iteraciones")
    
    if isinstance(result, dict) and 'grid' in result:
        print(f"\n📊 RESULTADOS:")
        print(f"   Iniciales: {result.get('initial_transformers', 0)}")
        print(f"   Nuevos: {result.get('new_transformers', 0)}")
        print(f"   Total: {result.get('transformers_placed', 0)}/{n_transformers}")
        print(f"   Éxito: {'✅' if result.get('success') else '❌'}")
        
        if result.get('success'):
            print(f"   ✓ Restricciones cumplidas")
        else:
            violations = result.get('final_check', {}).get('data', {}).get('violations', [])
            print(f"   ✗ {len(violations)} industrias sin cobertura")
            if violations:
                for v in violations[:5]:
                    print(f"     - {v['position']}: {v['current_transformers']}/2")
        
        try:
            output_file = os.path.join("salidas", filename)
            save_solution(result['grid'], output_file)
            
            log_file = os.path.join("salidas", f"{os.path.splitext(filename)[0]}_log.json")
            with open(log_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"📝 Log: {log_file}")
        except Exception as e:
            print(f"⚠️  Error guardando: {e}")


if __name__ == "__main__":
    asyncio.run(main())