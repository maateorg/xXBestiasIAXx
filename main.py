"""
Agente Autónomo de Colocación de Transformadores - VERSIÓN MEJORADA v4.0.1
CORRECCIÓN: Priorización absoluta de industrias críticas + Fix rotación API
"""

import os
import json
import re
import time
import asyncio
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
from collections import deque, defaultdict
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


# ==================== HERRAMIENTAS MEJORADAS ====================

class TransformerTools:
    """Herramientas disponibles para el agente con estrategia mejorada"""
    
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
        # Cache de industrias
        self.industries = self._find_industries()
        self.industry_coverage = {ind: 0 for ind in self.industries}
    
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
    
    def _find_industries(self) -> List[Tuple[int, int]]:
        """Encuentra todas las industrias en el mapa"""
        return [(r, c) for r in range(self.rows) 
                for c in range(self.cols) 
                if self.grid[r][c] == 'T']
    
    def get_unsatisfied_industries(self) -> List[Tuple[int, int]]:
        """Obtiene industrias que necesitan más transformadores"""
        unsatisfied = []
        for ind in self.industries:
            count = self.count_transformers_near_industry(ind)
            if count < 2:
                unsatisfied.append(ind)
        return unsatisfied
    
    def get_critical_industries(self) -> Dict[str, List[Tuple[int, int]]]:
        """Clasifica industrias por nivel de criticidad"""
        critical = []  # 0 transformadores
        urgent = []    # 1 transformador
        
        for ind in self.industries:
            count = self.count_transformers_near_industry(ind)
            if count == 0:
                critical.append(ind)
            elif count == 1:
                urgent.append(ind)
        
        return {
            "critical": critical,
            "urgent": urgent
        }
    
    def calculate_position_score_v3(self, row: int, col: int) -> float:
        """Calcula score MEJORADO v3 con MÁXIMA prioridad a industrias críticas"""
        if not self.is_valid_position(row, col):
            return -1000.0
        
        score = 0.0
        neighbors = self.get_neighbors(row, col)
        
        # Bonificaciones básicas por vecinos (reducidas)
        hospital_neighbors = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'O')
        industry_neighbors = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'T')
        house_neighbors = sum(1 for nr, nc in neighbors if self.grid[nr][nc] == 'X')
        
        score += hospital_neighbors * 5.0
        score += industry_neighbors * 3.0
        score += house_neighbors * 1.0
        
        # ESTRATEGIA CRÍTICA: Priorizar industrias insatisfechas
        classification = self.get_critical_industries()
        critical = classification["critical"]
        urgent = classification["urgent"]
        
        # MÁXIMA PRIORIDAD: Industrias con 0 transformadores
        for ind in critical:
            dist = abs(row - ind[0]) + abs(col - ind[1])
            if dist <= 3:
                # BONIFICACIÓN EXPONENCIAL
                score += (4 - dist) * 200.0  # Hasta 800 puntos
        
        # ALTA PRIORIDAD: Industrias con 1 transformador
        for ind in urgent:
            dist = abs(row - ind[0]) + abs(col - ind[1])
            if dist <= 3:
                score += (4 - dist) * 100.0  # Hasta 400 puntos
        
        # Penalización FUERTE por estar cerca de industrias satisfechas
        satisfied = [ind for ind in self.industries 
                    if ind not in critical and ind not in urgent]
        for ind in satisfied:
            dist = abs(row - ind[0]) + abs(col - ind[1])
            if dist <= 3:
                score -= 20.0
        
        return score
    
    def find_strategic_positions_for_industries(self, top_n: int = 20) -> ToolResult:
        """Encuentra posiciones estratégicas priorizando industrias críticas"""
        classification = self.get_critical_industries()
        critical = classification["critical"]
        urgent = classification["urgent"]
        
        if not critical and not urgent:
            return self.find_best_candidates(top_n)
        
        # Primero buscar para CRÍTICAS (0 transformadores)
        all_candidates = []
        
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
        
        # Luego para URGENTES (1 transformador)
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
            message=f"Encontradas {len(unique_candidates[:top_n])} posiciones estratégicas",
            data={
                "candidates": unique_candidates[:top_n],
                "critical_industries": len(critical),
                "urgent_industries": len(urgent),
                "total_unsatisfied": len(critical) + len(urgent)
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
        
        # Actualizar cobertura de industrias
        for ind in self.industries:
            dist = abs(row - ind[0]) + abs(col - ind[1])
            if dist <= 3:
                self.industry_coverage[ind] += 1
        
        self.history.append({
            "action": "place",
            "position": (row, col),
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "industries_affected": [ind for ind in self.industries if abs(row - ind[0]) + abs(col - ind[1]) <= 3]
        })
        
        return ToolResult(
            success=True,
            message=f"Transformador #{self.transformers_placed} colocado en ({row}, {col})",
            data={
                "total_placed": self.transformers_placed,
                "remaining": self.n_transformers - self.transformers_placed,
                "reason": reason,
                "unsatisfied_industries": len(self.get_unsatisfied_industries())
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
                "remaining": self.n_transformers - self.transformers_placed,
                "unsatisfied_industries": len(self.get_unsatisfied_industries())
            }
        )
    
    def check_constraints(self) -> ToolResult:
        """Verifica que se cumplan las restricciones de industrias"""
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
            message=f"✗ {len(violations)} industrias sin cobertura adecuada",
            data={"violations": violations}
        )
    
    def find_best_candidates(self, top_n: int = 10) -> ToolResult:
        """Encuentra las mejores posiciones candidatas (fallback)"""
        candidates = []
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_valid_position(r, c):
                    score = self.calculate_position_score_v3(r, c)
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
        for ind in self.industries:
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
            data={
                "coverage": coverage,
                "satisfaction_rate": satisfied/total_industries if total_industries > 0 else 0,
                "unsatisfied": [c for c in coverage if not c["satisfied"]]
            }
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


# ==================== WORKFLOW MEJORADO ====================

class ImprovedTransformerWorkflow(Workflow):
    """Workflow mejorado con estrategia industria-centrada"""
    
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

    async def _with_key_rotation(self, coro_func, *args, **kwargs):
        """Ejecuta una función con reintentos y rotación de API key"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Ejecutar la coroutine
                if asyncio.iscoroutinefunction(coro_func):
                    return await coro_func(*args, **kwargs)
                else:
                    return await coro_func
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "resource_exhausted" in msg or "permission_denied" in msg:
                    if attempt < max_retries - 1:
                        self._log(f"🚨 Error API detectado (intento {attempt + 1}/{max_retries}) → rotando key")
                        self.key_rotator.rotate()
                        initialize_llm(self.key_rotator)
                        self.llm = Settings.llm
                        await asyncio.sleep(1)  # Pequeña pausa
                        continue
                    else:
                        self._log(f"❌ Máximo de reintentos alcanzado")
                        raise
                else:
                    raise
        raise Exception("No se pudo completar la operación después de varios reintentos")
    
    @step
    async def think(self, ev: Union[StartEvent, LoopEvent]) -> ThoughtEvent:
        """Paso de razonamiento mejorado"""
        self.iteration += 1
        self._log(f"\n{'='*60}\n🔄 ITERACIÓN {self.iteration}/{self.max_iterations}\n{'='*60}")
        
        state = self.tools.get_state().to_dict()
        
        # Verificar si ya terminamos
        if self.tools.transformers_placed >= self.tools.n_transformers:
            self._log("🎯 Objetivo alcanzado. Verificando restricciones...")
            return ThoughtEvent(
                thought='{"thought": "Verificando restricciones finales", "action": "check_constraints", "parameters": {}}',
                state=state
            )
        
        remaining = self.tools.n_transformers - self.tools.transformers_placed
        classification = self.tools.get_critical_industries()
        critical = classification["critical"]
        urgent = classification["urgent"]
        
        self._log(f"📊 Estado: {self.tools.transformers_placed}/{self.tools.n_transformers} colocados")
        self._log(f"🔴 Industrias CRÍTICAS (0 transf): {len(critical)}")
        self._log(f"🟡 Industrias URGENTES (1 transf): {len(urgent)}")
        
        # ESTRATEGIA: Priorizar industrias críticas
        if critical or urgent:
            self._log(f"⚡ Buscando posiciones para industrias insatisfechas...")
            candidates_result = self.tools.find_strategic_positions_for_industries(top_n=20)
        else:
            self._log("⚡ Todas las industrias satisfechas, buscando candidatos generales...")
            candidates_result = self.tools.find_best_candidates(top_n=15)
        
        if not candidates_result.success or not candidates_result.data.get('candidates'):
            self._log("⚠️ No hay candidatos disponibles")
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                return ThoughtEvent(
                    thought='{"thought": "No hay posiciones válidas tras múltiples intentos", "action": "check_constraints", "parameters": {}}',
                    state=state
                )
            return ThoughtEvent(
                thought='{"thought": "Reintentar búsqueda", "action": "find_best_candidates", "parameters": {"top_n": 10}}',
                state=state
            )
        
        self.consecutive_failures = 0
        candidates = candidates_result.data['candidates']
        
        # Decidir batch size adaptativo
        batch_size = 1
        if len(critical) > 0 and remaining >= 2:
            # Si hay críticas, colocar múltiples
            batch_size = min(len(critical) * 2, remaining, 5)
        elif len(urgent) > 0 and remaining >= 2:
            batch_size = min(len(urgent), remaining, 3)
        elif remaining >= 10:
            batch_size = min(3, remaining // 3)
        elif remaining >= 5:
            batch_size = 2
        
        # Preparar info para LLM
        candidates_str = "\n".join([
            f"{i+1}. ({c['position'][0]},{c['position'][1]}) Score:{c['score']:.0f} "
            f"[{c.get('priority', 'N/A')}] Ind:{c.get('industry', 'N/A')} "
            f"Actual:{c.get('current_count', '?')}/2"
            for i, c in enumerate(candidates[:15])
        ])
        
        critical_str = "\n".join([
            f"  🔴 {ind}: 0/2 transformadores"
            for ind in critical[:5]
        ])
        
        urgent_str = "\n".join([
            f"  🟡 {ind}: 1/2 transformadores"
            for ind in urgent[:5]
        ])
        
        if batch_size > 1:
            prompt = f"""URGENTE: Colocar transformadores para industrias CRÍTICAS.

**ESTADO:**
- Colocados: {self.tools.transformers_placed}/{self.tools.n_transformers}
- Restantes: {remaining}
- CRÍTICAS (0 transf): {len(critical)}
- URGENTES (1 transf): {len(urgent)}

**INDUSTRIAS CRÍTICAS:**
{critical_str if critical else "  ✓ Ninguna"}

**INDUSTRIAS URGENTES:**
{urgent_str if urgent else "  ✓ Ninguna"}

**CANDIDATOS (ordenados por prioridad):**
{candidates_str}

**RESTRICCIÓN ABSOLUTA:**
Cada industria DEBE tener ≥2 transformadores en radio Manhattan ≤3.

**DECISIÓN:**
Elige {batch_size} posiciones de la lista.
PRIORIDAD MÁXIMA: Industrias con 0 transformadores.

**RESPONDE SOLO JSON:**
{{
  "thought": "[estrategia]",
  "action": "place_multiple_transformers",
  "parameters": {{
    "positions": [
      {{"row": X, "col": Y, "reason": "Para industria (A,B)"}},
      ...
    ]
  }}
}}"""
        else:
            prompt = f"""URGENTE: Colocar transformador para industria CRÍTICA.

**ESTADO:**
- Colocados: {self.tools.transformers_placed}/{self.tools.n_transformers}
- Restantes: {remaining}
- CRÍTICAS (0 transf): {len(critical)}
- URGENTES (1 transf): {len(urgent)}

**INDUSTRIAS CRÍTICAS:**
{critical_str if critical else "  ✓ Ninguna"}

**INDUSTRIAS URGENTES:**
{urgent_str if urgent else "  ✓ Ninguna"}

**CANDIDATOS:**
{candidates_str}

**DECISIÓN:**
Elige UNA posición. PRIORIDAD MÁXIMA a industrias con 0 transformadores.

**RESPONDE SOLO JSON:**
{{
  "thought": "[razón]",
  "action": "place_transformer",
  "parameters": {{"row": X, "col": Y, "reason": "[razón]"}}
}}"""
        
        try:
            self._log(f"🤖 LLM decidiendo {batch_size} posición/es...")
            
            # Crear función async para llamar al LLM
            async def llm_complete():
                return await asyncio.to_thread(self.llm.complete, prompt)
            
            response = await self._with_key_rotation(llm_complete)
            thought = response.text.strip()
            
            return ThoughtEvent(thought=thought, state=state)
        except Exception as e:
            self._log(f"❌ Error LLM: {e} → fallback al mejor candidato")
            best = candidates[0]
            return ThoughtEvent(
                thought=f'{{"thought": "Fallback heurístico", "action": "place_transformer", "parameters": {{"row": {best["position"][0]}, "col": {best["position"][1]}, "reason": "Fallback"}}}}',
                state=state
            )
    
    @step
    async def act(self, ev: ThoughtEvent) -> ObservationEvent:
        """Ejecuta la acción decidida"""
        decision = self._parse_llm_response(ev.thought)
        
        if not decision or 'action' not in decision:
            self._log("⚠️ Error parseando → usando fallback")
            # Buscar primera posición válida
            for r in range(self.tools.rows):
                for c in range(self.tools.cols):
                    if self.tools.is_valid_position(r, c):
                        decision = {
                            "thought": "Fallback por error",
                            "action": "place_transformer",
                            "parameters": {"row": r, "col": c, "reason": "Fallback"}
                        }
                        break
                if decision and 'action' in decision:
                    break
            
            if not decision or 'action' not in decision:
                decision = {
                    "thought": "Sin posiciones válidas",
                    "action": "check_constraints",
                    "parameters": {}
                }
        
        action = decision.get('action', 'check_constraints')
        params = decision.get('parameters', {})
        thought = decision.get('thought', 'Sin pensamiento')
        
        self._log(f"🎯 ACCIÓN: {action}")
        
        # Crear función async para ejecutar la herramienta
        async def execute_tool():
            return await asyncio.to_thread(self._execute_tool, action, params)
        
        tool_result = await self._with_key_rotation(execute_tool)
        
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
        """Decide si continuar o terminar"""
        
        if self.tools.transformers_placed >= self.tools.n_transformers:
            self._log("\n✅ Objetivo cumplido. Verificando restricciones...")
            check_result = self.tools.check_constraints()
            
            result = {
                "grid": self.tools.grid,
                "success": check_result.success,
                "transformers_placed": self.tools.transformers_placed,
                "constraints_satisfied": check_result.success,
                "iterations": self.iteration,
                "history": self.tools.history,
                "final_check": check_result.to_dict()
            }
            
            if check_result.success:
                self._log("🎉 ¡Todas las restricciones cumplidas!")
            else:
                self._log(f"⚠️ Restricciones no cumplidas: {check_result.message}")
            
            return StopEvent(result=result)
        
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
            "find_best_candidates": self.tools.find_best_candidates,
            "find_strategic_positions_for_industries": self.tools.find_strategic_positions_for_industries,
            "get_industry_coverage": self.tools.get_industry_coverage,
        }
        
        if action not in tool_mapping:
            return ToolResult(False, f"Herramienta desconocida: {action}")
        
        try:
            return tool_mapping[action](**params)
        except Exception as e:
            return ToolResult(False, f"Error ejecutando {action}: {e}")


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
        return [list(line.strip()) for line in file if line.strip()]


def save_solution(grid: List[List[str]], filepath: str = "salidas/solucion.txt"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for row in grid:
            f.write(''.join(row) + '\n')
    print(f"💾 Solución guardada en: {filepath}")


# ==================== MAIN ====================

async def main():
    print("🤖 AGENTE MEJORADO DE TRANSFORMADORES v4.0.1")
    print("="*60)
    print("⚡ CORRECCIÓN: Máxima prioridad a industrias críticas")
    print("="*60)
    
    # Configurar API
    api_keys = [
           


        ]
    
    if not api_keys or not api_keys[0]:
        print("❌ Configura tus API keys en el código")
        return
    
    try:
        key_rotator = APIKeyRotator(api_keys)
        initialize_llm(key_rotator)
        print("✓ LLM inicializado\n")
    except Exception as e:
        print(f"❌ Error configurando LLM: {e}")
        return
    
    # Solicitar archivo y número
    filename = input("📁 Archivo del mapa: ").strip()
    map_file = os.path.join("entradas", filename)
    
    if not os.path.exists(map_file):
        print(f"❌ Archivo no encontrado: {map_file}")
        return
    
    try:
        n_transformers = int(input("🔢 Número de transformadores: ").strip())
        if n_transformers <= 0:
            print("❌ Número debe ser positivo")
            return
    except ValueError:
        print("❌ Número inválido")
        return
    
    # Cargar mapa
    try:
        grid = parse_grid_file(map_file)
        print(f"\n✓ Mapa cargado: {len(grid)}x{len(grid[0])}")
    except Exception as e:
        print(f"❌ Error leyendo mapa: {e}")
        return
    
    print_grid(grid, "MAPA INICIAL")
    
    # Crear workflow
    tools = TransformerTools(grid, n_transformers)
    
    print(f"\n📊 Análisis inicial:")
    print(f"   - Industrias totales: {len(tools.industries)}")
    print(f"   - Transformadores necesarios: {n_transformers}")
    print(f"   - Ratio mínimo esperado: {len(tools.industries) * 2}")
    
    if n_transformers < len(tools.industries) * 2:
        print(f"\n⚠️  ADVERTENCIA: {n_transformers} transformadores podrían ser insuficientes")
        print(f"   Se necesitan al menos {len(tools.industries) * 2} para cubrir todas las industrias")
    
    workflow = ImprovedTransformerWorkflow(
        tools=tools,
        key_rotator=key_rotator,
        max_iterations=max(300, n_transformers * 4),
        verbose=True,
        timeout=600
    )
    
    print(f"\n🚀 Iniciando agente...\n")
    
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
    
    # Resultados
    print("\n" + "="*60)
    print("🏁 EJECUCIÓN FINALIZADA")
    print("="*60)
    print(f"⏱️  Tiempo: {duration:.2f}s")
    print(f"🔄 Iteraciones: {workflow.iteration}")
    
    if isinstance(result, dict) and 'grid' in result:
        print_grid(result['grid'], "MAPA FINAL")
        
        print(f"\n📊 RESULTADOS:")
        print(f"   Colocados: {result.get('transformers_placed', 0)}/{n_transformers}")
        print(f"   Éxito: {'✅ SÍ' if result.get('success') else '❌ NO'}")
        
        if result.get('success'):
            print(f"   Estado: ✓ Todas las restricciones cumplidas")
        else:
            final_check = result.get('final_check', {})
            violations = final_check.get('data', {}).get('violations', [])
            print(f"   Estado: ✗ {len(violations)} industrias sin cobertura")
            if violations:
                print(f"\n   Industrias insatisfechas:")
                for v in violations[:5]:
                    print(f"     - {v['position']}: {v['current_transformers']}/2")
        
        # Guardar
        try:
            output_file = os.path.join("salidas", filename)
            save_solution(result['grid'], output_file)
            
            log_file = os.path.join("salidas", f"{os.path.splitext(filename)[0]}_log.json")
            with open(log_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"📝 Log guardado: {log_file}")
        except Exception as e:
            print(f"⚠️  Error guardando: {e}")


if __name__ == "__main__":
    asyncio.run(main())