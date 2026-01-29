import asyncio
from typing import Any, List

from llama_index.core.tools import FunctionTool
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from llama_index.llms.google_genai import GoogleGenAI

# --- Inicializa el modelo de Google Gemini ---
llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=""
)

# --- Definición de eventos ---
class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


# --- Agente principal ---
class FunctionCallingAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm or llm
        assert hasattr(self.llm, "astream_chat_with_tools"), \
            "El LLM no soporta llamadas a herramientas."

    @step
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        # Inicializa el contexto
        ctx.update({"sources": []})

        memory = ctx.get("memory", None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        chat_history = memory.get()
        ctx.update({"memory": memory})
        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> ToolCallEvent | StopEvent:
        chat_history = ev.input

        # Llamada asíncrona al modelo
        response_stream = await self.llm.astream_chat_with_tools(self.tools, chat_history=chat_history)
        response = None
        async for chunk in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=chunk.delta or ""))
            response = chunk  # guarda el último fragmento

        if not response:
            return StopEvent(result={"response": "Sin respuesta del modelo", "sources": []})

        memory = ctx.get("memory")
        memory.put(response.message)
        ctx.update({"memory": memory})

        tool_calls = self.llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)

        if not tool_calls:
            sources = ctx.get("sources", [])
            return StopEvent(result={"response": response.message.content, "sources": sources})
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        tool_msgs = []
        sources = ctx.get("sources", [])

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool_call.tool_name,
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"La herramienta {tool_call.tool_name} no existe.",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                result = tool(**tool_call.tool_kwargs)
                sources.append(result)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=result.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Error al ejecutar la herramienta {tool_call.tool_name}: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        memory = ctx.get("memory")
        for msg in tool_msgs:
            memory.put(msg)

        ctx.update({"sources": sources, "memory": memory})
        chat_history = memory.get()
        return InputEvent(input=chat_history)


# --- Herramientas (Tools) ---
def get_temperature(city: str) -> str:
    city_temperatures = {
        "Madrid": "35°C",
        "Londres": "18°C",
        "Nueva York": "22°C",
        "Tokio": "27°C",
        "París": "20°C",
    }
    temperature = city_temperatures.get(city, "Ciudad no encontrada")
    return f"La temperatura en {city} es {temperature}"


def get_population(city: str) -> str:
    city_populations = {
        "Madrid": "9.3 millones",
        "Londres": "8.9 millones",
        "Nueva York": "8.4 millones",
        "Tokio": "14 millones",
        "París": "2.1 millones",
    }
    population = city_populations.get(city, "Ciudad no encontrada")
    return f"La población de {city} es {population}"


# Crear herramientas
time_tool = FunctionTool.from_defaults(
    fn=get_temperature,
    name="get_temperature",
    description="Devuelve la temperatura actual de una ciudad.",
)

population_tool = FunctionTool.from_defaults(
    fn=get_population,
    name="get_population",
    description="Devuelve la población actual de una ciudad.",
)


# --- Función principal ---
async def main():
    agent = FunctionCallingAgent(
        llm=llm,
        tools=[time_tool, population_tool],
        timeout=120,
        verbose=True,
    )

    ret = await agent.run(input="¿Cuál es la temperatura y población actuales de Madrid?")
    print("\n### Respuesta del agente ###")
    print(ret["response"])


if __name__ == "__main__":
    asyncio.run(main())