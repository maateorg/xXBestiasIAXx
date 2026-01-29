import asyncio
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult

# === Definir las funciones personalizadas ===
def get_time(city: str) -> str:
    city_temperatures = {
        "Madrid": "25°C",
        "Londres": "18°C",
        "Nueva York": "22°C",
        "Tokio": "27°C",
        "París": "20°C"
    }
    temperature = city_temperatures.get(city, "Ciudad no encontrada")
    return f"La temperatura en {city} es {temperature}"

def get_population(city: str) -> str:
    city_populations = {
        "Madrid": "9.0 millones",
        "Londres": "8.9 millones",
        "Nueva York": "8.4 millones",
        "Tokio": "14 millones",
        "París": "2.1 millones"
    }
    population = city_populations.get(city, "Ciudad no encontrada")
    return f"La población de {city} es {population}"

# === Inicializa el modelo de Google Gemini ===
llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key="",
    generation_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)  # deshabilita thinking
    )
)

# === Crear el agente con tus funciones ===
agent = FunctionAgent(
    tools=[get_time, get_population],
    llm=llm,
)

# === Función para ejecutar el agente con streaming ===
async def run_agent_verbose(query: str):
    handler = agent.run(query)
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            print(
                f"\n🧰 Llamada a herramienta: {event.tool_name}"
                f"\n📥 Parámetros: {event.tool_kwargs}"
                f"\n📤 Resultado: {event.tool_output}\n"
            )
    return await handler

# === Función principal ===
async def main():
    pregunta = "¿Cuál es la temperatura actual de Madrid y la población de Madrid?"
    print(f"🤖 Pregunta: {pregunta}\n")
    response = await run_agent_verbose(pregunta)
    print(f"\n✅ Respuesta final del agente:\n{response}")

# === Ejecutar ===
if __name__ == "__main__":
    asyncio.run(main())
