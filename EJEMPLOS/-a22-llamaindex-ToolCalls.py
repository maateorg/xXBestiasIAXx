from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.google_genai import GoogleGenAI
#from llama_index.core import Settings

# Inicializa el modelo de Google Gemini
llm = GoogleGenAI(
    model="gemini-2.5-flash",  # puedes cambiar a gemini-1.5-flash si quieres más velocidad
    api_key=""  # o usar variable de entorno GOOGLE_API_KEY
)

# Nueva función que calcula el siguiente número de una serie
def subir_temperatura(temperatura: int) -> int:
    """Suma 1 a la temperatura actual. Usa esta función repetidamente hasta alcanzar la temperatura deseada."""
    return temperatura + 1

# Crear herramienta con la función nueva
tools = [FunctionTool.from_defaults(subir_temperatura)]
tools_by_name = {t.metadata.name: t for t in tools}

# Chat de entrada
chat_history = [
    ChatMessage(role="user", content="Empieza en temperatura 20 y sube hasta llegar a 25 de un grado en un grado.")
]

# Llamar al LLM con herramientas y chat
resp = llm.chat_with_tools(tools, chat_history=chat_history)

# Procesar llamadas a herramientas
tool_calls = llm.get_tool_calls_from_response(resp, error_on_no_tool_call=False)

while tool_calls:
    chat_history.append(resp.message)

    for tool_call in tool_calls:
        tool_name = tool_call.tool_name
        tool_kwargs = tool_call.tool_kwargs

        print(f"Llamando a {tool_name} con {tool_kwargs}")
        func = globals().get(tool_name)
        tool_output = func(**tool_kwargs)
        chat_history.append(
            ChatMessage(
                role="tool",
                content=str(tool_output),
                additional_kwargs={"tool_call_id": tool_call.tool_id},
            )
        )

        # Nueva respuesta del LLM
        resp = llm.chat_with_tools(tools, chat_history=chat_history)
        print(f"chat_with_tools")
        tool_calls = llm.get_tool_calls_from_response(resp, error_on_no_tool_call=False)

# Mostrar la respuesta final
print(resp.message.content)