from llama_index.llms.ollama import Ollama
from llama_index.core.tools import ToolMetadata
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key="AIzaSyCDbOSn9BoDnRjsf_Sal4IhJBkR3DJ1b18",
    generation_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)  # deshabilita thinking
    )
)

# Definición de herramientas con descripciones más atractivas
choices = [
    ToolMetadata(
        name="financial_data_tool",
        description="Accede a reportes financieros históricos de empresas globales, incluyendo ingresos, crecimiento anual y datos por sector."
    ),
    ToolMetadata(
        name="news_summary_tool",
        description="Ofrece resúmenes de noticias económicas, anuncios corporativos y eventos clave en el mundo empresarial."
    ),
]

# Crear el selector
selector = LLMSingleSelector.from_defaults(llm=llm)

# Consulta en español
query = "¿Cuál fue el crecimiento de ingresos de IBM en el año 2007?"

# Seleccionar herramienta
selector_result = selector.select(choices, query=query)
print(selector_result)