
# pip install llama_index.llms.google_genai 
from llama_index.llms.google_genai import GoogleGenAI
#from llama_index.core import Settings

# Inicializa el modelo de Google Gemini
llm = GoogleGenAI(
    model="gemini-2.5-flash",  # puedes cambiar a gemini-1.5-flash si quieres más velocidad
    api_key=""  # o usar variable de entorno GOOGLE_API_KEY
)

# Establece este modelo como predeterminado
# Settings.llm = llm

# Consulta
query = "Dime las comunidades autónomas de España en formato JSON."
print("Consulta:", query)

# Genera la respuesta
result = llm.complete(query)
print("Respuesta:\n", result)