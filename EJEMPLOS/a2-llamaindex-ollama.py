# pip install llama-index-llms-ollama
from llama_index.llms.ollama import Ollama

# Local LLM to send user query to
local_llm = Ollama(model="llama3.2",
                   request_timeout=600.0, 
                   system_prompt='responde siempre en español',
                   base_url="http://156.35.95.18:11434",
                   context_window=8000,
                   temperature=0.1)

query = "¿Cuantas comunidades autonomas tiene España?"
print(query)
result = local_llm.complete(query)
print(result)

query = "Listame las comunidades autonomas de españa"
print(query)
for chunk in local_llm.stream_complete(query):
    print(chunk.text, end="")