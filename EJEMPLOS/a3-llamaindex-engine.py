from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine import SimpleChatEngine

# Inicializar LLM
llm = Ollama(model="llama3.2",
        request_timeout=600.0, 
        system_prompt='responde siempre en español',
        base_url="http://156.35.95.18:11434",
        context_window=8000,
        temperature=0.1)

# Crear motor de chat con memoria básica
chat_engine = SimpleChatEngine.from_defaults(llm=llm)

# Simular conversación
print(chat_engine.chat("Me llamo Luis."))
print(chat_engine.chat("¿Cuál es mi nombre?"))  # Debería recordar "Luis"