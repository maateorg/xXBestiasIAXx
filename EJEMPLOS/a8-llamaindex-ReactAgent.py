
# https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/usage_pattern/
import asyncio
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import ReActAgent

# Configurar el modelo de Ollama con Mistral
llm = Ollama(model="llama3.2",
                   request_timeout=600.0, 
                   system_prompt='responde siempre en español',
                   base_url="http://156.35.95.18:11434",
                   context_window=8000,
                   temperature=0.1)

# ReActAgent (Reasoning + Acting)  , razona y actua usando las tools Paso a Paso (hay otros)
agent = ReActAgent(
    llm=llm, 
    verbose=True,
    max_iterations=3 
)

async def main():
    response = await agent.run("¿Cual es el idioma más dificil de aprender?")
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())