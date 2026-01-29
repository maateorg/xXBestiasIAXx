import asyncio
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent

# Crear modelo LLM
llm = Ollama(model="llama3.2",
        request_timeout=600.0, 
        system_prompt='responde siempre en español',
        base_url="http://156.35.95.18:11434",
        context_window=8000,
        temperature=0.1)

# Crear contexto del agente con información adicional
agent = FunctionAgent(
    llm=llm,
    system_prompt=("Responde siempre en Español a mis preguntas.\n")
)

ctx = Context(agent)

async def main():
    response = await agent.run("Hola, mi nombre es Jordán!", ctx=ctx)
    print(str(response))
    response2 = await agent.run("¿como me llamo?", ctx=ctx)
    print(response2)

if __name__ == "__main__":
    asyncio.run(main())