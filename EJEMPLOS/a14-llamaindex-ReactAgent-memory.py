import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer


llm = Ollama(model="llama3.2",
        request_timeout=600.0, 
        system_prompt='responde siempre en español',
        base_url="http://156.35.95.18:11434",
        context_window=8000,
        temperature=0.1)

# Crear memoria de conversación (opcional pero recomendado)

memory = ChatMemoryBuffer(memory_key="chat_history", token_limit=4000)

# Crear contexto del agente con información adicional
agent = FunctionAgent(
    llm=llm,
    system_prompt=("Responde siempre en Español a mis preguntas.\n")
)

async def main():
    response = await agent.run(user_msg="Hola, mi nombre es Jordán!", memory=memory)
    print("########:"+str(response))

    print("Memory:::::"+str(memory.get_all()))
    #memory.reset()

    response2 = await agent.run(user_msg="¿Cual es mi nombre?", memory=memory)
    print("########:"+str(response2))


if __name__ == "__main__":
    asyncio.run(main())