import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage

llm = Ollama(model="llama3.2",
        request_timeout=600.0, 
        system_prompt='responde siempre en español',
        base_url="http://156.35.95.18:11434",
        context_window=8000,
        temperature=0.1)

# Crear memoria de conversación (opcional pero recomendado)
memory = ChatMemoryBuffer(memory_key="chat_history", token_limit=4000)
chat_history = [
    ChatMessage(role="user", content="Hola, me llamo Jordán"),
    ChatMessage(role="assistant", content="Hola Jordán, que puedo hacer por ti?")
]
memory.put_messages(chat_history)

# Crear contexto del agente con información adicional
agent = FunctionAgent(
    llm=llm,
    system_prompt=("Responde siempre en Español a mis preguntas.\n")
)

async def main():
    response1 = await agent.run("¿Cual es mi nombre?")
    print("########:"+str(response1))

    response2 = await agent.run("¿Cual es mi nombre?", memory=memory)
    print("########:"+str(response2))

    response3 = await agent.run("¿Cual es mi nombre?", chat_history=memory.get())
    print("########:"+str(response3))

if __name__ == "__main__":
    asyncio.run(main())