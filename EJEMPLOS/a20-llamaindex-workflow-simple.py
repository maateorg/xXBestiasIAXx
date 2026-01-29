from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step
import asyncio
from llama_index.llms.google_genai import GoogleGenAI

class EmailDraftEvent(Event):
    draft: str
    goal: str
    recipient: str


class EmailWriterFlow(Workflow):
    
    llm = GoogleGenAI(
        model="gemini-2.5-flash",  # puedes cambiar a gemini-1.5-flash si quieres más velocidad
        api_key="AIzaSyCDbOSn9BoDnRjsf_Sal4IhJBkR3DJ1b18"  # o usar variable de entorno GOOGLE_API_KEY
    )

    # Paso 1: generar borrador inicial
    @step
    async def draft_email(self, ev: StartEvent) -> EmailDraftEvent:
        goal = ev.goal
        recipient = ev.recipient

        prompt = (
            f"Escribe un email cuyo objetivo es: \"{goal}\". "
            f"Está dirigido a {recipient}. "
            f"Usa el tono adecuado según el contexto. Solo escribe el cuerpo del mensaje."
        )

        # Llamada asíncrona al modelo
        response = await self.llm.acomplete(prompt)
        return EmailDraftEvent(draft=response.text.strip(), goal=goal, recipient=recipient)

    # Paso 2: mejorar el borrador
    @step
    async def improve_email(self, ev: EmailDraftEvent) -> StopEvent:
        prompt = (
            f"Teniendo en cuenta el objetivo \"{ev.goal}\" y que está dirigido a {ev.recipient}, "
            f"reescribe el siguiente borrador de email para hacerlo más claro, profesional y efectivo:\n\n"
            f"{ev.draft}"
        )

        # Llamada asíncrona al modelo
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=response.text.strip())


# --- Ejemplo de ejecución ---
async def main():
    flow = EmailWriterFlow(timeout=3000)
    result = await flow.run(
        goal="solicitar una reunión para explorar una colaboración estratégica",
        recipient="el director de tecnología de una startup de salud digital"
    )
    print("########### Resultado final ###########")
    print(result)


asyncio.run(main())