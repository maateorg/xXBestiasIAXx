from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step
from llama_index.llms.google_genai import GoogleGenAI
import asyncio

MAX_REVISIONS = 2

class TweetEvent(Event):
    tweet: str
    attempts: int = 0

class IterateTweetEvent(Event):
    tweet: str
    review: str
    attempts: int

class FinishTweetEvent(Event):
    tweet: str

class TweetFlow(Workflow):
    llm = GoogleGenAI(
        model="gemini-2.5-flash",  # puedes cambiar a gemini-1.5-flash si quieres más velocidad
        api_key=""  # o usar variable de entorno GOOGLE_API_KEY
    )

    # Paso 1: generar tweet inicial
    @step
    async def generate_tweet(self, ev: StartEvent) -> TweetEvent:
        topic = ev.topic
        prompt = f"Escribe un tweet creativo sobre el tema: '{topic}'."
        response = await self.llm.acomplete(prompt)
        return TweetEvent(tweet=response.text.strip(), attempts=0)

    # Paso 2: revisar si cumple con las condiciones
    @step
    async def review_tweet(self, ev: TweetEvent) -> FinishTweetEvent | IterateTweetEvent:
        tweet = ev.tweet
        prompt = (
            f"Evalúa este tweet:\n\"{tweet}\"\n\n"
            "¿Cumple con las siguientes condiciones?\n"
            "- Contiene un dato curioso o llamativo\n"
            "- Incluye al menos un hashtag\n"
            "- Tiene menos de 100 caracteres\n\n"
            "Si cumple con todas, responde con: 'APROBADO'.\n"
            "Si no cumple, indica qué le falta y pide reescribirlo."
        )
        response = await self.llm.acomplete(prompt)
        review = response.text.strip().lower()

        if "aprobado" in review or ev.attempts >= MAX_REVISIONS:
            return FinishTweetEvent(tweet=ev.tweet)
        else:
            return IterateTweetEvent(tweet=ev.tweet, review=review, attempts=ev.attempts + 1)

    # Paso 3: reescribir tweet según el feedback
    @step
    async def rewrite_tweet(self, ev: IterateTweetEvent) -> TweetEvent:
        print(f"#previous: {ev.tweet}")
        prompt = (
            f"Este tweet necesita mejoras: \"{ev.tweet}\"\n\n"
            f"Feedback: {ev.review}\n\n"
            "Reescribe el tweet para que cumpla con las condiciones."
        )
        response = await self.llm.acomplete(prompt)
        return TweetEvent(tweet=response.text.strip(), attempts=ev.attempts)

    # Paso 4: finalizar flujo
    @step
    async def send_tweet(self, ev: FinishTweetEvent) -> StopEvent:
        return StopEvent(result=f"✅ Tweet final aprobado (o máximo de intentos alcanzado):\n\n{ev.tweet}")


# --- Ejecución del flujo ---
async def main():
    flow = TweetFlow(timeout=6000, verbose=True)
    result = await flow.run(topic="inteligencia artificial en la educación")
    print(result)

asyncio.run(main())