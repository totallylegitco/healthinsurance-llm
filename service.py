import bentoml

runner = bentoml.models.get("fighthealthinsurance_model_v0.2:latest").to_runner()

svc = bentoml.Service(
    name="fighthealthinsurance_model_v0.2", runners=[runner]
)

@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
async def summarize(text: str) -> str:
    generated = await runner.async_run(text, max_length=3000)
    return generated[0]["summary_text"]
