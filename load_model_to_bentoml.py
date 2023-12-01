import transformers
import bentoml

model="fighthealthinsurance_model_v0.2"
task="text-generation"
tokenizer = transformers.AutoTokenizer.from_pretrained(model)


bentoml.transformers.save_model(
    task,
    transformers.pipeline(task, model=model, tokenizer=tokenizer),
    metadata=dict(model_name=model),
)
