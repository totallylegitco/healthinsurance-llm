from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import runpod
import json
import time

model_name = "fighthealthinsurance_model_v0.2"
model = None
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
except Exception as e:
    print(f"Failed to load {e}")
    model_name = "/fighthealthinsurance_model_v0.2"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline(
     "text-generation", 
     model=model, 
     tokenizer = tokenizer, 
     torch_dtype=torch.bfloat16,
     device_map="auto"
 )


def generate_appeal(job):


    if "input" not in job:
        return {"error": f"No input in {job}"}

    input = job["input"]
    if "prompt" not in input:
        return {"error": f"No prompt in {job}"}
    
    prompt = input["prompt"]

    if not isinstance(prompt, str) or prompt == "":
        return {"error": "Prompt must not be none"}

    attempt = 0
    result = ""
    start_time = time.time()
    while result == "" and attempt < 3 and 40 > time.time()-start_time:
        print(f"Querying {prompt}")
        seqs = pipe(
            f"<s>[INST]{prompt}[/INST]",
            do_sample=True,
            max_new_tokens=200, 
            temperature=0.7, 
            top_k=50, 
            top_p=0.95,
            num_return_sequences=1,
        )
        result = seqs[0]['generated_text']

        print(f"Raw result {result}")
        if result.startswith(prompt):
            result = result.removeprefix(prompt)
        else:
            if "[/INST][/INST]" in result:
                result = result.split("[/INST][/INST]")[1]
            elif "[/INST]" in result:
                result = result.split("[/INST]")[1]
            else:
                print("Using vanilla result")

        print(f"Result: {result}")
        if result != "":
            return {"result": result, "prompt": prompt, "stuff": json.dumps(seqs) }
    return {"error": f"Could not generate for {prompt}"}



runpod.serverless.start({"handler": generate_appeal})
