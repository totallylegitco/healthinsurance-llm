from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import time


class AppealGenerator:
    def __init__(self, load_in_8bit=False):
        model_name = "fighthealthinsurance_model_v0.2"
        self.model = None
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", load_in_8bit=load_in_8bit
            )
        except Exception as e:
            print(f"Failed to load {e}")
            model_name = "/fighthealthinsurance_model_v0.2"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", load_in_8bit=load_in_8bit
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate_appeal(self, job):
        print(f"Initial generate_appeal called with job {job}.")

        if "input" not in job:
            return {"error": f"No input in {job}"}

        input = job["input"]
        if "prompt" not in input:
            return {"error": f"No prompt in {job}"}

        prompt = input["prompt"]

        if not isinstance(prompt, str) or prompt == "":
            return {"error": "Prompt must not be none"}
        return self.generate_appeal_str(prompt)

    def generate_appeal_str(self, prompt):
        attempt = 0
        result = ""
        start_time = time.time()
        while result == "" and attempt < 3 and 40 > time.time() - start_time:
            print(f"Querying {prompt}")
            seqs = self.pipe(
                f"<s>[INST]{prompt}[/INST]",
                do_sample=True,
                max_new_tokens=200,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
            )
            result = seqs[0]["generated_text"]

            print(f"Raw result {result}")
            if "The reviewer determined that the requested" in result:
                print("Skipping...")
                result = ""
            if result.startswith(prompt):
                result = result.removeprefix(prompt)
            else:
                if "[ASN]" in result:
                    result = result.split("[ASN]")[1]
                elif "[OUT]" in result:
                    result = result.split("[OUT]")[1]
                elif "[/INST][/INST]" in result:
                    result = result.split("[/INST][/INST]")[1]
                elif "[/INST]" in result:
                    result = result.split("[/INST]")[1]
                else:
                    print("Using vanilla result")

            print(f"Result: {result}")
            if result != "":
                return {"result": result, "prompt": prompt, "stuff": json.dumps(seqs)}
            return {"error": f"Could not generate for {prompt}"}
