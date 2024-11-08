import os
import requests
import re
import itertools
import backoff
from typing import Callable, Optional
from .utils import *

flatten = itertools.chain.from_iterable

treatment_regex = re.compile(
    r"""\s*(The|An|A)?\s*(parent|father|mother|patient|enrollee|member|provider)\s*[^.]*(requested|required|asked|requires|reimbursement|coverage|requesting|has)\s*[^.]*(of|for|medication|reimbursement|coverage|services)\s+(?P<treatment>\d*\w+.+?)\.""",
    re.IGNORECASE,
)
alt_treatment_regex = re.compile(
    r"""At issue\s*(in this case|)\s*(is|)\s*(whether|if)\s+(?P<treatment>\d*\w+.+?) (is|were|was) medically (necessary|indicated)""",
    re.IGNORECASE,
)
more_alt_treatment_regex = re.compile(
    r"""the requested (medication|treatment|service|procedure)\s+(?P<treatment>\d*\w+.+?) (is|were|was) (likely to be|medically necessary|medically indicated)""",
    re.IGNORECASE,
)

even_more_alt_treatment_regex = re.compile(
    r"""(Therefore|Thus|As such),[^.]*?\s+(an|a|the|that) (?P<treatment>\w+[^.]+?) (is|were|was|should be) (medically necessary|medically indicated|likely to be|authorized)""",
    re.IGNORECASE,
)

perscribed_regex = re.compile(
    r"""patients provider has prescribed the medication\s+(?P<treatment>[^.]+?).""",
    re.IGNORECASE,
)

wishes_to_regex = re.compile(
    r"""(wishes|desires|like) to (undergo|take)\s+(?P<treatment>[^.]+?).""",
    re.IGNORECASE,
)

health_plan_not_necessary_regex = re.compile(
    r"""The (Health Plan|Plan|Insurance Company) (determined the|determined|indicates) (?P<treatment>.+?) (is|was|were) not""",
    re.IGNORECASE,
)

almost_sketchy_regex = re.compile(
    r"""treatment[^.]*with\s+(?P<treatment>[^.]+?) (is|were|was)""", re.IGNORECASE
)

sketchy_regex = re.compile(
    r"""(requested|required|asked|requires|reimbursement|coverage|request|requesting)\s*[^.]*(for|medication|reimbursement|coverage|of)\s+(?P<treatment>\d*\w+.+?)\.""",
    re.IGNORECASE,
)

seeking_regex = re.compile(
    r"""is seeking (?P<treatement>) for (?P<diagnosis>[^.]+)""", re.IGNORECASE
)
admitted_regex = re.compile(
    r"""admitted to the hospital for (?P<diagnosis>[^.]+)""", re.IGNORECASE
)
recommended_regex = re.compile(
    r"""physicians recommended (?P<treatment>[^.]+)""", re.IGNORECASE
)

treatment_regexes = [
    treatment_regex,
    alt_treatment_regex,
    more_alt_treatment_regex,
    perscribed_regex,
    wishes_to_regex,
    health_plan_not_necessary_regex,
    almost_sketchy_regex,
    sketchy_regex,
    seeking_regex,
    recommended_regex,
]

diagnosis_regexes = [
    seeking_regex,
    admitted_regex,
]


def get_treatment_from_imr(imr):
    return get_treatment_from_findings_and_categories(
        imr["Findings"], imr["TreatmentCategory"], imr["TreatmentSubCategory"])

def get_treatment_from_findings_and_categories(
        findings: str, tc: str, tsc: str) -> Optional[str]:
    # See what we can find from regex
    for r in treatment_regexes:
        matches = r.search(findings)
        if matches is not None:
            t = matches.group("treatment")
            if t is not None and len(t) > 2:
                return t

    # No regex match fall back to input data (which is coarse AF)
    if tc == "Other":
        return None
    if tsc == "Other":
        tsc = ""
    return f"{tc}{tsc}"


def get_diagnosis_from_imr(imr):
    return get_diagnosis_from_findings_and_categories(
        imr["Findings"], imr["DiagnosisCategory"], imr["DiagnosisSubCategory"])

def get_diagnosis_from_findings_and_categories(
        findings: str, dc: str, dsc: str) -> Optional[str]:
    for r in diagnosis_regexes:
        matches = r.search(findings)
        if matches is not None:
            d = matches.group("diagnosis")
            if d is not None and len(d) > 2:
                return d
    # No regex match fall back to input data (which is coarse AF)
    if dc == "Other":
        return None
    if dsc == "Other":
        dsc = ""
    return f"{dc}{dsc}"

def extract_text(result):
    if result is None:
        return None
    if "generated_text" not in result[0]:
        return None
    return result[0]["generated_text"]

def generate_prompts_instruct(imr):
    def format_for_model(x):
        return f"<s>[INST]{x}[/INST]"

    return generate_prompts(imr, format_for_model=format_for_model)


# Specify the prompt order so we can chain them with previous versions.
prompt_order = ["treatment", "diagnosis", "reason_for_denial", "denial", "appeal", "medically_necessary", "studies", "patient_history", "patient_history_questions"]

def generate_prompts_from_imr(imr, format_for_model=lambda x: x):
    determination = imr["Determination"]
    treatment = None
    if "treatment_extracted" in imr:
        treatment = imr["treatment_extracted"]
    else:
        treatment = get_treatment_from_imr(imr)
    diagnosis = None
    if "diagnosis_extracted" in imr:
        diagnosis = imr["diagnosis_extracted"]
    else:
        diagnosis = get_diagnosis_from_imr(imr)    
    diagnosis = get_diagnosis_from_imr(imr)
    findings = imr["Findings"].strip("\n")
    grounds = imr["Type"]
    index = imr["ReferenceID"]
    return generate_prompts(
        determination=determination,
        treatment=treatment,
        diagnosis=diagnosis,
        findings=findings,
        grounds=grounds,
        index=index,
        format_for_model=format_for_model)

def generate_prompts(
        determination: Optional[str],
        treatment: Optional[str],
        diagnosis: Optional[str],
        findings: Optional[str],
        grounds: Optional[str],
        index: str,
        format_for_model: Callable[str, str]):
    treatment_extra = ""
    diagnosis_extra = ""
    if treatment is not None and treatment.lower() != "other":
        treatment_extra = f"We also guessed at treatment of {treatment}."
    if diagnosis is not None and diagnosis.lower() != "other":
        diagnosis_extra = f"We guessed at a diagnosis of {diagnosis}."

    prompts = {
        "denial": [
            format_for_model(
                f"""The independent medical review findings were {findings} and grounds for denial were {grounds}.{treatment_extra} Use this information to write the original insurance denial from the insurance company. Do not include any reference to the reviewers or their findings, instead focus on what the insurance company would have written denying the patients first claim. Keep in mind the denial would have been written before the independent review. Feel free to be verbose. You may wish to start your denial as a letter with \"Dear [Member];\""""),
            format_for_model(
                f"""Given the following medical reviewer findings:

{findings}{treatment_extra}{diagnosis_extra}
                Compose an initial rejection letter on behalf of the insurance company in response to a patient's request for medical coverage. Include specific details about the patient's case, addressing the reasons for denial without referencing any independent medical review findings. Ensure the letter is concise, professional, and clearly communicates the grounds for the denial. Focus on policy justifications, eligibility criteria, medical necessity, or any other relevant factors that would lead to the initial rejection. Omit any mention of the independent medical reviewers' assessments or findings as those happend later in the process.""")
        ],
        "appeal": [
            format_for_model(
                f"""The independent medical review findings were {findings} and grounds for denial were {grounds}.{treatment_extra}. In your response you are writing on your own on behalf (not that of a doctors office) and you do not have any credentials. Do not include any reference to the reviewers or their findings. Use this information to write the original appeal by the patient. Keep in mind the denial would be written before the appeal. Remember you are writing for yourself, not on behalf of anyone else. If any studies or guidelines support the medical necessity include them. Don't make up any references not found in the input. Feel free to be verbose and start your appeal with Dear [Insurance Company];"""),
            format_for_model(
                f"""Given the following medical reviewer findings:\n{findings}{treatment_extra}{diagnosis_extra}\n Do not include any information about the reviewers' findings. Instead, consider the patient's personal experience, medical history, and reasons for seeking the requested medical coverage. Craft the appeal to express the patient's perspective and emphasize their need for the requested medical intervention without referencing the independent medical review outcomes. Omit any mention of the independent medical reviewers' assessments or findings as those happend later in the process. Feel free to be verbose and write in the style of patio11 or a bureaucrat like sir humphrey appleby. Remember you are writing for yourself, not on behalf of anyone else. If any studies or guidelines from the reviewers support the medical necessity include them."""),
        ],
        "medically_necessary": [
            format_for_model(
                f"""Given the following medical review findings: {findings} and grounds for denial were {grounds}.{treatment_extra}. An earlier model suggested that the diagnosis was #diagnosis# and treatment #treatment#. Why was the treatment considered medically necessary? Don't refer to the reviewers findings directly instead write in a general fashion. For example if the reviewers found that facial feminization surgery was needed to treat gender dysphoria based on WPATH guidelines you would write something like: Facial feminization surgery is medically necessary for gender dysphoria per the WPATH guidelines. Do not refer to the reviewers qualifications or the reviewers themselves directly. If any studies or guidelines are referenced that support the medical necessity include them but don't make up new ones. Be concise (each word costs $200) and remember do not mention the reviewers."""),
        ],
        "studies": [
            format_for_model(
                f"""What studies/references are mentioned in {findings}?  Be concise (each word costs $2000) and do not mention the reviewers, if none say NONE. Provide each reference as a bullet on a new line. If an article is not explicilty mentioned by name in the source text above do not include it. Including an incorrect journal could result in someone not getting health care. If there is a journal article referenced but not by name or by auhtor write just the name of the journal don't guess. Remember be concise leave out things like \"the case summary mentions\" or anything that is not the references (like the final decision) -- just provide bulleted list of the references."""),
        ],
        "patient_history": [
            format_for_model(
                f"""Given the following medical review findings: {findings}{treatment_extra}. What were relevant factors of the patients history? Don't refer to the reviewers findings directly instead write in a general fashion. For example if the reviewers found the patient needed a brand name drug because the generics did not work you would write something like: Previous treatments including the frontline #nameofdrug# were not effective. Do not refer to the reviewers qualifications or the reviewers themselves directly. If you don't know any write NONE. It's expected you won't have a full history so only write NONE if you can't extract any relevant history. For example if you can extract previous treatments but not age you would just write the previous treatments."""),
        ],
        "patient_history_questions": [
            format_for_model(
                f"""Given the following medical review findings: {findings}{treatment_extra} and the patient history #patient_history# write a list of questions, one per line, which would allow you to have the required patient history to determine the above patient history and create an appeal for a denial. You are creating a form for the patient to answer. Do not nest entires."""),
        ],
        "reason_for_denial": [
            format_for_model(f"""Given the following medical review findings:  {findings} and grounds for denial were {grounds}.{treatment_extra}. What excuse did the insurance company use to deny the treatment? Some common reasons are medical necessary, STEP treatment required, experimental treatments, or a procedure being considered cosmetic. These are just examples though, insurance companies can deny care for many reasons. What was the reason here? Be concise and do not mention reviewer findings. Please summarize.""")
        ],
        "treatment": [
            format_for_model(
                f"""Based on the independent review findings: {findings}{treatment_extra}{diagnosis_extra}. You do not need to stick to our initial guess. Be consise. For example if the treatment was LINX just write LINX. What was the treatment, procedure, therapy, or surgery denied?""")
        ],
        "cpt_codes": [
            format_for_model(
                f"""Based on the independent review findings: {findings}{treatment_extra}{diagnosis_extra}. You do not need to stick to our initial guess. Be consise. For example if the treatment was LINX just write LINX. What was the CPT code(s) of the procedure, therapy, or surgery denied?""")
        ],
        "diagnosis": [
            format_for_model(
                f"""Based on the independent review findings: {findings}{treatment_extra}{diagnosis_extra}. You do not need to stick to our initial guess. Be consise, for example if the diagnosis was gastroesophageal reflux disease just write gastroesophageal reflux disease. What was the diagnosis (or NONE if there was none)?""")
        ],
        "icd10": [
            format_for_model(
                f"""Based on the independent review findings: {findings}{treatment_extra}{diagnosis_extra}. You do not need to stick to our initial guess. Be consise, for example if the diagnosis was gastroesophageal reflux disease just write gastroesophageal reflux disease. What was the ICD codes (or NONE if there was none)?""")
        ]
    }

    known = {}
    #if not is_unknown(treatment):
    #    del prompts["treatment"]
    #    known["treatment"] = treatment
    #if not is_unknown(diagnosis):
    #    del prompts["diagnosis"]
    #    known["diagnosis"] = diagnosis
    #if not is_unknown(grounds):
    #    del prompts["reason_for_denial"]
    #    known["reason_for_denial"] = grounds
    if not has_journal(grounds):
        del prompts["studies"]

    return (index, prompts, known)

def create_result_generator() -> Callable[[str, str, Optional[str], Optional[str]], str]:
    """Create the function which you can use for results."""
    backend = os.getenv("BACKEND_PROVIDER",
                        "https://api.perplexity.ai/chat/completions")
    print(f"Using backend {backend}")
    # Perplexity is an interesting backend for personal use.
    # The inference costs are a little high though for full training data
    # creation so look for whoever is cheapest when running in prod.
    # deepinfra was cheap when working on this last. Always check TOS
    # See https://artificialanalysis.ai/
    url = backend

    token = None
    if backend == "https://api.perplexity.ai/chat/completions":
        token = os.getenv("PERPLEXITY_API")
    else:
        token = os.getenv("SECRET_BACKEND_TOKEN")
    if backend == "https://api.deepinfra.com/v1/openai/chat/completions":
        token = os.getenv("DEEPINFRA_API")
    if token is None:
        raise Exception("Error no Token provided for inference.")

    max_delay = os.getenv("MAX_DELAY", None)
    if max_delay is not None:
        max_delay=int(max_delay)

    @backoff.on_exception(
        backoff.expo, requests.exceptions.RequestException, max_time=600
    )
    def make_request(model: str, prompt: str, previous_response=None, error=None) -> str:
        if max_delay is not None:
            time.sleep(random.randint(0, max_delay))
        messages = [{"role": "user", "content": prompt}]
        if previous_response is not None:
            messages.extend(
                [{"role": "assistant", "content": previous_response},
                 {"role": "user", "content": f"Please do better {error}"},
                 ]
                 )
        payload = {
            "model": model,
            "messages": messages,
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        print(f"Making request for {model} and {prompt}")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        response_text = response.json()["choices"][0]["message"]["content"]
        print(f"Prompt: {prompt}\nResponse text: {response_text}")
        return response_text
    return make_request


models = [
    #("mistral-7b-instruct", 0),
    #("openhermes-2-mistral-7b", 1),
    #("mistralai/Mixtral-8x7B-Instruct-v0.1", 3)
    #        ("mixtral-8x7b-instruct", 3),
    #("mixtral-8x22b-instruct", 4),
    #("dbrx-instruct", 5),
    #        ("llama-3.1-70b-instruct", 6),
    ("nvidia/Llama-3.1-Nemotron-70B-Instruct", 60),
]

relevant_columns = [
    "Determination",
    "TreatmentCategory",
    "TreatmentSubCategory",
    "DiagnosisCategory",
    "DiagnosisSubCategory",
    "Type",
    "Findings",
    "ReferenceID",
]

imr_data_input_path = "./data_sources/ca-independent-medical-review-imr-determinations-trends-utf8.csv"

make_request = create_result_generator()
