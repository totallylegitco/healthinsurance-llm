import urllib
import json
import requests
import re
import unicodedata

magic_re = re.compile(
    r".*/(.*?)(MAGIC[0-9]|FARTS[0-9]*|farts[0-9]|EXTRACTED)_*(appeal|rejection|denial|json|medically_necessary)\d*.txt"
)


def training_cleanup_appeal(text):
    if text is None:
        return None
    return cleanup_appeal(text)


was_rejected = re.compile(
    r"(deneied|no additional treatment|not covered|not reimbursed|not eligible)",
    re.IGNORECASE,
)
invert_regex = re.compile(
    r"(is|are|were|be)\s*medically\s*(necessary|required)", re.IGNORECASE
)


def sketchy_sentence_filter(sentence):
    if "I am a" in sentence:
        return False
    if "agrees with the reviewer's findings" in sentence:
        return False
    if "The reviewer " in sentence:
        return False
    return True


def training_cleanup_rejection(text):
    if text is None:
        return None
    if re.search(was_rejected, text) is None:
        text = f"{text}. Your request is denied."
    if not "[MEMBER]" in text:
        text = f"Dear [MEMBER]; {text}."

    def mark_unnecessary(match):
        return f"{match.group(1)} not medically {match.group(2)}"

    text = re.sub(invert_regex, mark_unnecessary, text)
    return cleanup_denial(text)


def check_for_invalid_urls(data):
    urls = re.findall(r"(https?://\S+)", data)
    for u in urls:
        if not is_valid_url(u):
            return True
    return False


def load_record(filename):
    with open(filename, encoding="utf-8") as f:
        raw_data = f.read()
    data = parse_record(raw_data)
    lt = letter_type(filename)
    return cleanup_lt(lt, data)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch == "\n")


def parse_record(data):
    data = re.sub("<\s*/?\s*(PARAGRAPH|FREETEXT)\s*>", "", data)
    if "### Response:" in data:
        return data.split("### Response:")[1]
    else:
        return data


def fix_missing_quotes(json_string):
    # Find all JSON keys without quotes (no spaces allowed in keys)
    pattern_keys = r"([{,])\s*([a-zA-Z_]\w*)\s*:"
    fixed_json = re.sub(pattern_keys, r' \1"\2":', json_string)

    # Find all JSON values without quotes (including null) and spaces allowed in values
    pattern_values = r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:\s*([^,"}\]]+|null)'
    fixed_json = re.sub(pattern_values, r' "\1": \2', fixed_json)

    return fixed_json


def fix_missing_colons(json_string):
    # Find all places where a colon is missing between keys and values
    pattern = r'([{,])\s*([a-zA-Z_]\w*)\s+([^",}\]]+)'
    fixed_json = re.sub(pattern, r' \1"\2": \3', json_string)
    return fixed_json


# Example usage:
json_string = '{"name": John, "age": 30, country: null, "email": test@example.com}'
fixed_json = fix_missing_quotes(json_string)

maybe_bad_url_endings = re.compile("^(.*)[\.\:\;\,\?\>]+$")

common_bad_result = [
    "The page you are trying to reach is not available. Please check the URL and try again.",
    "The requested article is not currently available on this site."]

def is_valid_url(url):
    try:
        # Some folks don't like the default urllib UA.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
        }
        request = urllib.request.Request(url, headers=headers)
        result = urllib.request.urlopen(request)
        if ".pdf" not in url:
            result_text = result.read().decode('utf-8').lower()
            for bad_result_text in common_bad_result:
                if bad_result_text.lower() in result_text:
                    raise Exception(f"Found {bad_result_text} in {result_text}")
        return True
    except Exception as e:
        groups = maybe_bad_url_endings.search(url)
        if groups is not None:
            return is_valid_url(groups.group(1))
        else:
            print(f"Bad url {url} e {e} with no bad to strip")


def cleanup_json(data):
    """
    Load a json *ish* record. The LLM seems to not end the JSON records very often (e.g. missing }
    and trailing ,s instead. This is kind of janky but YOLO.
    """

    def de_json(l):
        try:
            return json.loads(l)
        except:
            return re.sub("^\s*(.*?)\s*$", "\1", l).rstrip('","').lstrip('"')

    data = data.replace("None", "null")
    data = remove_control_characters(data)
    if data.endswith(","):
        data = data.rstrip(",")
    data = data.replace(",}", "}")

    # Handle some missing quotes if needed.
    try:
        return json.loads(data)
    except Exception as e:
        data = fix_missing_quotes(data)

    try:
        return json.loads(data)
    except Exception as e1:
        try:
            return json.loads(data + "}")
        except Exception as e2:
            try:
                return json.loads(data + '"}')
            except Exception as e3:
                result = {}
                for line in data.split("\n"):
                    if ":" in line:
                        elems = line.split(":")
                        result[de_json(elems[0])] = de_json(":".join(elems[1:]))
                if "condition" in result and "approval_reason" in result:
                    return result
                else:
                    return None

swaps = {
    "general": [
        ("Note that the information is inferred based on the reviewer's findings, but the language used is general rather than directly referencing the reviewer's findings.", ""),
        ("Based on the information provided, the following factors from the patient's history appear to have been relevant in the determination of", ""),
        ("Based on the information provided, ", ""),
        ("and the reviewer's clinical experience and expertise in treating such cases", ""),
    ],
    "patient_history": [
        ("There is no information provided about the patient's demographic details.", "") 
    ],
    "diagnosis": [
        ("The diagnosis is ", ""),
    ],
    "denial": [
                (
            "The Health Plans denial was overturned due to the reviewers determining that the requested services were likely to be more beneficial for treatment of the enrollees medical condition than any available standard therapy.",
            "",
        ),
        ("independent medical review findings were nature of statutory criteria/case summary:", ""),
        ("will be overturned.*", ""),
        ("the independent medical reviewer", "we"),
        ("The physician reviewer", "we"),
        ("We always say no to surgeries.", ""),
        ("The reason was originally denied was", "Your request is denied because"),
        ("Therefore, the Health Plans denial should be overturned.", ""),
        ("We thank the reviewer for their assessment of this case.", ""),
        ("The reviewers determined that", "We do not believe that"),
        ("should be overturned", "should stand."),
        (
            "that denying coverage for this treatment would be inappropriate",
            "we have chosen to deny coverage",
        ),
        ("it is not possible to deny or approve", "we must deny"),
        ("should be granted coverage", "will not be granted coverage"),
        ("patient's condition warrants", "patient's condition does not warrant"),
        ("deny the denial", "deny the "),
        ("  ", " "),
        ("As an AI language model", ""),
        ("\.\.\.", "."),
        ("\.\.", "."),
        ("by our independent medical representative", "by us"),
        ("is medically necessary", "is not medically necessary"),
        ("Review findings: The", ""),
        ("Review findings:", ""),
        ("The physician reviewer found that", "We determined that"),
        ("independent medical review has determined", "we have determined"),
        ("was indicated for this patient", "was not indicated for this patient"),
        ("the requested .* is appropriate for this patient", "the request has been denied for this patient"),
        ("Final Result: The reviewers determined that.*", ""),
        ("reviewers determined that.*", ""),
        ("findings: .* physician reviewers.*", ""),
        ("Thank you for providing me with this information.", ""),
        ("Consequently, the Health Plan's denial should be overturned." , ""),
        ("According to recent medical literature, [^\.]*.", ""),
    ],
    "appeal": [
        ("Dear Independent Medical Reviewers", "Dear [Insurance Company];"),
        ("coverage has been approved.", "coverage should be approved."),
        ("The final determination was that ", ""),
        ("We reviewed the medical records of patients", "In patients"),
        ("We conducted a retrospective cohort", "In a"),
        ("< / FREETEXT > < / ABSTRACT > ▃", ""),
        ("< / FREETEXT >", ""),
        ("< / ABSTRACT >", ""),
        ("  ", " "),
        ("\.\.", "."),
        (
            "trans men have well-developed jawlines",
            "trans women have well-developed jawlines",
        ),
        ("The provided denial was overturned", "The denial should be overturned"),
        (
            "Therefore, the provided denial should be upheld.",
            "Therefore, the denial should be overturned.",
        ),
        ("who is seeking authorization and coverage of", "I am seeking authorization and coverage of"),
        ("Therefore, it may not be covered by insurance", "Regardless, it should be covered"),
        ("Dear \[Medical Necessity\]", "Dear \[Insurance Company\],"),
        ("to the independent medical review findings", "to your decision"),
        ("Thank you for providing me with this information." , ""),
        ("The independent medical review findings of.*?:", ""),
        ("According to the independent medical review, ", ""),
        ("Hence,  concluded", ""),
    ]        
    }


def cleanup_lt(lt, data):
    if lt == json:
        return cleanup_json(data)
    my_swaps = {}
    my_swaps = swaps["general"]
    if lt in swaps:
        my_swaps += swaps[lt]

    old_data = ""
    while old_data != data:
        old_data = data
        for o, r in my_swaps:
            data = re.sub(o, r, data, flags=re.IGNORECASE)

    return data


# Load some strings we know the current model puts in appeals that are bad right away
with open("bad_appeal_strings.txt") as f:
    bad_appeal_strings = list(map(lambda f: f.lower(), f.read().split("\n")))

with open("bad_medically_necessary_strings.txt") as f:
    bad_medically_necessary_strings = list(map(lambda f: f.lower(), f.read().split("\n")))

with open("bad_treatment_strings.txt") as f:
    bad_treatment_strings = list(map(lambda f: f.lower(), f.read().split("\n")))


# Load some strings we know the current model puts in rejections that are bad right away
with open("bad_rejection_strings.txt") as f:
    bad_rejection_strings = list(map(lambda f: f.lower(), f.read().split("\n")))

bad_strings_dict = {
    "appeal": bad_appeal_strings,
    "rejection": bad_rejection_strings,
    "medically_necessary": bad_medically_necessary_strings,
    "treatment": bad_treatment_strings}

def check_record(record):
    response_type = letter_type(record)
    return not check_for_bad_file(response_type, record)

def check_for_bad_file(response_type, target):
    with open(target, 'r') as file:
        data = file.read().replace('\n', '')
        return check_for_bad(response_type, data)

def check_for_bad(response_type, data):
    ld = data.lower()
    if response_type in bad_strings_dict.keys():
        bad_strings = bad_strings_dict[response_type]
        for b in bad_strings:
            if b != "" and b.lower() in data:
                print(f"Rejecting {data} for {response_type} as it contains {b}")
                return True
            return False
    else:
        return False


def check_for_bad_appeal(data):
    return check_for_bad("appeal", data)


def check_for_bad_rejection(data):
    return check_for_bad("rejection", data)


def not_none(i):
    return i is not None


def file_name_to_magic_score(filename):
    # Newer models get a higher bias, humans get most.
    groups = magic_re.search(filename)
    if groups is not None:
        g = groups.group(2)
        # Humans get max bonus
        if "MAGIC" not in g and "EXTRACTED" not in g:
            return 1000000
        else:
            try:
                return int(re.findall(r'\d+-', g)[0])
            except:
                return 0
        

def file_name_to_case(filename):
    groups = magic_re.search(filename)
    if groups is not None:
        return groups.group(1)
    else:
        print(f"No group in {filename}")
        return None

def letter_type(filename):
    groups = magic_re.search(filename)
    if groups is not None:
        g = groups.group(3)
        if g == "denial":
            return "rejection"
        print(f"g is {g}")
        return g
    else:
        print(f"No group in {filename}")
        return None
