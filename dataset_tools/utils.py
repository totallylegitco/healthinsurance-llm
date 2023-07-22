import re

def training_cleanup_appeal(text):
    if text is None:
        return None
    return cleanup_appeal(text)

was_rejected = re.compile(r"(deneied|no additional treatment|not covered|not reimbursed|not eligible)", re.IGNORECASE)
invert_regex = re.compile(r"(is|are|were|be)\s*medically\s*(necessary|required)", re.IGNORECASE)

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




def letter_type(filename):
    if filename.endswith("_rejection.txt"):
        return "rejection"
    elif filename.endswith("json.txt"):
        return "json"
    else:
        return "appeal"

def check_record(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read().lower()
        if letter_type(filename) == "appeal":
            return not (check_for_bad_appeal(data) or check_for_invalid_urls(data))
        else:
            return not check_for_bad_rejection(data)


def check_for_invalid_urls(data):
    urls = re.findall(r'(https?://\S+)', data)
    for u in urls:
        try:
            response = requests.get(u)
        except Exception as e:
            # For cases where the url is at the end of a sentence
            # Try and see if it exists without the last after . part
            try:
                u2 = re.sub(r"(.*)\..*?$", r"\1", u)
                if (u2 != u):
                    response = requests.get(u2)
                else:
                    return True
            except Exception as e:
                print(f"Failed to get {u} (or {u2}) dropping from candidates.")
                return True
    return False


def load_record(filename):
    with open(filename, encoding="utf-8") as f: raw_data = f.read()
    data = parse_record(raw_data)
    if letter_type(filename) == "appeal":
        return cleanup_appeal(data)
    elif letter_type(filename) == "rejection":
        return cleanup_denial(data)
    elif letter_type(filename) == "json":
        return cleanup_json(data)
    else:
        return data


def parse_record(data):
    if "### Response:" in data:
        return data.split("### Response:")[1]
    else:
        return data

def cleanup_json(data):
    try:
        return json.loads(data)
    except:
        try:
            return json.loads(data +"\"}")
        except:
            return None


def cleanup_denial(data):
    swap = [
        ("The Health Plans denial was overturned due to the reviewers determining that the requested services were likely to be more beneficial for treatment of the enrollees medical condition than any available standard therapy.", ""),
        ("We always say no to surgeries.", ""),
        ("The reason was originally denied was", "Your request is denied because"),
        ("Therefore, the Health Plans denial should be overturned.", ""),
        ("We thank the reviewer for their assessment of this case.", ""),
        ("The reviewers determined that", "We do not believe that"),
        ("should be overturned", "should stand."),
        ("that denying coverage for this treatment would be inappropriate", "we have chosen to deny coverage"),
        ("it is not possible to deny or approve", "we must deny"),
        ("should be granted coverage", "will not be granted coverage"),
        ("patient's condition warrants", "patient's condition does not warrant"),
        ("deny the denial", "deny the "), 
        ("  ", " "),
        ("As an AI language model", ""),
        ("\.\.\.", "."),
        ("\.\.", "."),
    ]
    for o, r in swap:
        data = re.sub(o, r, data)

    return data


def cleanup_appeal(data):
    swap = [
        ("As an AI language model", ""),
        ("\W+ of \W+ reviewers (determined|found) that", "It is believed that "),
        ("\W+ of \W+ reviewers (determined|found)", "It is believed "),
        ("the reviewer", " "),
        ("Therefore, the reviewer concludes that", "That should "),
        ("coverage has been approved.", "coverage should be approved."),
        ("Sincerely, Doctor", "Sincerely, [YOURNAME]"),
        ("The final determination was that ", ""),
        ("We reviewed the medical records of patients", "In patients"),
        ("We conducted a retrospective cohort", "In a"),
        ("< / FREETEXT > < / ABSTRACT > ▃", ""),
        ("< / FREETEXT >", ""),
        ("< / ABSTRACT >", ""),
        ("  ", " "),
        ("\.\.", "."),
        ("was overturned", "should be overturned"),
        ("is not medically necessary", "is medically necessary"),
        ("this surgery is not necessary", "this surgery is necessary"),
        ("and is solely based on appeasing an overly-sufiisticated view of what women look like.",
         ""),
        ("trans men have well-developed jawlines",
         "trans women have well-developed jawlines"),
        ("The requested procedures are aesthetic procedures to make the patient more attractive in her new gender.",
         ""),
        ("The provided denial was overturned", "The denial should be overturned"),
        ("Therefore, the provided denial should be upheld.", "Therefore, the denial should be overturned."),
        ("denial should be upheld", "denial should be overturned"),
        ("did not have improved mental health outcomes compared to those who had",
         "have improved mental health outcomes compared to those who had"),
    ]
    for o, r in swap:
        data = re.sub(o, r, data)

    return data

# Load some strings we know the current model puts in appeals that are bad right away
with open("bad_appeal_strings.txt") as f: bad_appeal_strings = list(map(lambda f: f.lower(), f.read().split("\n")))

# Load some strings we know the current model puts in rejections that are bad right away
with open("bad_rejection_strings.txt") as f: bad_rejection_strings = list(map(lambda f: f.lower(), f.read().split("\n")))


def check_for_bad_appeal(data):
    for b in bad_appeal_strings:
        if b != "" and b in data:
            return True
    return False

def check_for_bad_rejection(data):
    for b in bad_rejection_strings:
        if b != "" and b in data:
            return True
    return False


def not_none(i):
    return i is not None
