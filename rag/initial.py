from llama_index import download_loader, StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, set_global_tokenizer, ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from glob import glob
from dataset_tools.utils import load_record
import os
from transformers import AutoTokenizer
import time
from itertools import chain
import backoff
import requests

flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]


print("Loading LLM (ish)")

local_llm = os.getenv(
    "LOCAL_MODEL",
    os.getenv("LLM", "/TotallyLegitCo/fighthealthinsurance_model_v0.3"))
global_llm = os.getenv(
    "GLOBAL_MODEL",
    os.getenv("LLM", "TotallyLegitCo/fighthealthinsurance_model_v0.3"))

llm_server_base = os.getenv("OPENAI_BASE_API")

print(f"Using {local_llm} / {global_llm} with backend {llm_server_base}")

try:
    set_global_tokenizer(
        AutoTokenizer.from_pretrained(global_llm).encode
    )
except:
    time.sleep(5)
    set_global_tokenizer(
        AutoTokenizer.from_pretrained(global_llm).encode
    )

llm = OpenAILike(
    model=local_llm,
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=llm_server_base)

print(f"WTTF: {llm.api_base}")
#print(llm.complete("San Francisco:"))

sentence_context = ServiceContext.from_defaults(
    llm = llm,
    embed_model="local",
)

print("Downloading loaders (e.g. random untrusted code from the web?)")

PubmedReader = download_loader("PubmedReader")

PDFReader = download_loader("PDFReader")

print("Running the loaders (see above). Fingers crossed.")

pubmed_loader = PubmedReader()

pdf_loader = PDFReader()

def load_pdf_docs():

    def load_pdf_doc(filename):
        print(f"Loading pdf doc {filename}")
        return pdf_loader.load_data(filename)

    pdf_docs = list(flat_map(load_pdf_doc, glob("data_sources/*.pdf")))

    return pdf_docs


pdf_docs = load_pdf_docs()

print("Constructing pubmed queries")


def echo(x):
    print(x)
    return x


def load_pubmed_docs():
    # For now do nothing
    return []
    @backoff.on_exception(
        backoff.expo, requests.exceptions.RequestException, max_time=600
    )
    def load_for_query(q):
        return pubmed_loader.load_data(q)

    treatments = set(map(echo, map(load_record, glob("generated-llm-data/*treatment.txt"))))
    diagnosis = set(map(echo, map(load_record, glob("generated-llm-data/*diagnosis.txt"))))

    queries = treatments.union(diagnosis)

    pubmed_docs = list(flat_map(load_for_query, queries))


pubmed_docs = load_pubmed_docs()

docs = pdf_docs + pubmed_docs


print("Ok party time!")

index = VectorStoreIndex.from_documents(docs, show_progress=True, service_context=sentence_context)

index.storage_context.persist()
