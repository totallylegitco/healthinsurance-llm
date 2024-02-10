from llama_index import download_loader, StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, set_global_tokenizer
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from glob import glob
from dataset_tools.utils import load_record
import os
from transformers import AutoTokenizer
import time
from itertools import chain

print("Loading LLM (ish)")

local_llm = os.getenv(
    "LOCAL_MODEL",
    os.getenv("LLM", "/TotallyLegitCo/fighthealthinsurance_model_v0.3"))
global_llm = os.getenv(
    "GLOBAL_MODEL",
    os.getenv("LLM", "TotallyLegitCo/fighthealthinsurance_model_v0.3"))

try:
    set_global_tokenizer(
        AutoTokenizer.from_pretrained(global_llm).encode
    )
except:
    time.sleep(5)
    set_global_tokenizer(
        AutoTokenizer.from_pretrained(global_llm).encode
    )

llm = OpenAILike(model=local_llm)

print("Downloading loaders (e.g. random untrusted code from the web?)")

PubmedReader = download_loader("PubmedReader")

PDFReader = download_loader("PDFReader")

print("Running the loaders (see above). Fingers crossed.")

pubmed_loader = PubmedReader()

pdf_loader = PDFReader()

def load_pdf_doc(filename):
    print(f"Loading pdf doc {filename}")
    return pdf_loader.load_data(filename)

pdf_docs = list(map(load_pdf_doc, glob("data_sources/*.pdf")))

print("Constructing pubmed queries")

treatments = set(map(load_record, glob("generated-llm-data/*treatment.txt")))
diagnosis = set(map(load_record, glob("generated-llm-data/*diagnosis.txt")))

queries = treatments.union(diagnosis)

pubmed_docs = list(map(lambda x: pubmed_loader(x), queries))

docs = pdf_docs + pubmed_docs


print("Ok party time!")

index = VectorStoreIndex.from_documents(docs, show_progress=True)

index.storage_context.persist()
