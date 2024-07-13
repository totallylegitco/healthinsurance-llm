from llama_index.core import Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import set_global_tokenizer
from glob import glob
from dataset_tools.utils import load_record
import os
from transformers import AutoTokenizer
import time
from itertools import chain
import backoff
import requests
from .fullpubmedreader import FullPubMedReader

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

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900

def load_pubmed_docs():
    reader = FullPubMedReader()
    return reader.load_data("data_sources/pubmed/ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/")

print(f"Loading PubMed data")

docs = load_pubmed_docs()


print("Ok party time!")

index = VectorStoreIndex.from_documents(docs, show_progress=True, service_context=sentence_context)

index.storage_context.persist()
