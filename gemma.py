# -*- coding: utf-8 -*-
"""Gemma.ipynb
"""

# !pip install git+https://github.com/huggingface/transformers.git
# !pip install --upgrade torch datasets accelerate peft bitsandbytes trl
# !pip install pdfminer.six langchain sentence-transformers chromadb cohere

import transformers
import torch
from transformers import AutoTokenizer,BitsAndBytesConfig,AutoModelForCausalLM


import chromadb
from chromadb.config import Settings

from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# Model Used and token for access
model_name='google/gemma-2b-it'
token = "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"

# Loading model
model_config = transformers.AutoConfig.from_pretrained(model_name,token=token)

# Tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,token=token)
tokenizer.padding_side = "right"


# Quantization Settings
use_4bit = True

bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Load 4 Bit -> Reduces model from 32 to 4 bit -> Hence reduces memory usage
# bnb_4bit_quant_type -> Type of 4 bit quantization here normal float 4
# bnb_4bit_compute_dtype -> Datta Type for computation within the model
# bnb_4bit_use_double_quant -> Makes the model undergo double quantization where the first quantization reduces the format to a lower value
# And second one which changes to target value

bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)

# MOdel

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    quantization_config=bnb_config,
)


# Pipeline for text generation
text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=2048,
    output_scores=True
)

GemmaLLM = HuggingFacePipeline(pipeline=text_generation_pipeline)

loader = TextLoader("/content/transcript.txt",encoding="utf8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)



model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=GemmaLLM,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

query = "What were the main topics in the State of the Union in 2023? Summarize. Keep it under 200 words."
qa.run(query)