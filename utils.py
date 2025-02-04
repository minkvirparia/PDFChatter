import os
import torch
from huggingface_hub import login
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

def configure_langchain_environment():
    """
    Configure environment variables for LangChain and Hugging Face API from environment.
    """
    os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2', 'true')
    os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT', 'pdf-chatter')


def authenticate_huggingface():
    """
    Authenticate Hugging Face API using the key from the environment.
    """
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise ValueError("Hugging Face API key not found. Please set it in the environment variables.")
    login(api_key)


def load_tokenizer(model_name: str):
    """
    Load the tokenizer for the specified model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str, quantization_config):
    """
    Load the causal language model with given quantization configuration.
    """
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )

def load_pipeline(_model, _tokenizer):
    """
    Load the text generation pipeline with the specified model and tokenizer.
    """
    return pipeline(
        task="text-generation",
        model=_model,
        tokenizer=_tokenizer,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=1000,
    )

def create_vectorstore(_documents):
    """
    Create a vector store using Chroma and Hugging Face embeddings.
    """
    return Chroma.from_documents(
        documents=_documents,
        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'),
        persist_directory="chromadb_storage",
    )
