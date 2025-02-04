import time
import numpy as np
import streamlit as st
from typing import List
from operator import itemgetter

from sentence_transformers import CrossEncoder
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from utils import authenticate_huggingface, load_tokenizer, load_model, load_pipeline, create_vectorstore, configure_langchain_environment


# Load environment variables
MODEL_NAME = os.getenv("MODEL_NAME")

authenticate_huggingface()
configure_langchain_environment()

# Main application function
def main():
    st.markdown("""<h1 style='text-align: center; color: black;'>PDF Chatter</h1>""", unsafe_allow_html=True)

    authenticate_huggingface()

    # Tokenizer
    model_name = MODEL_NAME
    tokenizer = load_tokenizer(model_name)

    # BitsandBytes Configs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=False,
    )

    # Model Loading
    model = load_model(model_name, bnb_config)

    pipe = load_pipeline(_model=model, _tokenizer=tokenizer)
    mistral_llm = HuggingFacePipeline(pipeline=pipe)

    st.write("Model Loaded!!")

    # PDF File Upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if not uploaded_file:
        st.warning("Please upload a PDF file to proceed.")
        return

    # Document Loader
    try:
        loader = PyPDFLoader(uploaded_file)
        docs = loader.load()
        st.write("PDF Loaded!!")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    try:
        all_splits = text_splitter.split_documents(docs)
        st.write("Text Splitter Completed!!")
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return

    try:
        # Retriever
        vectorstore = create_vectorstore(_documents=all_splits)
        retriever = vectorstore.as_retriever()
        st.write("VectorDB Initialized!!")
    except Exception as e:
        st.error(f"Error initializing retriever: {e}")
        return

    # Multi-Query Retriever Setup
    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""

        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return list(filter(None, lines))  # Remove empty lines

    output_parser = LineListOutputParser()

    TEMPLATE = """You are an AI language model assistant. Your task is to generate three
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines. Original question: {question}"""

    QUERY_PROMPT = PromptTemplate(input_variables=["question"], template=TEMPLATE)

    llm_chain = QUERY_PROMPT | mistral_llm | output_parser

    multi_query_retriever = MultiQueryRetriever(retriever=retriever, llm_chain=llm_chain, parser_key="lines")

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512)

    user_question = st.text_input("Ask a Question from the PDF.")


    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            try:
                unique_docs = multi_query_retriever.invoke(user_question)

                document_text = [doc.page_content for doc in unique_docs]
                response = [[user_question, doc_text] for doc_text in document_text]

                scores = cross_encoder.predict(response)
                if len(scores) == 0:
                    st.error("No scores generated!")
                    return

                context = np.argsort(scores)[::-1][0]

                template = """Answer the following question based on this context: {context} Question: {question} """

                prompt = ChatPromptTemplate.from_template(template)
                final_rag_chain = prompt | mistral_llm | StrOutputParser()

                result = final_rag_chain.invoke({"context": context, "question": user_question})

                st.write(result)
            except Exception as e:
                st.error(f"Error during processing: {e}")

if __name__ == "__main__":
    main()