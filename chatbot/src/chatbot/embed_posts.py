import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading blog post documents.")

loader = DirectoryLoader(
   "./chatbot/knowledge/blog_examples",
   glob="**/*.txt",
   loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
   show_progress=True
)

documents = loader.load()
if not documents:
   print("No documents found in specified directory.")
   exit()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
if not docs:
   print("No text chunks created from document.")
   exit()
   
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if not embedding_model:
   print("Failed to load embedding model")
   exit()
   
vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory="./chatbot/vectorstore")

print(f"✅ Successfully embedded {len(docs)} chunks into vectorstore.")
