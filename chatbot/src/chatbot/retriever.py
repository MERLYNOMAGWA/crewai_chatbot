import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

VECTORSTORE_DIR = "./chatbot/vectorstore"

def load_vectorstore():
   """Load the persisted Chroma vectorstore with embeddings."""
   embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
   return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embedding_model)

def retrieve_similar_docs(query: str, k: int = 3):
   """Search the vectorstore for documents similar to the query."""
   vectorstore = load_vectorstore()
   retriever = vectorstore.as_retriever(search_kwargs={"k": k})
   results = retriever.invoke(query)
   return results

# Example
if __name__ == "__main__":
   query = input("Enter your query: ")
   docs = retrieve_similar_docs(query)
   print("\nTop matching blog chunks:\n")
   for i, doc in enumerate(docs, 1):
      print(f"[{i}] {doc.page_content}\n")
