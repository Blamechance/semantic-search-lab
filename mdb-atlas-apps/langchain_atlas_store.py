from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch


import getpass
import os

# Load environment variables from .env file
load_dotenv()

# Choose embedding model and load API key: 
embeddings = VoyageAIEmbeddings(model="voyage-3-lite")

if not os.environ.get("VOYAGE_API_KEY"):
    os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter API key for Voyage AI: ")

# Instantiate the vector store using your MongoDB connection string
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
  connection_string = os.getenv("ATLAS_CONNECTION_STRING"),        # MongoDB cluster URI
  namespace = "sample_mflix.embedded_movies",  # Database and collection name
  # embedding = VoyageAIEmbeddings(),                 # Embedding model to use
  embedding = embeddings, # re-use placeholder above for now
  index_name = "langchain_vector_index",                      # Name of the vector search index
)

# Use the vector store as a retriever
retriever = vector_store.as_retriever()

# Define your query
query = "Are there any time travel movies including japanese people?"

# Print results
documents = retriever.invoke(query)
for doc in documents:
   print(doc)