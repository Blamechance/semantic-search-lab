from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.retrievers import KNNRetriever

import getpass
import os

# Load environment variables from .env file
load_dotenv()

# Choose embedding model and load API key: 
# embeddings = VoyageAIEmbeddings(model="voyage-3-large")

embeddings = VoyageAIEmbeddings(model="voyage-3-large", output_dimension=2048)  

if not os.environ.get("VOYAGE_API_KEY"):
    os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter API key for Voyage AI: ")

# Instantiate the vector store using your MongoDB connection string
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
  connection_string = os.getenv("ATLAS_CONNECTION_STRING"),        # MongoDB cluster URI
  namespace = "sample_mflix.embedded_movies",  # Database and collection name
  # embedding = VoyageAIEmbeddings(),                 # Embedding model to use
  embedding = embeddings, # re-use placeholder above for now
  index_name = "vector_index_test",                      # Name of the vector search index
  embedding_key="plot_embedding_voyage_3_large"
)

# Test retrieval using instantiated Atlas Vector Store:
