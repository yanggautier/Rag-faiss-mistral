import os
import sys
import json
import faiss
import shutil
import argparse
import numpy as np
from uuid import uuid4
from datetime import datetime
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from app.documents import *


EMBEDDING_DIM = 1024
N_CELLS = 10
N_PROBE = 5


# Définition de l'état    
class State(TypedDict):
    query: str
    documents: List[dict]
    ids: List[uuid4]
    results: List[Document]
    response: str
    chat_history: list
    memory: dict

def create_optimized_index(dimension, embeddings, documents, n_lists=100):
    """Create and train an optimized FAISS index with documents to be indexed
    
    Parameters:
    dimension (int): Vector dimension
    embeddings: Embedding model
    documents (list): Liste de Document Langchain pour l'indexation
    n_lists (int): Number of Voronoi cells
    
    Returns:
    index: Trained FAISS index,
    vectors: Document vectors for adding to index
    """
    # Create index
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, n_lists)
    
    # Create vectors for all documents
    texts = [doc.page_content for doc in documents]
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors).astype('float32')
    
    # Train the index with same vectors we'll add
    index.train(vectors)
    
    return index, vectors

class FaissVectorStore:
    """
    Une classe pour gérer un vectorstore FAISS avec mémoire de conversation.
    """
    def __init__(self, embeddings, index_file, index, llm):
        """Initialise le vectorstore avec les configurations par défaut."""
        self.index_file = index_file
        self.embeddings = embeddings
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.memory_file = "conversation_memory.json"
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                memory_data = json.load(f)
                for interaction in memory_data:
                    self.memory.save_context(
                        {"input": interaction["input"]},
                        {"output": interaction["output"]}
                    )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es un assistant qui aide à répondre aux questions en utilisant le contexte fourni."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{context}\n\nQuestion: {query}")
        ])

        self.llm = llm

        # Créer la chaîne de documents
        self.doc_chain = create_stuff_documents_chain(
            self.llm,
            self.prompt
        )

        self.workflow = self._create_search_graph()

    
    def _create_search_graph(self):
        """Création de graphe pour intéragir avec le bot."""
        workflow = StateGraph(State)
        workflow.add_node("search", self._search_node)
        workflow.add_node("process_memory", self._process_memory_node)
        workflow.add_node("process_results", self._process_results_node)
        
        workflow.set_entry_point("search")
        workflow.add_edge("search", "process_memory")
        workflow.add_edge("process_memory", "process_results")
        
        return workflow.compile()

        
    def _search_node(self, state: State) -> State:
        """Effectue une recherche de similarité dans le vectorstore FAISS."""
        try:
            results = self.vector_store.similarity_search(state["query"])
            state["results"] = results if results else []
            return state
    
        except Exception as e:
            state["results"] = []
            print(f"Erreur lors de la recherche: {str(e)}")
            return state

    def _process_memory_node(self, state: State) -> State:
        """Ajouter la requête et le résultat à la mémoire.

        Parameters:
        state (State): State

        Returns:
        state: retourne un state
        """
        result_text = state["results"][0].page_content if state["results"] else "Pas de résultat"

        self.memory.save_context(
            {"input": state["query"]},
            {"output": result_text}
        )
        
        # Récupérer l'historique des conversations
        memory_data = self.memory.load_memory_variables({})
        state["chat_history"] = memory_data["chat_history"]
        state["memory"] = {
            "last_updated": datetime.now().isoformat(),
            "conversation_length": len(memory_data["chat_history"])
        }

        new_interaction = {
            "input": state["query"],
            "output": state["results"][0].page_content if state["results"] else "Pas de résultat"
        }
        
        try:
            with open(self.memory_file, 'r') as f:
                memory_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            memory_data = []
            
        memory_data.append(new_interaction)
        
        with open(self.memory_file, 'w') as f:
            json.dump(memory_data, f)
    
        return state
    
    def _process_results_node(self, state: State) -> State:
        """Traiter le résultat de recherche.

        Parameters:
        state (State): State

        Returns:
        state: retourne un state
        """
        results = state.get("results", [])

        chat_history = state.get("chat_history", [])
        chain_input = {
            "query": state["query"],
            "context": results,  
            "chat_history": chat_history
        }

        try:
            response = self.doc_chain.invoke(chain_input)
            state["response"] = {
                "current_answer": response,
                "chat_history": chat_history,
                "context": {"total_interactions": len(chat_history)}
            }

            # Sauvegarde de requête et réponse dans le mémoire
            self.memory.save_context(
                {"input": state["query"]},
                {"output": response}
            )

        except Exception as e:
            print(f"Erreur inattendue : {e}")
            state["response"] = {
                "current_answer": f"Erreur inattendue - {e}",
                "chat_history": chat_history,
               "context": {"total_interactions": len(chat_history)}
            }

        return state

    def process_query(self, query: str):
        """Traite une requête en utilisant le workflow LangGraph.
       
        Parameters:
        query (str): requête pour envoyer au chatbot

        Returns:
        state: retourne un state
        """
        initial_state = {
            "query": query,
            "documents": [],
            "results": [],
            "response": "",
            "chat_history": [],
            "memory": {}
        }
        return self.workflow.invoke(initial_state)

    def add_documents(self, documents):
        """Ajoute des documents au vectorstore FAISS.
               
        Parameters:
        documents (Document): une liste de documents Langchain
        """
        if not documents:
            raise ValueError("Liste de documents ne peuvent pas être vides")
        
        ## TODO récupérer les documents en local
        self.vector_store.add_documents(documents)
        self.vector_store.save_local(self.index_file)
        print(f"{len(documents)} documents ajoutés avec succès.")


    def verify_indexing(self, original_df):
        """Verify all events are properly indexed
        
        Parameters:
        vector_store: FAISS vector store
        original_df: Original DataFrame with events
        
        Returns:
        bool: True if verification passes
        """
        # Get all document IDs from vector store
        all_docs = self.vector_store.docstore._dict
        indexed_ids = set(doc.metadata["id"] for doc in all_docs.values())
        
        # Compare with original DataFrame
        original_ids = set(original_df["uid"])
        missing_ids = original_ids - indexed_ids
        
        if missing_ids:
            print(f"Warning: {len(missing_ids)} events not indexed")
            return False
        return True

    def clear_memory(self):
        """Efface l'historique des conversations stocké en mémoire."""
        self.memory.clear()

    def load_vectorstore(self, my_index_path):
        """Charge un vectorstore existant depuis un chemin spécifié."""
        self.index = my_index_path
        self.vector_store = FAISS.load_local(
            self.index, self.embeddings, allow_dangerous_deserialization=True
        )

    def get_memory_variables(self):
        """Récupère toutes les variables de la mémoire."""
        return self.memory.load_memory_variables({})
    
    def remove_index(self):
        shutil.rmtree(self.index_file)
        os.remove(self.memory_file)
    

def  main_function():
    load_dotenv()
    api_key = os.getenv('MISTRAL_AI_KEY')
    
    parser = argparse.ArgumentParser(
                    prog='Langchain RAG',
                    description='Langchain RAG with Faiss and Mistral')
    
    # Créer des sous-parseurs pour différentes commandes
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')

    # Sous-parseur pour l'ajout de documents
    subparsers.add_parser('add', help='Ajouter des documents depuis Opendatasoft')
    
    # Sous-parseur pour la recherche
    search_parser = subparsers.add_parser('search', help='Rechercher dans les documents')
    search_parser.add_argument('query', type=str, help='Requête de recherche')
    
    subparsers.add_parser('test', help='Tester si on répond des requêtes correctement')
    
    args = parser.parse_args()
    
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    index_file = "culture_event_index"

    llm = ChatMistralAI(
        model="mistral-medium",
        api_key=api_key,
        temperature=0.7,
        language="fr"
    )

    # Entraîner l'index pour Faiss
    df = get_example_for_index()
    documents = get_documents(df)
    index, _ = create_optimized_index(EMBEDDING_DIM, embeddings, documents, 50)

    # Initialisation de Vector store avec de l'index entraîné
    faiss_vector_store = FaissVectorStore(embeddings, index_file, index, llm)

    if os.path.isfile(index_file):
        faiss_vector_store.load_vectorstore(index_file)

    if args.command == 'add':
        # Ajout de documents dans la base vectorielle
        df = get_culture_event_agenda(location="Île-de-France")
        documents = get_documents(df)
        faiss_vector_store.add_documents(documents)
        
        # Vérification d'indexation
        if not faiss_vector_store.verify_indexing(df):
            print("Vérification de l'indexation a échoué")
        else:
            print("Vérification de l'indexation réussit")

    elif  args.command == 'search':

        query = args.query

        if os.path.exists(faiss_vector_store.index_file):

            faiss_vector_store.load_vectorstore(index_file)
            # Exécution de la recherche
            result = faiss_vector_store.process_query(query=query)
            print("\nRéponse:", result["response"]["current_answer"])
            print("\nNombre d'interactions:", result["response"]["context"]["total_interactions"])
        else:
            print("Fichier de base vectorielle n'existe pas, merci de faire d'indexation d'abord! ")

    elif  args.command == 'clean':

        # Suppression de fichier d'indexation et le historique de conversation 
        faiss_vector_store.remove_index()

    else:
        parser.print_help()

