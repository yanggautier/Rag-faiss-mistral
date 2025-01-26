import os
import unittest
import pandas as pd
from dotenv import load_dotenv
from app.chat import *
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_mistralai import ChatMistralAI
from app.documents import get_documents, get_example_for_index
from langchain_mistralai.embeddings import MistralAIEmbeddings
import tempfile
import shutil
import logging

# Configuration du logging pour déboguer
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()
api_key = os.getenv('MISTRAL_AI_KEY')

class TestFaissVectorStore(unittest.TestCase):
    """
    Test unitaire pour le chatbot
    """

    def setUp(self):
        """Préparer les ressources pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()  # Répertoire temporaire pour les fichiers FAISS
        self.index_file = os.path.join(self.temp_dir, "test_index")

        # Charger ou générer des données d'exemple
        if os.path.exists("test.csv"):
            self.sample_data = pd.read_csv("test.csv")
        else:
            self.sample_data = get_example_for_index()
            self.sample_data.to_csv("test.csv", index=False)

        # Générer les documents et les embeddings
        self.documents = get_documents(self.sample_data)
        self.embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
        self.llm = ChatMistralAI(model="mistral-medium", api_key=api_key, temperature=0.7)

    def tearDown(self):
        """Nettoyer les fichiers temporaires après chaque test."""
        shutil.rmtree(self.temp_dir)  # Supprimer le répertoire temporaire

    def test_index_creation(self):
        """Test de création et d'entraînement de l'index FAISS."""
        index, vectors = create_optimized_index(EMBEDDING_DIM, self.embeddings, self.documents, 50)
        self.assertIsNotNone(index, "Indexation a échoué")
        self.assertEqual(len(vectors), len(self.documents), "Nombre de vecteurs incorrect")

    def test_documents(self):
        """Test de l'ajout de documents à la base vectorielle."""
        index, vectors = create_optimized_index(EMBEDDING_DIM, self.embeddings, self.documents, 50)
        faiss_vector_store = FaissVectorStore(self.embeddings, self.index_file, index, self.llm)

        # Ajout des documents
        faiss_vector_store.add_documents(self.documents)

        # Vérifications
        self.assertIsInstance(self.sample_data, pd.DataFrame, "Le type de sample_data doit être une instance de Pandas DataFrame")
        self.assertGreater(len(self.sample_data), 0, "Le nombre de lignes doit être supérieur à 0")
        self.assertTrue(os.path.exists(self.index_file), "Le fichier d'index n'existe pas")
        self.assertEqual(len(self.documents), len(faiss_vector_store.vector_store.docstore._dict), "Le nombre de documents ajoutés ne correspond pas")

        # Logs pour vérifier le contenu du docstore
        logger.debug(f"Documents dans le docstore : {faiss_vector_store.vector_store.docstore._dict.keys()}")

    def test_query_processing(self):
        """Test de traitement d'une requête utilisateur."""
        index, vectors = create_optimized_index(EMBEDDING_DIM, self.embeddings, self.documents, 50)
        faiss_vector_store = FaissVectorStore(self.embeddings, self.index_file, index, self.llm)

        # Ajout des documents
        faiss_vector_store.add_documents(self.documents)

        # Chargement de l'index
        faiss_vector_store.load_vectorstore(self.index_file)

        # Exécution d'une requête
        query = "Quels événements sont disponibles en janvier 2024 ?"
        response = faiss_vector_store.process_query(query=query)

        # Vérifications
        self.assertIn("current_answer", response["response"], "La réponse n'est pas incluse dans la réponse retournée")
        self.assertGreaterEqual(len(response["chat_history"]), 0, "L'historique de conversation est vide")


if __name__ == "__main__":
    unittest.main()
