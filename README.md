# Système RAG avec FAISS et Mistral AI

Système de génération augmentée par recherche (RAG) utilisant la base de données vectorielle FAISS et Mistral AI pour répondre aux questions en français sur les événements publics.

## Fonctionnalités

- Recherche vectorielle avec FAISS
- Embeddings et LLM Mistral AI
- Gestion de l'historique des conversations
- Intégration OpenDataSoft pour les données d'événements
- Indexation IVFFlat optimisée
- Suite de tests unitaires

## Prérequis

```txt
langchain==0.3.14
langgraph==0.2.64
faiss-cpu==1.9.0.post1
pandas==2.2.3
beautifulsoup4==4.12.3
```

## Installation

```bash
git clone https://github.com/yanggautier/Rag-faiss-mistral.git
cd Rag-faiss-mistral
pip install -r requirements.txt
```

Configurez votre clé API Mistral AI dans `.env`:
```
MISTRAL_API_KEY=votre_clé_api
```

## Utilisation

Ajouter des documents à la base vectorielle:
```bash
python main.py add
```

Rechercher des événements:
```bash
python main.py search "votre requête ici"
```

Lancer les tests:
```bash
python -m unittest tests/test_chat.py
```

## Architecture

Le système est composé de trois modules principaux:

- `documents.py`: Traitement des documents et intégration OpenDataSoft
- `chat.py`: Base vectorielle FAISS et fonctionnalités de chat
- `test_chat.py`: Tests unitaires

## Configuration

Paramètres clés dans `chat.py`:
```python
EMBEDDING_DIM = 1024  # Dimension des embeddings Mistral
N_CELLS = 10         # Nombre de cellules Voronoi
N_PROBE = 5          # Nombre de cellules à explorer
```

