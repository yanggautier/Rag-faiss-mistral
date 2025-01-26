import re
import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_mistralai import MistralAIEmbeddings


load_dotenv()
api_key = os.getenv('MISTRAL_AI_KEY')

embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

def clean_text(text):
    """
    Fonction permet de nettoyer un texte qui sert à alimenter la base de données vectorielle

    Parameters:
    text (str): le texte à nettoyer
    """
    # Supprime HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Convertit en minuscules
    text = text.lower()
    # Supprime caractères spéciaux et emoji tout en gardant accents
    text = re.sub(r'[^\w\s.,!?;:\'\"À-ÿ]', ' ', text)
    # Normalise espaces
    text = ' '.join(text.split())
    return text


def get_agenda(event_type, start_year, location):
    """
    Récupérer les données sur le site d'Opendatsoft

    Parameters:
    event_type (str): type d'évenèment, exemple:  Tout public, musique, concert, théâtre, Recrutement, gratuit
    start_year (int): l'année de début des évenèments, exemple:  2023, 2024
    location (str): la région, exemple: Île-de-France, Hauts-de-France

    Returns:
    df (Dataframe): retourne un Pandas DataFrame
    """
    # Requête pour obtenir le nombre total d'objet
    url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?limit=1&refine=keywords_fr%3A%22{event_type}%22&refine=firstdate_begin%3A%22{str(start_year)}%22&refine=location_region%3A%22{location}%22"
    response = requests.get(url)
    total_count = response.json()["total_count"]

    # Requête des données d'agenda par 100
    results = []
    for offset_index in range(int(total_count/100)+1):
        offset_url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?limit=100&offset={str(offset_index * 100)}&refine=keywords_fr%3A%22{event_type}%22&refine=firstdate_begin%3A%22{str(start_year)}%22&refine=location_region%3A%22{location}%22" 
        offset_response = requests.get(offset_url)
        offset_results = offset_response.json()["results"]
        results = results + offset_results
        time.sleep(1)

    # Transformer les données obtenues et transformations de d données pour 
    df = pd.DataFrame.from_dict(results)
    df["description_fr"] = df['description_fr'].apply(clean_text)
    df["content"] = df["description_fr"] + " lieu: " + df["location_name"] + " adresse: " + df["location_address"] + " " +df["location_city"] + " " + df["location_postalcode"] + " dates: " +df["daterange_fr"] + " date de début: " + df["firstdate_begin"] + " date de fin:" + df["lastdate_end"] + " mots clés:" + df["keywords_fr"].astype(str)

    return df


def get_example_for_index():
    """
    Récupérer les données sur le site d'Opendatsoft

    Returns:
    df (Dataframe): retourne un Pandas DataFrame
    """
    # Requête pour obtenir le nombre total d'objet
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?limit=100&refine=keywords_fr%3A%22Recrutement%22&refine=firstdate_begin%3A%222024%22&refine=location_region%3A%22%C3%8Ele-de-France%22"
    response = requests.get(url)
    results = response.json()["results"]

    # Transformer les données obtenues et transformations de d données pour 
    df = pd.DataFrame.from_dict(results)
    #df.drop_duplicates(inplace=True)
    df["description_fr"] = df['description_fr'].apply(clean_text)
    df["content"] = df["description_fr"] + " \n lieu: " + df["location_name"] + " \n adresse: " + df["location_address"] + " " +df["location_city"] + " " + df["location_postalcode"] + " \ndates: " +df["daterange_fr"]

    return df
def get_documents(df):
    """Transform DataFrame to Langchain Documents list"""
    documents = []
    for _, row in df.iterrows():
        # Vérifier si content n'est pas NaN
        content = row["content"]
        if pd.isna(content):
            content = f"description: {row['description_fr']} \nlieu: {row['location_name']} \nadresse: {row['location_address']} {row['location_city']} {row['location_postalcode']} \ndates: {row['daterange_fr']}"
            
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "opendatasoft", 
                    "id": row["uid"], 
                    "title": clean_text(row["title_fr"]), 
                    "description": row["description_fr"],
                    "firstdate_begin": row["firstdate_begin"],
                    "firstdate_end": row["firstdate_end"],
                    "lastdate_begin": row["lastdate_begin"],
                    "lastdate_end": row["lastdate_end"],
                    "location_coordinates": row["location_coordinates"],
                    "location_name": row["location_name"], 
                    "location_address": row["location_address"], 
                    "location_district": row["location_district"], 
                    "location_postalcode": row["location_postalcode"], 
                    "location_city": row["location_city"],
                    "location_description": row["location_description_fr"]
                }
            )
        )
    return documents


def splitter_documents(docs):
    """Fonction permets de découper des documents ou textes en plus petits documents
    

    Args:
        docs (List): Liste de Documents Langchain

    Returns:
        splitted_docs: Liste de documents splittés
    """

    text_splitter = SemanticChunker(MistralAIEmbeddings())
    splitted_docs = text_splitter.create_documents(docs)

    return splitted_docs