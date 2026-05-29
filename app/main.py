import os
import warnings
import logging
import re
import json
import numpy as np
import torch
import pymorphy3
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Отключение лишних системных сообщений
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Пути к данным
BASE_PATH = "E:\\VKR"
CORPUS_PATH = os.path.join(BASE_PATH, "data", "processed", "search_corpus.json")
EMBEDDINGS_PATH = os.path.join(BASE_PATH, "data", "processed", "articles_with_embeddings.json")
PDF_DIR = os.path.join(BASE_PATH, "data", "raw_pdf")

st.set_page_config(page_title="Поиск статей", page_icon="🔍", layout="wide")

@st.cache_resource
def init_nlp():
    return pymorphy3.MorphAnalyzer(), SentenceTransformer("intfloat/multilingual-e5-small")

@st.cache_data
def load_data():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        lexical_db = json.load(f)
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        semantic_db = json.load(f)

    filenames = list(lexical_db.keys())
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform([lexical_db[f]["lemmatized_text"] for f in filenames])
    db_embeddings = torch.tensor([semantic_db[f]["embedding"] for f in filenames])
    
    return filenames, semantic_db, vectorizer, tfidf_matrix, db_embeddings

morph, model = init_nlp()
filenames, semantic_db, vectorizer, tfidf_matrix, db_embeddings = load_data()

def preprocess(text):
    tokens = re.findall(r'\b[а-яa-z0-9\-]+\b', text.lower())
    return " ".join([morph.parse(t)[0].normal_form for t in tokens])

def fix_text(text):
    return re.sub(r'([а-яА-Яa-zA-Z])-\s+([а-яА-Яa-zA-Z])', r'\1\2', text)

# Интерфейс
st.sidebar.title("Настройки")
st.sidebar.metric("Статей в базе", len(filenames))
top_n = st.sidebar.slider("Количество результатов", 1, 20, 5)

st.title("Поисковая система «Вестник ВГУ»")
st.caption("Системный анализ и информационные технологии")

query = st.text_input("", placeholder="Введите поисковый запрос")

if query:
    clean_query = preprocess(query)
    
   # Лексический и семантический поиск
    tfidf_sims = cosine_similarity(vectorizer.transform([clean_query]), tfidf_matrix).flatten()
    q_emb = model.encode(f"query: {query}", convert_to_tensor=True)
    sem_sims = util.cos_sim(q_emb, db_embeddings).cpu().numpy().flatten()
    tfidf_sorted_indices = np.argsort(-tfidf_sims)
    sem_sorted_indices = np.argsort(-sem_sims)
    tfidf_ranks = {filenames[idx]: rank for rank, idx in enumerate(tfidf_sorted_indices)}
    sem_ranks = {filenames[idx]: rank for rank, idx in enumerate(sem_sorted_indices)}

    #RRF
    rrf = {}
    for f in filenames:
        rrf[f] = (1 / (60 + tfidf_ranks[f] + 1)) + (1 / (60 + sem_ranks[f] + 1))
    
    # Сортировка по убыванию гибридного балла
    results = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_n]

    st.markdown("### Результаты:")
    for i, (f_name, _) in enumerate(results, 1):
        data = semantic_db[f_name]
        pdf_path = os.path.join(PDF_DIR, f_name)
        
        with st.container(border=True):
            # Заголовок и кнопка в одной строке
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"#### {i}. {f_name}")
            with col2:
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button("⬇ PDF", f, f_name, "application/pdf", key=f"dl_{i}")
            
            st.markdown(f"**УДК:** `{data.get('udk', 'Не указан')}`")
            
            with st.expander("Посмотреть аннотацию"):
                st.write(fix_text(data.get("abstract", "Нет аннотации")))