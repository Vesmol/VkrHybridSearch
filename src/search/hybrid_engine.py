import os
import json
import re
import logging
import torch
import numpy as np
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridSearchEngine:
    """Гибридная поисковая система, объединяющая TF-IDF и Dense Retrieval через алгоритм RRF."""
    
    STOP_WORDS = {
        "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
        "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
        "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
        "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
        "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
        "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
        "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
        "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже"
    }

    def __init__(self, corpus_path: str, embeddings_path: str):
        self.corpus_path = corpus_path
        self.embeddings_path = embeddings_path
        self.morph = pymorphy3.MorphAnalyzer()
        self.vectorizer = TfidfVectorizer()
        
        self.filenames = []
        self.semantic_db = {}
        self.tfidf_matrix = None
        self.db_embeddings = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        words = re.findall(r'\b[а-яa-z0-9\-]+\b', text)
        lemmas = [self.morph.parse(w)[0].normal_form for w in words if w not in self.STOP_WORDS]
        return " ".join(lemmas)

    def load_and_index(self) -> bool:
        if not os.path.exists(self.corpus_path) or not os.path.exists(self.embeddings_path):
            logging.error("Отсутствуют необходимые файлы индексов.")
            return False

        logging.info("Инициализация лексического модуля...")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            lexical_db = json.load(f)
            
        logging.info("Инициализация семантического модуля...")
        with open(self.embeddings_path, "r", encoding="utf-8") as f:
            self.semantic_db = json.load(f)

        self.filenames = list(self.semantic_db.keys())
        
        corpus_texts = [lexical_db[f]["lemmatized_text"] for f in self.filenames]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

        self.db_embeddings = torch.tensor([self.semantic_db[f]["embedding"] for f in self.filenames])
        self.model = SentenceTransformer("intfloat/multilingual-e5-small", device=self.device)
        
        logging.info("Гибридный индекс успешно сформирован.")
        return True

    def search(self, query: str, top_n: int = 5, k_param: int = 60) -> None:
        if not query:
            return

        # 1. Лексический поток (TF-IDF)
        clean_query = self._preprocess_text(query)
        query_tfidf = self.vectorizer.transform([clean_query])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        lexical_rank_indices = np.argsort(-tfidf_scores)
        lexical_ranks = {self.filenames[idx]: rank for rank, idx in enumerate(lexical_rank_indices)}

        # 2. Семантический поток (E5)
        prepared_query = f"query: {query}"
        query_embedding = self.model.encode(prepared_query, convert_to_tensor=True, show_progress_bar=False)
        cos_scores = util.cos_sim(query_embedding, self.db_embeddings)[0]
        semantic_rank_indices = torch.argsort(cos_scores, descending=True)
        semantic_ranks = {self.filenames[idx.item()]: rank for rank, idx in enumerate(semantic_rank_indices)}

        # 3. Слияние (Reciprocal Rank Fusion)
        rrf_scores = {}
        for f_name in self.filenames:
            r_tfidf = lexical_ranks[f_name]
            r_e5 = semantic_ranks[f_name]
            rrf_scores[f_name] = (1.0 / (k_param + r_tfidf)) + (1.0 / (k_param + r_e5))

        sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]

        print(f"\n--- Результаты гибридного поиска: '{query}' ---")
        for i, (f_name, rrf_score) in enumerate(sorted_results, 1):
            article_data = self.semantic_db[f_name]
            orig_tfidf_place = lexical_ranks[f_name] + 1
            orig_e5_place = semantic_ranks[f_name] + 1
            
            abstract = article_data.get("abstract", "Нет аннотации")
            snippet = abstract[:250] + "..." if len(abstract) > 250 else abstract
            
            print(f"[{i}] Документ: {f_name} | Итоговый RRF-балл: {rrf_score:.5f}")
            print(f"    Место в TF-IDF: {orig_tfidf_place} | Место в E5: {orig_e5_place}")
            print(f"    УДК: {article_data.get('udk', 'Не указан')}")
            print(f"    Аннотация: {snippet}")
            print("-" * 70)

def run_app():
    base_path = "E:\\VKR"
    corpus_path = os.path.join(base_path, "data", "processed", "search_corpus.json")
    embeddings_path = os.path.join(base_path, "data", "processed", "articles_with_embeddings.json")
    
    engine = HybridSearchEngine(corpus_path, embeddings_path)
    if not engine.load_and_index():
        return
        
    print("\nМодуль гибридного поиска запущен. Для выхода введите 'exit'.")
    while True:
        try:
            query = input("\nГибридный запрос > ").strip()
            if query.lower() in ["exit", "выход", "quit"]:
                print("Завершение сеанса.")
                break
            engine.search(query)
        except KeyboardInterrupt:
            print("\nПринудительное завершение.")
            break

if __name__ == "__main__":
    run_app()