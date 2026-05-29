import os
import json
import re
import logging
import pymorphy3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LexicalSearchEngine:
    """Модуль поиска на основе векторной модели TF-IDF."""
    
    STOP_WORDS = {
        "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
        "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
        "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
        "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
        "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
        "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
        "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
        "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
        "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
        "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
        "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда",
        "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
        "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
        "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
        "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более",
        "всегда", "конечно", "всю", "между", "причем", "это"
    }

    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.morph = pymorphy3.MorphAnalyzer()
        self.filenames = []
        self.documents = []
        self.metadata = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        
    def _preprocess_query(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^а-яёa-z]', ' ', text)
        lemmatized_words = [
            self.morph.parse(word)[0].normal_form 
            for word in text.split() 
            if word not in self.STOP_WORDS and len(word) > 2
        ]
        return " ".join(lemmatized_words)

    def load_and_index(self) -> bool:
        if not os.path.exists(self.corpus_path):
            logging.error(f"Индекс не найден: {self.corpus_path}")
            return False

        logging.info("Инициализация загрузки лексического корпуса...")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for filename, info in data.items():
            self.filenames.append(filename)
            self.documents.append(info["lemmatized_text"])
            self.metadata.append({
                "udk": info["udk"],
                "abstract": info["original_abstract"]
            })
            
        logging.info("Построение пространства признаков TF-IDF...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        logging.info(f"Индексация завершена. Векторизовано документов: {len(self.documents)}")
        return True
        
    def search(self, query: str, top_n: int = 5) -> None:
        clean_query = self._preprocess_query(query)
        if not clean_query:
            print("\nОшибка: Запрос пуст или содержит только стоп-слова.")
            return

        query_vector = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]
        
        print(f"\n--- Результаты лексического поиска: '{query}' ---")
        
        found = False
        for i, idx in enumerate(top_indices, 1):
            score = similarities[idx]
            if score == 0:
                continue
                
            found = True
            meta = self.metadata[idx]
            short_abstract = meta['abstract'][:300] + "..." if len(meta['abstract']) > 300 else meta['abstract']
            
            print(f"[{i}] Документ: {self.filenames[idx]} | Косинусное сходство: {score:.4f}")
            print(f"    УДК: {meta['udk']}")
            print(f"    Аннотация: {short_abstract}")
            print("-" * 70)
            
        if not found:
            print("Релевантные документы не обнаружены.")

def run_app():
    base_path = "E:\\VKR"
    corpus_path = os.path.join(base_path, "data", "processed", "search_corpus.json")
    
    engine = LexicalSearchEngine(corpus_path)
    if not engine.load_and_index():
        return
    
    print("\nМодуль лексического поиска запущен. Для завершения введите 'exit'.")
    while True:
        try:
            user_query = input("\nПоисковый запрос > ").strip()
            if user_query.lower() in ['exit', 'quit', 'выход']:
                print("Завершение сеанса.")
                break
            if user_query:
                engine.search(user_query, top_n=5)
        except KeyboardInterrupt:
            print("\nПринудительное завершение.")
            break

if __name__ == "__main__":
    run_app()