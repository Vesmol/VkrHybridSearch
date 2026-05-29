import os
import json
import logging
import torch
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SemanticSearchEngine:
    """Модуль поиска на основе нейросетевых векторных представлений (Dense Retrieval)."""
    
    def __init__(self, embeddings_path: str):
        self.embeddings_path = embeddings_path
        self.filenames = []
        self.articles_db = {}
        self.db_embeddings = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_and_index(self) -> bool:
        if not os.path.exists(self.embeddings_path):
            logging.error(f"Файл векторного индекса не найден: {self.embeddings_path}")
            return False

        logging.info("Загрузка семантической базы данных...")
        with open(self.embeddings_path, "r", encoding="utf-8") as f:
            self.articles_db = json.load(f)

        self.filenames = list(self.articles_db.keys())
        self.db_embeddings = torch.tensor([self.articles_db[f]["embedding"] for f in self.filenames])

        model_name = "intfloat/multilingual-e5-small"
        logging.info(f"Инициализация ИИ-модели {model_name} на узле: {self.device.upper()}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        logging.info(f"Семантический индекс загружен. Векторов: {len(self.filenames)}")
        return True

    def search(self, query: str, top_n: int = 5) -> None:
        if not query:
            return

        prepared_query = f"query: {query}"
        query_embedding = self.model.encode(prepared_query, convert_to_tensor=True, show_progress_bar=False)
        
        cos_scores = util.cos_sim(query_embedding, self.db_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_n, len(self.filenames)))
        
        print(f"\n--- Результаты семантического поиска: '{query}' ---")
        for score, idx in zip(top_results.values, top_results.indices):
            article_idx = idx.item()
            file_name = self.filenames[article_idx]
            article_data = self.articles_db[file_name]
            
            score_percent = score.item() * 100
            abstract = article_data.get("abstract", "Нет аннотации")
            snippet = abstract[:300] + "..." if len(abstract) > 300 else abstract
            
            print(f"Документ: {file_name} | Релевантность: {score_percent:.2f}%")
            print(f"УДК: {article_data.get('udk', 'Не указан')}")
            print(f"Аннотация: {snippet}")
            print("-" * 70)

def run_app():
    base_path = "E:\\VKR"
    embeddings_path = os.path.join(base_path, "data", "processed", "articles_with_embeddings.json")
    
    engine = SemanticSearchEngine(embeddings_path)
    if not engine.load_and_index():
        return
        
    print("\nМодуль семантического поиска запущен. Для выхода введите 'exit'.")
    while True:
        try:
            query = input("\nПоисковый запрос > ").strip()
            if query.lower() in ["exit", "выход", "quit"]:
                print("Завершение сеанса.")
                break
            engine.search(query)
        except KeyboardInterrupt:
            print("\nПринудительное завершение.")
            break

if __name__ == "__main__":
    run_app()