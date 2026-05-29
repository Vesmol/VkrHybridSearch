import os
import json
import logging
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_vkr_embeddings(base_path: str = "E:\\VKR") -> None:
    """Генерация семантических векторных представлений (эмбеддингов) для корпуса документов."""
    input_json_path = os.path.join(base_path, "data", "processed", "cleaned_articles.json")
    output_json_path = os.path.join(base_path, "data", "processed", "articles_with_embeddings.json")
    
    if not os.path.exists(input_json_path):
        logging.error(f"Датасет не найден: {input_json_path}")
        return

    logging.info("Загрузка очищенного датасета...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    model_name = "intfloat/multilingual-e5-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Инициализация модели {model_name} на вычислительном узле: {device.upper()}")
    model = SentenceTransformer(model_name, device=device)

    logging.info("Формирование текстовых чанков с префиксами...")
    file_names = []
    passages = []
    
    for file_name, article in articles.items():
        abstract = article.get("abstract", "")
        keywords_list = article.get("keywords", [])
        keywords = ", ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)
        
        if not abstract or abstract == "Не найдена":
            text_to_embed = f"Ключевые слова: {keywords}. Текст: {article.get('raw_text', '')[:500]}"
        else:
            text_to_embed = f"Аннотация: {abstract}. Ключевые слова: {keywords}"
            
        full_passage = f"passage: {text_to_embed}"
        
        file_names.append(file_name)
        passages.append(full_passage)

    logging.info(f"Запуск тензорных вычислений для {len(passages)} документов...")
    embeddings = model.encode(
        passages, 
        batch_size=16, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )

    logging.info("Сборка и сериализация векторной базы данных...")
    for idx, file_name in enumerate(file_names):
        vector_list = embeddings[idx].tolist()
        articles[file_name]["embedding"] = vector_list

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
        
    logging.info("Векторизация корпуса успешно завершена.")

if __name__ == "__main__":
    generate_vkr_embeddings()