import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_json_results(base_path: str = "E:\\VKR") -> None:
    """Проверка структуры и целостности сгенерированного JSON-файла парсера."""
    json_path = os.path.join(base_path, "data", "processed", "parsed_articles.json")
    
    if not os.path.exists(json_path):
        logging.error(f"Файл не найден: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logging.info(f"Всего документов в базе: {len(data)}")
    
    if not data:
        return

    first_key = list(data.keys())[0]
    sample_article = data[first_key]
    
    logging.info(f"Тестовая проверка файла: {sample_article.get('file_name', 'Unknown')}")
    logging.info(f"Длина извлеченного текста: {len(sample_article.get('raw_text', ''))} символов")
    
    metadata = sample_article.get('metadata', {})
    logging.info(f"Метаданные | Автор: {metadata.get('Author', 'N/A')}")
    logging.info(f"Метаданные | ПО: {metadata.get('xmp:CreatorTool', 'N/A')}")

if __name__ == "__main__":
    check_json_results()