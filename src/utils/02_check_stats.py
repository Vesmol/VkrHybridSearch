import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def analyze_extraction_quality(base_path: str = "E:\\VKR") -> None:
    """Анализ полноты извлечения метаданных (УДК, аннотации, ключевые слова)."""
    json_path = os.path.join(base_path, "data", "processed", "cleaned_articles.json")
    
    if not os.path.exists(json_path):
        logging.error(f"Файл не найден: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total = len(data)
    if total == 0:
        logging.warning("Датасет пуст.")
        return

    stats = {'udk': 0, 'abstract': 0, 'keywords': 0}

    for article in data.values():
        if article.get("udk") != "Не найден": stats['udk'] += 1
        if article.get("abstract") != "Не найдена": stats['abstract'] += 1
        if len(article.get("keywords", [])) > 0: stats['keywords'] += 1

    logging.info("Отчет о качестве парсинга датасета:")
    logging.info("-" * 40)
    logging.info(f"Всего документов: {total}")
    logging.info(f"Извлечено УДК: {stats['udk']} ({stats['udk']/total*100:.1f}%)")
    logging.info(f"Извлечено аннотаций: {stats['abstract']} ({stats['abstract']/total*100:.1f}%)")
    logging.info(f"Извлечено ключевых слов: {stats['keywords']} ({stats['keywords']/total*100:.1f}%)")

if __name__ == "__main__":
    analyze_extraction_quality()