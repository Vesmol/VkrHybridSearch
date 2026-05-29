import os
import json
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text: str) -> str:
    """Удаление избыточных пробельных символов."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def process_articles(base_path: str = "E:\\VKR") -> None:
    """Структурирование сырого текста путем выделения УДК, аннотаций и ключевых слов."""
    input_json_path = os.path.join(base_path, "data", "processed", "parsed_articles.json")
    
    if not os.path.exists(input_json_path):
        logging.error(f"Исходный файл не найден: {input_json_path}")
        return

    logging.info("Загрузка неструктурированного корпуса...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    processed_count = 0

    for file_name, article in data.items():
        raw_text = article.get("raw_text", "")
        
        udk_match = re.search(r"\b(?:УДК|UDC)\b\s*([\d\.\+:\s/]+)", raw_text, re.IGNORECASE)
        article["udk"] = clean_text(udk_match.group(1)) if udk_match else "Не найден"

        annotation_pattern = r"(?:Аннотация|Abstract)[\.\:]?\s*(.*?)(?=Ключевые\s+слова|Key\s*words|Введение|Introduction|©|\n[А-ЯA-Z]{2,}\s+\n)"
        annotation_match = re.search(annotation_pattern, raw_text, re.DOTALL | re.IGNORECASE)
        article["abstract"] = clean_text(annotation_match.group(1)) if annotation_match else "Не найдена"

        keywords_pattern = r"(?:Ключевые\s+слова|Key\s*words)[\.\:]?\s*(.*?)(?=\n\n|\n[А-ЯA-Z\s]{5,}|\n\s*Введение|\n\s*Introduction|©)"
        keywords_match = re.search(keywords_pattern, raw_text, re.IGNORECASE)
        if keywords_match:
            raw_keywords = keywords_match.group(1)
            article["keywords"] = [clean_text(k) for k in re.split(r'[,;.]', raw_keywords) if clean_text(k)]
        else:
            article["keywords"] = []

        processed_count += 1

    logging.info(f"Обработано документов: {processed_count}. Сохранение данных...")
    with open(input_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logging.info("Структурирование (Regex Processor) успешно завершено.")

if __name__ == "__main__":
    process_articles()