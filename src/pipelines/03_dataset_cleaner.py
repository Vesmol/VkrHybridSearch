import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_and_finalize_dataset(base_path: str = "E:\\VKR") -> None:
    """Фильтрация корпуса: удаление документов без научных признаков (информационный шум)."""
    input_json_path = os.path.join(base_path, "data", "processed", "parsed_articles.json")
    output_json_path = os.path.join(base_path, "data", "processed", "cleaned_articles.json")
    
    if not os.path.exists(input_json_path):
        logging.error(f"Исходный файл не найден: {input_json_path}")
        return

    logging.info("Анализ структурированной базы документов...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total_input = len(data)
    cleaned_data = {}
    removed_count = 0
    
    for file_name, article in data.items():
        if article.get("udk") == "Не найден" and article.get("abstract") == "Не найдена":
            removed_count += 1
            continue
            
        cleaned_data[file_name] = article

    logging.info("Сохранение очищенного датасета...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    logging.info(f"Исходный объем: {total_input} документов.")
    logging.info(f"Отфильтровано (шум): {removed_count} документов.")
    logging.info(f"Эффективный объем базы: {len(cleaned_data)} документов ({(len(cleaned_data)/total_input*100):.1f}%).")

if __name__ == "__main__":
    clean_and_finalize_dataset()