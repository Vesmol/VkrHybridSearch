import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def compare_datasets(base_path: str = "E:\\VKR") -> None:
    """Сравнение сырого и очищенного датасетов для выявления отфильтрованных документов."""
    dirty_path = os.path.join(base_path, "data", "processed", "parsed_articles.json")
    clean_path = os.path.join(base_path, "data", "processed", "cleaned_articles.json")

    try:
        with open(dirty_path, "r", encoding="utf-8") as f:
            dirty_data = json.load(f)
        with open(clean_path, "r", encoding="utf-8") as f:
            clean_data = json.load(f)
    except FileNotFoundError as e:
        logging.error(f"Ошибка доступа к файлам данных: {e}")
        return

    trash_files = [fname for fname in dirty_data if fname not in clean_data]

    logging.info(f"Отфильтровано документов (информационный шум): {len(trash_files)}")
    for idx, file_name in enumerate(sorted(trash_files), 1):
        logging.info(f"{idx}. {file_name}")

if __name__ == "__main__":
    compare_datasets()