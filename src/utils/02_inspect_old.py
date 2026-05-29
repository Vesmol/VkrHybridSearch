import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def inspect_problematic_files(base_path: str = "E:\\VKR") -> None:
    """Вывод фрагментов сырого текста для отладки регулярных выражений."""
    json_path = os.path.join(base_path, "data", "processed", "parsed_articles.json")

    if not os.path.exists(json_path):
        logging.error(f"Файл не найден: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    problem_files = [
        '2010_2010-02-24.pdf',
        '2013_2013-02-01.pdf',
        '2010_2010-01-06.pdf'
    ]

    for file_name in problem_files:
        if file_name in data:
            logging.info(f"\n--- Инспекция файла: {file_name} ---")
            logging.info(data[file_name]["raw_text"][:1200])
            logging.info("-" * 40)
        else:
            logging.warning(f"Файл {file_name} отсутствует в датасете.")

if __name__ == "__main__":
    inspect_problematic_files()