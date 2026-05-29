import os
import json
import time
import logging
from tika import parser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_pdf_folder(base_path: str = "E:\\VKR") -> None:
    """Извлечение сырого текста и метаданных из PDF-файлов с использованием Apache Tika."""
    pdf_folder = os.path.join(base_path, "data", "raw_pdf")
    output_folder = os.path.join(base_path, "data", "processed")
    output_json_path = os.path.join(output_folder, "parsed_articles.json")

    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    existing_data = {}
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            logging.info(f"Загружена существующая база. Статей: {len(existing_data)}")
        except Exception as e:
            logging.error(f"Ошибка чтения JSON: {e}. Индексация начнется с нуля.")

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    logging.info(f"Найдено PDF-файлов для обработки: {len(pdf_files)}")

    if not pdf_files:
        logging.warning("Директория raw_pdf пуста.")
        return

    logging.info("Инициализация сервера Apache Tika...")
    new_count = 0
    start_time = time.time()

    for file_name in pdf_files:
        if file_name in existing_data:
            continue

        file_path = os.path.join(pdf_folder, file_name)
        logging.info(f"Обработка файла: {file_name}")

        try:
            parsed = parser.from_file(file_path)
            raw_text = parsed.get("content", "")
            metadata = parsed.get("metadata", {})

            if raw_text:
                existing_data[file_name] = {
                    "file_name": file_name,
                    "raw_text": raw_text.strip(),
                    "metadata": metadata,
                    "title": "",
                    "authors": [],
                    "abstract": "",
                    "keywords": []
                }
                new_count += 1
            else:
                logging.warning(f"Текст не извлечен (вероятно, графический скан): {file_name}")

        except Exception as e:
            logging.error(f"Ошибка при обработке {file_name}: {e}")

    if new_count > 0:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        
        logging.info(f"Парсинг завершен за {time.time() - start_time:.2f} сек.")
        logging.info(f"Добавлено новых документов: {new_count}. Всего в базе: {len(existing_data)}")
    else:
        logging.info("Новых файлов для обработки не обнаружено.")

if __name__ == "__main__":
    parse_pdf_folder()