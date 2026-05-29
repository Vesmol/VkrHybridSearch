import os
import json
import re
import time
import logging
import pymorphy3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MORPH = pymorphy3.MorphAnalyzer()
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

def preprocess_text(text: str) -> str:
    """Токенизация, удаление стоп-слов и лемматизация текста для TF-IDF индекса."""
    if not text or text == "Не найдена":
        return ""

    text = text.lower()
    text = re.sub(r'[^а-яёa-z]', ' ', text)
    
    lemmatized_words = [
        MORPH.parse(word)[0].normal_form 
        for word in text.split() 
        if word not in STOP_WORDS and len(word) > 2
    ]
    
    return " ".join(lemmatized_words)

def build_search_corpus(base_path: str = "E:\\VKR") -> None:
    """Подготовка лексического корпуса для точного текстового поиска."""
    input_path = os.path.join(base_path, "data", "processed", "cleaned_articles.json")
    output_path = os.path.join(base_path, "data", "processed", "search_corpus.json")

    logging.info("Инициализация сборки лексического корпуса...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus_data = {}
    start_time = time.time()

    for idx, (filename, article) in enumerate(data.items(), 1):
        if idx % 100 == 0:
            logging.info(f"Лемматизация: обработано {idx}/{len(data)} документов.")

        abstract = article.get("abstract", "")
        keywords_list = article.get("keywords", [])
        keywords = " ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)
        
        raw_search_text = f"{abstract} {keywords}"
        clean_lemmas = preprocess_text(raw_search_text)

        corpus_data[filename] = {
            "udk": article.get("udk", "Не найден"),
            "lemmatized_text": clean_lemmas,
            "original_abstract": abstract,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=4)

    logging.info(f"Поисковый индекс сформирован за {time.time() - start_time:.2f} сек.")

if __name__ == "__main__":
    build_search_corpus()