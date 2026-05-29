# VkrHybridSearch
Программный комплекс гибридного поиска по архивам научных публикаций (ВКР)
Стек технологий: Python 3.12, Streamlit, PyTorch, Sentence-Transformers, Tika.
Структура проекта: 
   1. Исходный код модуля автоматизированной загрузки веб-архива 00_download.py
   2. Исходный код модуля синтаксического анализа 01_parser_tika.py
   3. Исходный код модуля сегментации 02_regex_processor.py
   4. Исходный код модуля отчистки датасета 03_dataset_cleaner.py
   5. Исходный код модуля морфологической нормализации 04_search_pipeline.py
   6. Исходный код модуля лексического поиска tfidf_engine.py
   7. Исходный код модуля генерации эмбеддингов 05_generate_embeddings.py
   8. Исходный код гибридного ранжирования hybrid_engine.py
   9. Исходный код веб-приложения main.py
Как запустить: pip install -r requirements.txt и streamlit run main.py.
