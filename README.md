# Segmentation of cargo x-ray images

## Структура проекта

```bash
.
├── configs                <--  Конфигурационные файлы
│   └── config.json
├── Dockerfile             <-- Докер-образ с установленными зависимостями и проектом внутри
├── missed_planes          <-- Основной каталог
│   ├── dataset.py             <-- Класс для описания набора данных
│   ├── __init__.py            <-- Инициализация, определение версии данного пакета
│   ├── predict.py             <-- Процедура получения результатов и оценки модели
│   ├── security.py            <-- Функция для расшифровки данных
│   └── train.py               <-- Процедура обучения модели
├── README.md              <-- Этот файл
├── requirements.txt       <-- Описание зависимостей
└── setup.py               <-- Файл для сборки python-пакета
```
