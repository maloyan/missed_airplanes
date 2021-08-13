# Missed airplanes classification (https://cups.mail.ru/)

## Структура проекта

```bash
.
├── configs                <--  Конфигурационные файлы
│   └── config.json
├── Dockerfile
├── missed_planes          <-- Основной каталог
│   ├── __init__.py            <-- Инициализация, определение версии данного пакета
│   ├── dataset.py             <-- Класс для описания набора данных
│   ├── engine.py              <-- Train loop
│   ├── meter.py               <-- Класс для метрик
│   ├── metrics.py             <-- Метрики
│   ├── predict.py             <-- Генерация сабмита
│   └── train.py               <-- Обучение
├── README.md
├── requirements.txt       <-- Описание зависимостей
└── setup.py               <-- Файл для сборки python-пакета
```
