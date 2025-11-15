# Базовый образ с Python
FROM python:3.10.11

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями
COPY requierements.txt .

# Устанавливаем Python библиотеки
RUN pip install --no-cache-dir -r requierements.txt

# Копируем весь исходный код
COPY src/ ./src/
COPY data/ ./data/

# Команда для запуска бота
CMD ["python", "src/bot_max.py"]