# Используем базовый образ с Python
FROM python:3.11-slim
# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app
# Копируем файл с зависимостями в контейнер
COPY requirements.txt /app/
# Устанавливаем зависимости
RUN pip install -r requirements.txt
# Копируем весь проект в контейнер
COPY .. /app/
# Открываем порт для FastAPI
EXPOSE 8000
# Запускаем приложение через Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

