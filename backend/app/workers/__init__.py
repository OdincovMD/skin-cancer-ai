"""Celery: приложение и зарегистрированные задачи."""

from workers.app import celery_app

# Имя `app` — ожидание CLI: `celery -A workers worker`
app = celery_app

from workers import tasks  # noqa: E402, F401 — регистрация задач при импорте пакета

__all__ = ["app", "celery_app"]
