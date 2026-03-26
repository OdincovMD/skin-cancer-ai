from workers.app import celery_app

app = celery_app

from workers import tasks  # noqa: E402, F401

__all__ = ["app", "celery_app"]
