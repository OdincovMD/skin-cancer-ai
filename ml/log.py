import logging
from functools import wraps
import time


class Logger():
    def __init__(self, name):
        """
        Инициализация логгирования.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(name)s][%(levelname)s]: [%(message)s]",
            handlers=[
                logging.FileHandler("log.txt"),  
                logging.StreamHandler(), 
            ],
        )
        self.logger = logging.getLogger(name=name)
    
    def log_function_entry_exit(self, func):
        """
        Декоратор для логирования входа и выхода из функции.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.logger.info(f"Начало выполнения блока {func.__module__}")
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                execution_time = end_time - start_time 

                self.logger.info(
                    f"Завершение выполнения блока {func.__module__}. "
                    f"Время выполнения: {execution_time:.4f} секунд"
                )
                return result

            except Exception as e:
                self.logger.error(f"Ошибка в блоке {func.__module__}: {e}")
                raise 

        return wrapper