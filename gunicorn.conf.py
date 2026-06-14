# gunicorn.conf.py
timeout = 300
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
