web: python download_model.py && gunicorn app:app --preload --workers 1 --threads 1 --timeout 300 --graceful-timeout 300 --bind 0.0.0.0:$PORT
