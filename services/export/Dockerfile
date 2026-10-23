FROM python:3.12-slim
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx pyyaml
COPY img_export.py /app/img_export.py
WORKDIR /app
CMD ["uvicorn", "img_export:app", "--host", "0.0.0.0", "--port", "8054"]
