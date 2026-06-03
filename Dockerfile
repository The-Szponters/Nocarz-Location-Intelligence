FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn
COPY . .
RUN chmod +x docker/entrypoint.sh
ENV PYTHONPATH="/app/src"
EXPOSE 8080
# One image, several modes (serve | pipeline | ab | test); see docker/entrypoint.sh.
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["serve"]
