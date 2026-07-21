# 1. Imagem base oficial do Python (slim para ser leve e eficiente)
FROM python:3.14

# 2. Impede o Python de gravar arquivos .pyc no disco e habilita logs sem buffer
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Define o diretório de trabalho dentro do container
WORKDIR /app

# 4. Instala dependências de sistema necessárias para compilação (FAISS, PyTorch e libs C++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copia e instala as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copia todo o código-fonte da aplicação para o container
COPY . .

# 7. Expõe a porta em que o Uvicorn vai rodar
EXPOSE 8000

# 8. Comando padrão para iniciar a API FastAPI em produção
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

