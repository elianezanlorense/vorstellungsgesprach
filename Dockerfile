FROM python:3.12-slim

# Instala o uv (gerenciador de dependências usado no projeto)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copia apenas os arquivos de dependências primeiro (melhor cache de build)
COPY pyproject.toml uv.lock ./

# Instala as dependências (sem instalar o próprio projeto como pacote)
RUN uv sync --frozen --no-install-project

# Copia o restante do código
COPY . .

# Instala o projeto em si
RUN uv sync --frozen

# Constrói a base de conhecimento (ChromaDB) durante o build
RUN uv run python main.py pipeline

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]