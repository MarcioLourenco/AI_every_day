FROM python:3.10.11-slim

# Definir diretório de trabalho dentro do container
WORKDIR /app

# Copiar todos os arquivos do diretório atual para o container
COPY . /app

# Instalar dependências necessárias
RUN pip install --no-cache-dir -r requirements.txt

# Comando para iniciar a aplicação Flask
CMD ["python", "app.py"]