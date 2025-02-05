import os
import pickle
import json
import numpy as np
import tarfile

# Caminho onde o SageMaker vai extrair os arquivos do .tar.gz
MODEL_PATH = "/opt/ml/model"

# Função para extrair e carregar o modelo e o vectorizer
def model_fn(model_dir):
    model_file = os.path.join(model_dir, "modelo.pkl")
    vectorizer_file = os.path.join(model_dir, "tfidf_vectorizer.pkl")

    # Descompactar o modelo se ele ainda não estiver descompactado
    if not os.path.exists(model_file):
        with tarfile.open(os.path.join(model_dir, "model.tar.gz"), "r:gz") as tar:
            tar.extractall(path=model_dir)
    
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    
    with open(vectorizer_file, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

# Função para processar a entrada da requisição
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        request = json.loads(request_body)
        return request["inputs"]
    else:
        raise ValueError("Formato de entrada não suportado!")

# Função para realizar a predição
def predict_fn(input_data, model):
    model, vectorizer = model
    transformed_input = vectorizer.transform(input_data)
    predictions = model.predict(transformed_input)
    return predictions.tolist()

# Função para formatar a resposta
def output_fn(prediction, response_content_type):
    return json.dumps({"predictions": prediction})
