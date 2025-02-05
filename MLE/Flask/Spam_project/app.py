from flask import Flask, request, jsonify
from   spam_classifier import spam_detector
import pickle
import pandas as pd

try:
    with open(".\\data\\model_1.pkl", "rb") as file:
        model = pickle.load(file)

    with open(".\\data\\tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
except:
    print("Model not found.")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400  
    text_vector = vectorizer.transform([text])  
    prediction = model.predict(text_vector)
    return jsonify({"prediction": prediction.tolist()})

@app.route("/train", methods=["POST"])
def train():
    train_df = pd.read_csv(".\\data\\data_1\\train.csv")
    valid_df = pd.read_csv(".\\data\\data_1\\valid.csv")
    test_df = pd.read_csv(".\\data\\data_1\\test.csv")
    results = spam_detector(train_df, valid_df, test_df)
    with open(".\\data\\model_1.pkl", "wb") as file:
        pickle.dump(results["Model"], file)

    return jsonify({"Model": results["BestClassifier"], 
                    "Metrics": results[results["BestClassifier"]].tolist(),
                    "Train": "Done!"})

if __name__ == "__main__":
    app.run(debug=True)