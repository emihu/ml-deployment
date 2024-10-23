from flask import Flask, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

@application.route("/")
def index():
    return "Hello, World!"

@application.route("/load")
def load_model():
    global loaded_model
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)
    
    global vectorizer
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)
    
    return "Model loaded"

@application.route("/predict/<string:input>", methods=["GET"])
def predict(input):
    prediction = loaded_model.predict(vectorizer.transform([input]))[0]
    return prediction

if __name__ == "__main__":
    application.run(port=5000, debug=True)