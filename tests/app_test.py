import requests
import pytest
from pathlib import Path

from application import application

# URL of the running Flask app
url = "http://127.0.0.1:5000/predict/"

# List of inputs: two fake news, two real news
test_inputs = [
    ["This is fake news about an event that never happened.", "FAKE"],
    ["The aliens are coming to kill us.", "FAKE"],
    ["This is real news about an event that happened.", "REAL"],
    ["Red, blue and yellow are primary colours.", "REAL"]
]

iterations = 100

def test_index():
    tester = application.test_client()
    response = tester.get("/", content_type="html/text")
    assert response.status_code == 200
    assert response.data == b"Hello, World!"

@pytest.fixture
def client():
    BASE_DIR = Path(__file__).resolve().parent.parent
    application.config["TESTING"] = True
    yield application.test_client() # tests run here

def test_load(client):
    """Logout helper function"""
    response = client.get("/load")
    assert response.status_code == 200

def test_predict(client):
    """Ensure the prediction is correct"""
    for input in test_inputs:
        encoded_input = requests.utils.quote(input[0])  # Encode the input for the URL
        print(encoded_input)
        response = client.get("/predict/" + encoded_input, content_type="html/text")
        print (response)
        print(response.get_data(as_text=True))
        assert response.status_code == 200
        assert response.get_data(as_text=True) == input[1]
 