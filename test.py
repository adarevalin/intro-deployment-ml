from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_random_prediction():
    response = client.post('v1/prediction', json = {
                    'opening_gross': 8330681,
                    'screens': 2271,
                    'production_budget': 1300000,
                    'title_year': 1999,
                    'aspect_ratio': 1.86,
                    'duration': 97,
                    'budget': 16000000,
                    'imdb_score': 7.2
    })


    assert response.status_code == 200
    assert response.json()['worldwide_gross'] != 0