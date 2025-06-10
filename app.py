from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Konfigurasi URL untuk API eksternal
LANGUAGE_MODEL_API_URL = "http://localhost:8000/predict"
SEARCH_DOCUMENT_API_URL = "http://localhost:7000/search"

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "Query is empty"}), 400

    # Panggil API dari languageModelApi untuk prediksi kata
    try:
        lm_response = requests.post(LANGUAGE_MODEL_API_URL, json={"query": query})
        lm_response.raise_for_status()
        predictions = lm_response.json().get('prediksi', [])
        print("Predictions:", predictions)
    except requests.exceptions.RequestException as e:
        print(f"Language Model API error: {str(e)}")
        return jsonify({"error": f"Language Model API error: {str(e)}"}), 500

    return jsonify({
        "query": query,
        "predictions": predictions
    })

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data['query']

    if not query:
        return jsonify({'error': 'Query tidak boleh kosong'}), 400

    # Panggil API dari searchDocumentApi untuk pencarian dokumen
    try:
        sd_response = requests.post(SEARCH_DOCUMENT_API_URL, json={"query": query})
        sd_response.raise_for_status()
        hasil = sd_response.json().get('hasil', [])
        print("Search Results:", hasil)
    except requests.exceptions.RequestException as e:
        print(f"Search Document API error: {str(e)}")
        return jsonify({"error": f"Search Document API error: {str(e)}"}), 500

    return jsonify({
        'query': query,
        'hasil': hasil
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)