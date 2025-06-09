from flask import Flask, request, jsonify
import os
import re
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

# Download NLTK data (hanya sekali)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# Global variables untuk menyimpan model
unigrams = None
bigrams = None
trigrams = None
model_loaded = False

PDF_FOLDER = './document'

def read_pdf_text_fitz(file_path):
    doc = fitz.open(file_path)
    text = ''
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

def build_ngram_index(documents):
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    for doc in documents:
        tokens = preprocess(doc)
        unigrams.update(tokens)
        bigrams.update(ngrams(tokens, 2))
        trigrams.update(ngrams(tokens, 3))
    return unigrams, bigrams, trigrams

def predict_next_word(query, unigrams, bigrams, trigrams):
    tokens = preprocess(query)
    if len(tokens) == 2:
        prefix = tuple(tokens)
        candidates = {k[2]: v for k, v in trigrams.items() if k[:2] == prefix}
    elif len(tokens) == 1:
        prefix = tokens[0]
        candidates = {k[1]: v for k, v in bigrams.items() if k[0] == prefix}
    else:
        return []
    
    if not candidates:
        return []
    
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_candidates[:5]]  # Top 5 prediksi

def load_model():
    global unigrams, bigrams, trigrams, model_loaded
    
    if model_loaded:
        return True
    
    try:
        
        documents = []
        if not os.path.exists(PDF_FOLDER):
            print(f"Folder {PDF_FOLDER} tidak ditemukan")
            return False
        
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"Tidak ada file PDF di {PDF_FOLDER}")
            return False
        
        for filename in pdf_files:
            filepath = os.path.join(PDF_FOLDER, filename)
            try:
                text = read_pdf_text_fitz(filepath)
                documents.append(text)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        if not documents:
            return False
        
        unigrams, bigrams, trigrams = build_ngram_index(documents)
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    global unigrams, bigrams, trigrams, model_loaded
    
    if not model_loaded:
        if not load_model():
            return jsonify({'error': 'Model tidak dapat dimuat'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query tidak boleh kosong'}), 400
        
        predictions = predict_next_word(query, unigrams, bigrams, trigrams)
        
        return jsonify({
            'query': query,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Language Model API...")
    load_model()
    app.run(debug=True, port=5000)