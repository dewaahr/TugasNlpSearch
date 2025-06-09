from flask import Flask, render_template, request, jsonify
import os
import re
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')

app = Flask(__name__)

# Folder PDF
PDF_FOLDER = './document'

# Fungsi dari script Anda
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
        return ["Enter 1 or 2 words only."]

    if not candidates:
        return ["No predictions found."]

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_candidates[:5]]  # Top 5 predictions

# Baca PDF dan bangun indeks
documents = []
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith('.pdf'):
        filepath = os.path.join(PDF_FOLDER, filename)
        try:
            text = read_pdf_text_fitz(filepath)
            documents.append(text)
        except Exception as e:
            print(f"[!] Failed to read {filename}: {e}")

unigrams, bigrams, trigrams = build_ngram_index(documents)

# Routes
@app.route('/')
def home():
    return render_template('search.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "Query is empty"}), 400

    # Prediksi kata berikutnya
    predictions = predict_next_word(query, unigrams, bigrams, trigrams)

    # Cari dokumen relevan berdasarkan query
    relevant_documents = []
    for filename, doc in zip(os.listdir(PDF_FOLDER), documents):
        if query.lower() in doc.lower():
            # Cari cuplikan kalimat yang mengandung kata pencarian
            sentences = doc.split('.')
            snippet = next((sentence for sentence in sentences if query.lower() in sentence.lower()), '')
            snippet = snippet.strip()

            # Potong cuplikan menjadi -3 + kata pencarian +3
            words = snippet.split()
            query_lower = query.lower()
            for i, word in enumerate(words):
                if query_lower in word.lower():
                    start = max(0, i - 3)
                    end = min(len(words), i + 4)
                    snippet = ' '.join(words[start:end])
                    break

            # Tambahkan hasil dengan judul dokumen dan cuplikan
            relevant_documents.append({
                "title": filename.replace('.pdf', ''),  # Judul dokumen
                "snippet": snippet
            })

    return jsonify({
        "predictions": predictions,
        "relevant_documents": relevant_documents
    })

if __name__ == '__main__':
    app.run(debug=True)