from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import csv
from multiprocessing import Pool
import pickle

app = Flask(__name__)
CORS(app)  # Tambahkan ini untuk mengizinkan CORS

# Initialize NLTK and Sastrawi
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords  
stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory() 
stemmer = factory.create_stemmer() 

# Global variables untuk menyimpan model yang sudah dilatih
tfidf_matrix = None
vectorizer = None
nama_dokumen = []
is_initialized = False

def normalisasi(text):
    text_lower = text.lower()
    text_hapus_tanda_baca = text_lower.translate(str.maketrans('', '', string.punctuation))
    return text_hapus_tanda_baca

def preproses(text):
    norma = normalisasi(text)
    tokenize = nltk.word_tokenize(norma)
    filtered = []
    for word in tokenize:
        if word not in stop_words and word.isalpha():
            filtered.append(stemmer.stem(word))
    return filtered

def ambil_text_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def proses_dokumen_parallel(file_paths):
    with Pool() as pool:
        results = pool.map(ambil_text_pdf, file_paths)
    return results

def proses_dokumen(direktori):
    file_paths = [os.path.join(direktori, filename) for filename in sorted(os.listdir(direktori)) if filename.endswith('.pdf')]
    dokumens = proses_dokumen_parallel(file_paths)
    nama_dokumen = [os.path.basename(path) for path in file_paths]
    return dokumens, nama_dokumen

def gabungkan_dokumen(dokumen):
    dokumen_gabung = []
    for doc in dokumen:
        gabung = ' '.join(doc)
        dokumen_gabung.append(gabung)
    return dokumen_gabung

def hitung_tfidf(dokumen_str):
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit vocabulary size
    tfidf_matrix = vectorizer.fit_transform(dokumen_str)
    fitur = vectorizer.get_feature_names_out()
    return tfidf_matrix, vectorizer, fitur

def simpan_index_csv(tfidf_matrix, nama_dokumen, fitur, nama_file='index_tfidf.csv'):
    with open(nama_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Dokumen'] + list(fitur))
        for i, dok in enumerate(nama_dokumen):
            row = [dok] + list(tfidf_matrix[i].toarray()[0])
            writer.writerow(row)

def cari_dokumen(query, vectorizer, tfidf_matrix, nama_dokumen):
    try:
        print("Query:", query)  # Debug log
        query_tokens = preproses(query)
        print("Query tokens:", query_tokens)  # Debug log
        query_str = ' '.join(query_tokens)
        query_vec = vectorizer.transform([query_str])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        hasil = sorted(zip(nama_dokumen, sim_scores), key=lambda x: x[1], reverse=True)
        print("Results:", hasil)  # Debug log
        return hasil
    except Exception as e:
        print("Error in cari_dokumen:", e)  # Debug log
        raise

def simpan_cache(tfidf_matrix, vectorizer, nama_dokumen, cache_file='cache.pkl'):
    with open(cache_file, 'wb') as f:
        pickle.dump((tfidf_matrix, vectorizer, nama_dokumen), f)

def muat_cache(cache_file='cache.pkl'):
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def initialize_system():
    global tfidf_matrix, vectorizer, nama_dokumen, is_initialized
    try:
        direktori = "./document"
        dokumens, nama_dokumen = proses_dokumen(direktori)
        dokumen_str = gabungkan_dokumen(dokumens)
        tfidf_matrix, vectorizer, fitur = hitung_tfidf(dokumen_str)
        is_initialized = True
        print("System initialized successfully")  # Debug log
    except Exception as e:
        print("Initialization error:", e)  # Debug log

@app.route('/search', methods=['POST', 'OPTIONS'])
def search_documents():
    try:
        data = request.get_json()
        print("Query received:", data)  # Debug log
        query = data.get('query', '')
        if not query:
            return jsonify({"error": "Query is empty"}), 400
        
        results = cari_dokumen(query, vectorizer, tfidf_matrix, nama_dokumen)
        print("Search results:", results)  # Debug log
        return jsonify({"results": results, "total_documents": len(nama_dokumen)})
    except Exception as e:
        print("Error:", e)  # Debug log
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# initialize_system()

