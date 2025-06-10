from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from searchDocument import normalisasi, cari_dokumen
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')
nltk.download('stopwords')

# Inisialisasi Flask
app = Flask(__name__)

# Load TF-IDF dari file CSV
def load_tfidf_from_csv(nama_file='index_tfidf.csv'):
    df = pd.read_csv(nama_file)
    nama_dokumen = df['Dokumen'].tolist()
    fitur = [f.lower() for f in df.columns[1:]]  # Ubah semua fitur menjadi lowercase
    tfidf_matrix = df.iloc[:, 1:].values

    # Fit vectorizer dengan fitur yang ada
    vectorizer = TfidfVectorizer(vocabulary=fitur, lowercase=False)
    dummy_corpus = [' '.join(fitur)]  # Supaya vectorizer dianggap "fitted"
    vectorizer.fit(dummy_corpus)

    return tfidf_matrix, vectorizer, nama_dokumen

# Load data TF-IDF
tfidf_matrix, vectorizer, nama_dokumen = load_tfidf_from_csv()

# Inisialisasi stop words dan stemmer
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preproses(text):
    norma = normalisasi(text).lower()  # Pastikan teks menjadi lowercase
    tokenize = nltk.word_tokenize(norma)
    filtered = []
    for word in tokenize:
        if word not in stop_words and word.isalpha():
            filtered.append(stemmer.stem(word))  # Gunakan Sastrawi untuk stemming
    return filtered

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data['query']

    if not query:
        return jsonify({'error': 'Query tidak boleh kosong'}), 400

    hasil = cari_dokumen(query, vectorizer, tfidf_matrix, nama_dokumen)
    print("Hasil pencarian:", hasil)
    return jsonify({
        'query': query,
        'hasil': [{'nama_dokumen': nama, 'skor': float(f"{skor:.4f}")} for nama, skor in hasil]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)
