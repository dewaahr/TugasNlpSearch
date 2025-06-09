import os
import re
import csv
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')

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

def save_ngrams_to_csv(counter, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ngram', 'frequency'])
        for ngram, freq in counter.items():
            writer.writerow([' '.join(ngram) if isinstance(ngram, tuple) else ngram, freq])

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
        return ["Masukkan 1 atau 2 kata saja."]

    if not candidates:
        return ["Tidak ada prediksi ditemukan."]

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_candidates[:5]]  # Top 5 prediksi

def main():
    print("Membaca PDF dan membangun indeks...")

    documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith('.pdf'):
            filepath = os.path.join(PDF_FOLDER, filename)
            try:
                text = read_pdf_text_fitz(filepath)
                documents.append(text)
            except Exception as e:
                print(f"[!] Gagal membaca {filename}: {e}")


    unigrams, bigrams, trigrams = build_ngram_index(documents)

    # Simpan index ke file
    save_ngrams_to_csv(unigrams, 'unigram_index.csv')
    save_ngrams_to_csv(bigrams, 'bigram_index.csv')
    save_ngrams_to_csv(trigrams, 'trigram_index.csv')

    # Prediksi interaktif
    while True:
        query = input("Ketik kueri (1–2 kata, atau ketik 'exit' untuk keluar): ")
        if query.lower() == 'exit':
            break
        predictions = predict_next_word(query, unigrams, bigrams, trigrams)
        print("Prediksi kata berikutnya:", predictions)

if __name__ == "__main__":
    main()
