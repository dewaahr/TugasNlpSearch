import os
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import csv

# nltk.download('stopwords')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


from nltk.corpus import stopwords  
stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory() 
stemmer = factory.create_stemmer() 

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

def proses_dokumen(direktori):
    dokumens = []
    nama_dokumen = []
    for filename in sorted(os.listdir(direktori)):
        if filename.endswith('.pdf'):  
            file_path = os.path.join(direktori, filename)
            dokumen = ambil_text_pdf(file_path)  
            processed = preproses(dokumen)       
            dokumens.append(processed)          
            nama_dokumen.append(filename) 
    return dokumens, nama_dokumen

def gabungkan_dokumen(dokumen):
    dokumen_gabung = []
    for doc in dokumen:
        gabung = ' '.join(doc)
        dokumen_gabung.append(gabung)
    return dokumen_gabung

def hitung_tfidf(dokumen_str):
    vectorizer = TfidfVectorizer()
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
    query_tokens = preproses(query)
    query_str = ' '.join(query_tokens)
    query_vec = vectorizer.transform([query_str])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    hasil = sorted(zip(nama_dokumen, sim_scores), key=lambda x: x[1], reverse=True)
    return hasil

def main():
    direktori = "./dokumen"
    
    dokumens, nama_dokumen = proses_dokumen(direktori)
    dokumen_str = gabungkan_dokumen(dokumens)

    tfidf_matrix, vectorizer, fitur = hitung_tfidf(dokumen_str)
    simpan_index_csv(tfidf_matrix, nama_dokumen, fitur)

    query = input("\nMasukkan kata pencarian: ")
    hasil = cari_dokumen(query, vectorizer, tfidf_matrix, nama_dokumen)

    print("\nHasil Pencarian:")
    print(f"\n{'Nama Dokumen'.ljust(120)} | Skor Kemiripan")
    for nama, skor in hasil:
        print(f"{nama.ljust(120)} : {skor:.4f}")

if __name__ == "__main__":
    main()






