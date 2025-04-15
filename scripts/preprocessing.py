import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import os
from deep_translator import GoogleTranslator

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Preprocessor:
    def __init__(self, data_path="./data/semantic_scholar_results.csv", save_path="./data/processed_semantic_data.csv", callback=None):
        # Inisialisasi Preprocessor dengan path dataset dan hasil penyimpanan.
        # Parameter:
        # - data_path (str): Lokasi file dataset
        # - save_path (str): Lokasi penyimpanan hasil preprocessing
        # - callback (function): Fungsi callback yang dipanggil setelah tiap tahap selesai
        self.data_path = data_path
        self.save_path = save_path
        self.callback = callback
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=5, random_state=42)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)  # Pastikan folder penyimpanan ada

    def _invoke_callback(self, step, message):
        # Memanggil fungsi callback jika disediakan.
        if self.callback:
            self.callback(step, message)

    def load_data(self):
        # Memuat dataset dari CSV.
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File {self.data_path} tidak ditemukan!")
        
        self.df = pd.read_csv(self.data_path)
        if 'title' not in self.df.columns:
            raise KeyError("Kolom 'title' tidak ditemukan dalam dataset")

        self._invoke_callback("load_data", "Dataset berhasil dimuat!")

    def translate_text(self, text):
        # Mendeteksi dan menerjemahkan teks ke bahasa Inggris jika diperlukan.
        try:
            lang = GoogleTranslator().detect(text)
            if lang != 'en':
                return GoogleTranslator(source=lang, target='en').translate(text)
            return text
        except Exception:
            return text  # Kembalikan teks asli jika gagal diterjemahkan

    def preprocess_text(self, text):
        # Membersihkan, menerjemahkan, dan melakukan tokenisasi pada teks.
        text = str(text).lower()
        text = self.translate_text(text)  # Translate before cleaning
        text = re.sub(r'\[pdf\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def apply_preprocessing(self):
        # Menerapkan preprocessing ke kolom 'title'.

        self.df['Processed_Title'] = self.df['title'].astype(str).apply(self.preprocess_text)
        self._invoke_callback("apply_preprocessing", "Preprocessing teks selesai!")

    # def topic_modeling(self):
    #     # Menerapkan LDA Topic Modeling untuk menentukan topik dalam teks.
    #     dtm = self.vectorizer.fit_transform(self.df['Processed_Title'])
    #     self.lda.fit(dtm)

    #     # Menampilkan top 10 kata dalam setiap topik
    #     topics_summary = []
    #     for index, topic in enumerate(self.lda.components_):
    #         top_words = [self.vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
    #         topics_summary.append({"Topic": index, "Top Words": top_words})

    #     # Menentukan topik dominan untuk setiap dokumen
    #     topic_results = self.lda.transform(dtm)
    #     self.df['Topic'] = topic_results.argmax(axis=1)

    #     self._invoke_callback("topic_modeling", {"message": "Topic modeling selesai!", "topics": topics_summary})

    def save_data(self):
        # Menyimpan hasil preprocessing ke dalam file CSV.
        self.df.to_csv(self.save_path, index=False)
        self._invoke_callback("save_data", f"Preprocessed data saved to '{self.save_path}'")

    def run(self):
        # Menjalankan seluruh proses preprocessing.
        self._invoke_callback("start", "Preprocessing dimulai...")
        try:
            self.load_data()
            self.apply_preprocessing()
            # self.topic_modeling()
            self.save_data()
            self._invoke_callback("complete", "Preprocessing selesai!")
            return self.save_path
        except Exception as e:
            self._invoke_callback("error", f"Terjadi kesalahan: {str(e)}")


# Contoh fungsi callback
def process_callback(step, message):
    if isinstance(message, dict):  # Jika message berupa dict (seperti topic_modeling)
        print(f"[{step.upper()}] {message['message']}")
        for topic in message["topics"]:
            print(f"  - Topic {topic['Topic']}: {', '.join(topic['Top Words'])}")
    else:
        print(f"[{step.upper()}] {message}")

# Menjalankan preprocessing dengan callback
if __name__ == "__main__":
    preprocessor = Preprocessor(callback=process_callback)
    preprocessor.run()
