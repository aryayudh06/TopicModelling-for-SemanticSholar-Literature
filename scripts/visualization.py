import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import torch
from transformers import BertTokenizer, BertModel

class Visualization:
    def __init__(self, data_path="./data/processed_semantic_data.csv", output_dir="./charts"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Membuat folder charts jika belum ada

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File {self.data_path} tidak ditemukan!")
        self.df = pd.read_csv(self.data_path)

    def plot_topic_distribution(self):
        topic_counts = self.df['Topic'].value_counts().sort_index()
        plt.figure(figsize=(8, 6))
        plt.bar(topic_counts.index, topic_counts.values, color='skyblue')
        plt.xlabel("Topic Number")
        plt.ylabel("Document Count")
        plt.title("Topic Distribution in Documents")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        save_path = os.path.join(self.output_dir, "topic_distribution.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Gambar distribusi topik disimpan di: {save_path}")

    def generate_wordcloud(self):
        text = ' '.join(self.df['Processed_Title'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud of Processed Titles")
        save_path = os.path.join(self.output_dir, "wordcloud.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Gambar word cloud disimpan di: {save_path}")

    def generate_tfidf_analysis(self):
        documents = self.df['Processed_Title'].dropna().tolist()
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        save_path = os.path.join(self.output_dir, "tfidf_table.csv")
        tfidf_df.to_csv(save_path, index=False)
        print(f"Representasi TF-IDF disimpan di: {save_path}")

    def generate_bert_pca_visualization(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        def get_bert_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            mean_embeddings = torch.mean(embeddings, dim=1)
            return mean_embeddings

        texts = self.df['Processed_Title'].dropna()[:10]
        embeddings = torch.cat([get_bert_embeddings(text) for text in texts], dim=0)

        pca = PCA(n_components=2)
        bert_2d = pca.fit_transform(embeddings.numpy())

        plt.figure(figsize=(8, 6))
        plt.scatter(bert_2d[:, 0], bert_2d[:, 1], color='blue', alpha=0.7)
        plt.title('PCA of BERT Embeddings')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        save_path = os.path.join(self.output_dir, "bert_pca.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Gambar PCA BERT embeddings disimpan di: {save_path}")

    def generate_charts(self):
        """Menjalankan semua proses visualisasi."""
        self.plot_topic_distribution()
        self.generate_wordcloud()
        self.generate_tfidf_analysis()
        self.generate_bert_pca_visualization()

if __name__ == "__main__":
    vis = Visualization()
    vis.generate_charts()