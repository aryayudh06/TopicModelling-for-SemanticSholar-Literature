import pandas as pd

# Load the processed data
df = pd.read_csv('../../data/processed_data.csv')

# Basic statistics
print(df.describe())

# Distribution of topics
print(df['Topic'].value_counts())

# Frequency of words in Processed_Title
from collections import Counter

all_words = ' '.join(df['Processed_Title']).split()
word_freq = Counter(all_words)
print(word_freq.most_common(20))

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Visualize topic distribution
df['Topic'].value_counts().plot(kind='bar')
plt.title('Distribution of Topics')
plt.xlabel('Topic')
plt.ylabel('Number of Documents')
plt.show()

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Processed_Title']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Processed Titles')
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Contoh data teks
documents = df['Processed_Title'].tolist()  # Ambil kolom teks yang sudah diproses

# Inisialisasi TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Hitung TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Konversi ke DataFrame untuk visualisasi yang lebih baik
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Tampilkan representasi TF-IDF untuk 5 dokumen pertama
print("Representasi TF-IDF untuk 5 dokumen pertama:")
print(tfidf_df.head())

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model dan tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Fungsi untuk mendapatkan BERT embeddings dari teks
def get_bert_embeddings(text):
    # Tokenisasi teks
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Dapatkan embeddings dari model BERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Ambil embeddings dari lapisan terakhir (last hidden state)
    embeddings = outputs.last_hidden_state
    
    # Rata-rata embeddings untuk seluruh token (bisa juga menggunakan [CLS] token)
    mean_embeddings = torch.mean(embeddings, dim=1)
    
    return mean_embeddings

# Contoh: Ambil teks dari dokumen pertama
text = df['Processed_Title'][0]

# Dapatkan BERT embeddings
bert_embeddings = get_bert_embeddings(text)

# Tampilkan representasi BERT embeddings
print("Representasi BERT embeddings untuk dokumen pertama:")
print(bert_embeddings)


# Ambil beberapa teks dari DataFrame
texts = df['Processed_Title'][:10]  # Ambil 10 teks pertama

# Dapatkan embeddings untuk semua teks
embeddings = torch.cat([get_bert_embeddings(text) for text in texts], dim=0)

# Reduksi dimensi dengan PCA
pca = PCA(n_components=2)
bert_2d = pca.fit_transform(embeddings.numpy())

# Plot hasil PCA
plt.scatter(bert_2d[:, 0], bert_2d[:, 1])
plt.title('PCA of BERT Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
