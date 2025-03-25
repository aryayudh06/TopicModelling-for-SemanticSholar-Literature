import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from deep_translator import GoogleTranslator

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data_name = "semantic_scholar_results.csv"
data_path = f"./data/{data_name}"

# Load the CSV file
df = pd.read_csv(data_path)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to detect and translate text
def translate_text(text):
    try:
        lang = GoogleTranslator().detect(text)
        if lang != 'en':
            return GoogleTranslator(source=lang, target='en').translate(text)
        return text
    except Exception as e:
        return text  # Return original text if translation fails

# Define a function to preprocess the text
def preprocess_text(text):
    text = str(text).lower()
    text = translate_text(text)  # Translate before cleaning
    text = re.sub(r'\[pdf\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Terapkan preprocessing pada kolom judul
if 'title' in df.columns:
    df['Processed_Title'] = df['title'].astype(str).apply(preprocess_text)
else:
    raise KeyError("Kolom 'title' tidak ditemukan dalam dataset")

# Apply the preprocessing function to the 'Title' column
df['Processed_Title'] = df['title'].apply(preprocess_text)


# Initialize the CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')

# Fit and transform the processed text data
dtm = vectorizer.fit_transform(df['Processed_Title'])

# Initialize the LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)

# Fit the LDA model on the document-term matrix
lda.fit(dtm)

# Display the top words for each topic
for index, topic in enumerate(lda.components_):
    print(f'Top 10 words for Topic #{index}:')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print('\n')

# Assign the most relevant topic to each document
topic_results = lda.transform(dtm)
df['Topic'] = topic_results.argmax(axis=1)


# Save the preprocessed data
path = './data/processed_semantic_data.csv'
df.to_csv(path, index=False)
print(f"Preprocessed data saved to '{path}'")
