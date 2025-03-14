# preprocessing.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file
df = pd.read_csv('../../data/universitas_brawijaya_scholar_results2.csv')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove "[PDF]" or "[pdf]" using regular expression (case-insensitive)
    text = re.sub(r'\[pdf\]', '', text, flags=re.IGNORECASE)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply the preprocessing function to the 'Title' column
df['Processed_Title'] = df['Title'].apply(preprocess_text)

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

# Display the first few rows of the dataframe with assigned topics
print(df[['Title', 'Topic']].head())

# Save the preprocessed data
path = '../../data/processed_data.csv'
df.to_csv(path, index=False)
print(f"Preprocessed data saved to '{path}'")