import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os
import sys
import json
from pathlib import Path

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class Preprocessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Pastikan direktori output ada
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def preprocess_text(self, text):
        """Basic text cleaning"""
        text = str(text).lower()
        text = re.sub(r'\[pdf\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def process_data(self):
        """Main processing function"""
        try:
            # Load data
            if self.input_path.endswith('.json'):
                df = pd.read_json(self.input_path)
            elif self.input_path.endswith('.csv'):
                df = pd.read_csv(self.input_path)
            else:
                raise ValueError("Unsupported file format")

            # Validasi kolom
            if 'title' not in df.columns:
                raise KeyError("Dataframe must contain 'title' column")
            
            # Preprocess titles
            df['processed_title'] = df['title'].astype(str).apply(self.preprocess_text)
            
            # Save processed data
            if self.output_path.endswith('.json'):
                df.to_json(self.output_path, orient='records')
            else:
                df.to_csv(self.output_path, index=False)
            
            # Generate metrics
            metrics = {
                'original_count': len(df),
                'processed_count': len(df),
                'columns_processed': ['title'],
                'processing_date': pd.Timestamp.now().isoformat()
            }
            
            metrics_path = Path(self.output_path).parent / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            print("Preprocessing completed successfully!")
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python preprocessing.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    preprocessor = Preprocessor(input_path, output_path)
    preprocessor.process_data()

if __name__ == "__main__":
    main()