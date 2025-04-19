# services/training_service.py
from typing import List, Dict, Union
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from bertopic import BERTopic
from pathlib import Path
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self, model_type='lda'):
        """
        Initialize training service with specified model type.
        
        Args:
            model_type (str): Type of model to use ('lda' or 'bertopic')
        """
        self.model_type = model_type.lower()
        self.vectorizer = None
        self.model = None
        self.coherence_score = None
        self.topic_keywords = {}  # To store topic keywords
        self.topic_names = {}     # To store topic names (BERTopic only)
        
    def train_model(self, data_path='data/processed_semantic_data.csv', 
                   text_column='Processed_Title', 
                   save_path='models/', 
                   output_csv='data/topic_modeling_results.csv') -> pd.DataFrame:
        """
        Train topic modeling model and save results with original data + topic columns.
        
        Args:
            data_path (str): Path to processed CSV file
            text_column (str): Column containing processed text
            save_path (str): Directory to save trained models
            output_csv (str): Output file for topic results
            
        Returns:
            pd.DataFrame: DataFrame with original data and topic columns
        """
        try:
            # Load and validate data
            df = self._load_and_validate_data(data_path, text_column)
            texts = df[text_column].astype(str).tolist()
            
            # Create output directory if not exists
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            # Train the selected model
            if self.model_type == 'lda':
                tokenized_texts = [text.split() for text in texts]
                self._train_lda(tokenized_texts, save_path)
                results_df = self._prepare_lda_results(df, texts)
            elif self.model_type == 'bertopic':
                self._train_bertopic(texts, save_path)
                results_df = self._prepare_bertopic_results(df, texts)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Clean and save results
            results_df = self._clean_dataframe(results_df)
            self._save_results(results_df, output_csv)
            self.topic_results = results_df
            
            # Print training summary
            self._print_training_summary()
            
            return results_df

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def _load_and_validate_data(self, data_path: str, text_column: str) -> pd.DataFrame:
        """Load and validate input data."""
        df = pd.read_csv(data_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
            
        if df[text_column].isnull().any():
            logger.warning("Found missing values in text column, filling with empty strings")
            df[text_column] = df[text_column].fillna('')
            
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the resulting dataframe."""
        # Remove any unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Ensure topic columns are properly typed
        if 'Topic' in df.columns:
            df['Topic'] = df['Topic'].astype(int)
            
        return df
    
    def _save_results(self, df: pd.DataFrame, output_path: str):
        """Save results to CSV."""
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    def _print_training_summary(self):
        """Print training summary information."""
        print(f"\nTraining completed for {self.model_type.upper()} model")
        print(f"Coherence score: {self.coherence_score:.4f}")
        
        print("\nTopic Keywords:")
        for topic_id, keywords in self.topic_keywords.items():
            name = self.topic_names.get(topic_id, f"Topic {topic_id}")
            print(f"{name}: {', '.join(keywords[:5])}...")
    
    def _prepare_lda_results(self, original_df: pd.DataFrame, texts: List[str]) -> pd.DataFrame:
        """Prepare LDA results with original data + topic columns."""
        dictionary = Dictionary([text.split() for text in texts])
        corpus = [dictionary.doc2bow(text.split()) for text in texts]
        
        # Get topic distributions and dominant topics
        topic_distributions = [self.model.get_document_topics(bow) for bow in corpus]
        dominant_topics = [
            max(topics, key=lambda x: x[1])[0] if topics else -1 
            for topics in topic_distributions
        ]
        topic_probs = [
            max(topics, key=lambda x: x[1])[1] if topics else 0
            for topics in topic_distributions
        ]
        
        # Store topic keywords
        for topic_id in range(self.model.num_topics):
            self.topic_keywords[topic_id] = [word for word, _ in self.model.show_topic(topic_id)]
            self.topic_names[topic_id] = f"Topic_{topic_id}"
        
        # Add topic columns to original data
        result_df = original_df.copy()
        result_df['Topic'] = dominant_topics
        result_df['Topic_Probability'] = topic_probs
        result_df['Topic_Keywords'] = result_df['Topic'].map(
            lambda x: ', '.join(self.topic_keywords.get(x, ['N/A']))
        )
        result_df['Topic_Name'] = result_df['Topic'].map(self.topic_names)
        
        return result_df
    
    def _prepare_bertopic_results(self, original_df: pd.DataFrame, texts: List[str]) -> pd.DataFrame:
        """Prepare BERTopic results with original data + topic columns."""
        topics, probs = self.model.transform(texts)
        
        # Store topic information
        topic_info = self.model.get_topic_info()
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            self.topic_keywords[topic_id] = [word for word, _ in self.model.get_topic(topic_id)]
            self.topic_names[topic_id] = row['Name']
        
        # Add topic columns to original data
        result_df = original_df.copy()
        result_df['Topic'] = topics
        result_df['Topic_Probability'] = [max(p) if p else 0 for p in probs]
        result_df['Topic_Keywords'] = result_df['Topic'].map(
            lambda x: ', '.join(self.topic_keywords.get(x, ['N/A']))
        )
        result_df['Topic_Name'] = result_df['Topic'].map(
            lambda x: self.topic_names.get(x, 'N/A')
        )
        
        return result_df
            
    def _train_lda(self, tokenized_texts: List[List[str]], save_path: str):
        """Train LDA model."""
        dictionary = Dictionary(tokenized_texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5)  # Filter rare and common words
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        self.model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=10,
            random_state=42,
            passes=10,
            alpha='auto',
            eta='auto'
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=self.model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        self.coherence_score = coherence_model.get_coherence()
        
        # Save model artifacts
        self.model.save(f"{save_path}/lda_model.gensim")
        dictionary.save(f"{save_path}/lda_dictionary.gensim")
        joblib.dump(self.coherence_score, f"{save_path}/lda_coherence.pkl")
        
    def _train_bertopic(self, texts: List[str], save_path: str):
        """Train BERTopic model."""
        self.model = BERTopic(
            language="english", 
            calculate_probabilities=True,
            verbose=False
        )
        
        topics, _ = self.model.fit_transform(texts)
        
        # Calculate average coherence score
        try:
            self.coherence_score = self.model.get_topic_info()['Coherence'].mean()
        except:
            self.coherence_score = 0.5  # Default if coherence can't be calculated
        
        # Save model
        self.model.save(f"{save_path}/bertopic_model")
        joblib.dump(self.coherence_score, f"{save_path}/bertopic_coherence.pkl")

if __name__ == "__main__":
    # Example usage
    trainer = TrainingService(model_type='lda')
    results = trainer.train_model(
        data_path="data/processed_semantic_data.csv",
        text_column="Processed_Title",
        output_csv="data/topic_modeling_results.csv"
    )