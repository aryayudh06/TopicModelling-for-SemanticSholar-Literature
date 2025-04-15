# services/training_service.py
from typing import List, Dict
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from bertopic import BERTopic
from pathlib import Path
import logging

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
        
    def train_model(self, data_path='data/processed_semantic_data.csv', text_column='Processed_Title', 
                   save_path='models/', output_csv='data/topic_modeling_results.csv'):
        """
        Train topic modeling model and save results with original data + topic columns.
        
        Args:
            data_path (str): Path to processed CSV file
            text_column (str): Column containing processed text
            save_path (str): Directory to save trained models
            output_csv (str): Output file for topic results
        """
        try:
            # Load original data
            df = pd.read_csv(data_path)
            texts = df[text_column].astype(str).tolist()
            tokenized_texts = [text.split() for text in texts]
            
            # Create output directory if not exists
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            if self.model_type == 'lda':
                self._train_lda(tokenized_texts, save_path)
                results_df = self._prepare_lda_results(df, texts)
            elif self.model_type == 'bertopic':
                self._train_bertopic(texts, save_path)
                results_df = self._prepare_bertopic_results(df, texts)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Simpan tanpa index dan hapus kolom unnamed jika ada
            result_df = result_df.loc[:, ~result_df.columns.str.contains('^Unnamed')]
            result_df.to_csv(output_csv, index=False)
            self.topic_results = results_df
            
            # Print coherence score and topic keywords
            print(f"\nCoherence score: {self.coherence_score:.4f}")
            print("\nTopic Keywords:")
            for topic_id, keywords in self.topic_keywords.items():
                print(f"Topic {topic_id}: {', '.join(keywords)}")

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
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
        
        # Store topic keywords
        for topic_id in range(self.model.num_topics):
            self.topic_keywords[topic_id] = [word for word, _ in self.model.show_topic(topic_id)]
        
        # Add topic columns to original data
        result_df = original_df.copy()
        result_df['Topic'] = dominant_topics
        result_df['Topic_Keywords'] = result_df['Topic'].map(
            lambda x: ', '.join(self.topic_keywords.get(x, ['N/A']))
        )
        
        return result_df
    
    def _prepare_bertopic_results(self, original_df: pd.DataFrame, texts: List[str]) -> pd.DataFrame:
        """Prepare BERTopic results with original data + topic columns."""
        topics, _ = self.model.transform(texts)
        
        # Store topic keywords
        topic_info = self.model.get_topic_info()
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            self.topic_keywords[topic_id] = [word for word, _ in self.model.get_topic(topic_id)]
        
        # Add topic columns to original data
        result_df = original_df.copy()
        result_df['Topic'] = topics
        result_df['Topic_Keywords'] = result_df['Topic'].map(
            lambda x: ', '.join(self.topic_keywords.get(x, ['N/A']))
        )
        result_df['Topic_Name'] = result_df['Topic'].map(
            lambda x: topic_info[topic_info['Topic'] == x]['Name'].values[0] 
            if x in topic_info['Topic'].values else 'N/A')
        
        return result_df
            
    def _train_lda(self, tokenized_texts, save_path):
        """Train LDA model."""
        dictionary = Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        self.model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=10,
            random_state=42,
            passes=10
        )
        
        coherence_model = CoherenceModel(
            model=self.model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        self.coherence_score = coherence_model.get_coherence()
        
        self.model.save(f"{save_path}/lda_model.gensim")
        dictionary.save(f"{save_path}/lda_dictionary.gensim")
        joblib.dump(self.coherence_score, f"{save_path}/lda_coherence.pkl")
        
    def _train_bertopic(self, texts, save_path):
        """Train BERTopic model."""
        self.model = BERTopic(
            language="english", 
            calculate_probabilities=True,
            verbose=True
        )
        
        topics, _ = self.model.fit_transform(texts)
        self.coherence_score = self.model.get_topic_info()['Coherence'].mean()
        
        self.model.save(f"{save_path}/bertopic_model")
        joblib.dump(self.coherence_score, f"{save_path}/bertopic_coherence.pkl")

if __name__ == "__main__":
    trainer = TrainingService(model_type='lda')
    trainer.train_model(
        data_path="data/processed_semantic_data.csv",
    )