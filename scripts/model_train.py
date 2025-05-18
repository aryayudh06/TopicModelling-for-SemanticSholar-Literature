from typing import List, Dict
import os
import pandas as pd
import numpy as np
import joblib
from bertopic import BERTopic
from pathlib import Path
import logging
import warnings
import mlflow
import mlflow.sklearn
from datetime import datetime
from prometheus_client import start_http_server, Summary, Gauge, Counter, Histogram
import prometheus_client
import time

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class BERTopicTrainingService:
    def __init__(self):
        """Initialize BERTopic training service."""
        self.model = None
        self.coherence_score = None
        self.topic_keywords = {}  # To store topic keywords
        self.topic_names = {}     # To store topic names
        
        # Initialize metrics
        self._init_prometheus_metrics()
        
        # Start Prometheus metrics server
        start_http_server(8000)
        
        # MLflow setup
        self._init_mlflow()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.training_duration = Summary(
            'bertopic_training_seconds', 
            'Time spent training BERTopic model'
        )
        self.coherence_score_gauge = Gauge(
            'bertopic_coherence_score',
            'Model coherence score'
        )
        self.topic_count_gauge = Gauge(
            'bertopic_topic_count',
            'Number of topics generated'
        )
        self.text_length_histogram = Histogram(
            'bertopic_text_length',
            'Distribution of text lengths',
            buckets=[0, 50, 100, 200, 500, 1000]
        )
        self.training_counter = Counter(
            'bertopic_training_runs',
            'Number of training runs'
        )
        self.training_errors = Counter(
            'bertopic_training_errors',
            'Number of training errors'
        )
        self.data_loading_time = Summary(
            'bertopic_data_loading_seconds',
            'Time spent loading data'
        )
    
    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        self.experiment_name = f"BERTopic_Modeling_{datetime.now().strftime('%Y%m%d')}"
        mlflow.set_experiment(self.experiment_name)
        mlflow.set_tracking_uri("http://localhost:5001")  # MLflow server URI
        
    def train_model(self, data_path: str = 'data/processed_semantic_data.csv', 
                   text_column: str = 'Processed_Title', 
                   save_path: str = 'models/', 
                   output_csv: str = 'data/topic_modeling_results.csv') -> pd.DataFrame:
        """
        Train BERTopic model with MLflow and Prometheus tracking.
        """
        self.training_counter.inc()
        start_time = time.time()
        
        try:
            with mlflow.start_run(run_name=f"bertopic_run_{datetime.now().strftime('%H%M%S')}"):
                # Track training duration
                with self.training_duration.time():
                    # Log parameters to both systems
                    self._log_parameters(data_path, text_column, save_path)
                    
                    # Load and validate data with metrics
                    df = self._load_data_with_metrics(data_path, text_column)
                    texts = df[text_column].astype(str).tolist()
                    
                    # Record text lengths
                    for text in texts:
                        self.text_length_histogram.observe(len(text))
                    
                    # Train model
                    self._train_bertopic(texts, save_path)
                    
                    # Prepare results
                    results_df = self._prepare_results(df, texts)
                    results_df = self._clean_dataframe(results_df)
                    self._save_results(results_df, output_csv)
                    
                    # Log metrics to both systems
                    self._log_metrics(save_path, output_csv)
                    
                    # Print summary
                    self._print_training_summary()
                    
                    return results_df
                    
        except Exception as e:
            self.training_errors.inc()
            logger.error(f"Error during training: {str(e)}")
            mlflow.log_param("error", str(e))
            raise
        finally:
            # Log total execution time
            mlflow.log_metric("total_execution_seconds", time.time() - start_time)
    
    def _load_data_with_metrics(self, data_path: str, text_column: str) -> pd.DataFrame:
        """Load data with performance metrics."""
        with self.data_loading_time.time():
            df = pd.read_csv(data_path)
            
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in data")
                
            if df[text_column].isnull().any():
                logger.warning("Found missing values in text column, filling with empty strings")
                df[text_column] = df[text_column].fillna('')
                
            return df
    
    def _log_parameters(self, data_path: str, text_column: str, save_path: str):
        """Log parameters to both MLflow and as metrics."""
        params = {
            "model_type": "bertopic",
            "data_path": data_path,
            "text_column": text_column,
            "save_path": save_path
        }
        mlflow.log_params(params)
        
        # Could add parameters as labels to Prometheus metrics if needed
    
    def _log_metrics(self, save_path: str, output_csv: str):
        """Log metrics to both systems."""
        # Update Prometheus gauges
        self.coherence_score_gauge.set(self.coherence_score)
        self.topic_count_gauge.set(len(self.topic_keywords))
        
        # Log to MLflow
        mlflow.log_metrics({
            "coherence_score": self.coherence_score,
            "num_topics": len(self.topic_keywords)
        })
        
        # Log artifacts
        if os.path.exists(output_csv):
            mlflow.log_artifact(output_csv)
        
        eval_file = f"{save_path}/model_evaluations.csv"
        self._save_evaluation_results(save_path)
        if os.path.exists(eval_file):
            mlflow.log_artifact(eval_file)
    
    # [Keep all other existing methods unchanged...]
    
    def _load_and_validate_data(self, data_path: str, text_column: str) -> pd.DataFrame:
        """Load and validate input data."""
        df = pd.read_csv(data_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
            
        if df[text_column].isnull().any():
            logger.warning("Found missing values in text column, filling with empty strings")
            df[text_column] = df[text_column].fillna('')
            
        return df
    
    def _save_evaluation_results(self, save_path: str = "models"):
        """Save evaluation results to CSV file."""
        eval_data = {
            'Model_Type': ["bertopic"],
            'Coherence_Score': [self.coherence_score],
            'Num_Topics': [len(self.topic_keywords)],
            'Model_Path': [f"{save_path}/bertopic_model"],
            'Training_Date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        
        # Add topic information
        for topic_id, keywords in self.topic_keywords.items():
            eval_data[f'Topic_{topic_id}_Keywords'] = [', '.join(keywords[:5])]
            eval_data[f'Topic_{topic_id}_Name'] = [self.topic_names.get(topic_id, f"Topic_{topic_id}")]
        
        eval_df = pd.DataFrame(eval_data)
        eval_file = f"{save_path}/model_evaluations.csv"
        
        if os.path.exists(eval_file):
            existing_df = pd.read_csv(eval_file)
            eval_df = pd.concat([existing_df, eval_df], ignore_index=True)
        
        eval_df.to_csv(eval_file, index=False)
        logger.info(f"Evaluation results saved to {eval_file}")
    
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
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise
    
    def _print_training_summary(self):
        """Print training summary information."""
        print(f"\nTraining completed for BERTopic model")
        print(f"Coherence score: {self.coherence_score:.4f}")
        print(f"Number of topics: {len(self.topic_keywords)}")
        
        print("\nTopic Keywords:")
        for topic_id, keywords in self.topic_keywords.items():
            name = self.topic_names.get(topic_id, f"Topic {topic_id}")
            print(f"{name}: {', '.join(keywords[:5])}...")
    
    def _prepare_results(self, original_df: pd.DataFrame, texts: List[str]) -> pd.DataFrame:
        """Prepare BERTopic results with original data + topic columns."""
        topics, probs = self.model.transform(texts)
        
        # Store topic information
        topic_info = self.model.get_topic_info()
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            self.topic_keywords[topic_id] = [word for word, _ in self.model.get_topic(topic_id)]
            self.topic_names[topic_id] = row['Name']
        
        # Handle probabilities
        topic_probs = [np.max(p) if p.size > 0 else 0 for p in probs] if probs is not None else [0] * len(topics)
        
        # Add topic columns to original data
        result_df = original_df.copy()
        result_df['Topic'] = topics
        result_df['Topic_Probability'] = topic_probs
        result_df['Topic_Keywords'] = result_df['Topic'].map(
            lambda x: ', '.join(self.topic_keywords.get(x, ['N/A']))
        )
        result_df['Topic_Name'] = result_df['Topic'].map(
            lambda x: self.topic_names.get(x, 'N/A')
        )
        
        return result_df
            
    def _train_bertopic(self, texts: List[str], save_path: str):
        """Train BERTopic model with MLflow tracking."""
        try:
            bertopic_params = {
                "language": "english",
                "calculate_probabilities": True,
                "verbose": False
            }
            mlflow.log_params(bertopic_params)
            
            self.model = BERTopic(**bertopic_params)
            topics, _ = self.model.fit_transform(texts)
            
            # Calculate coherence score
            try:
                self.coherence_score = self.model.get_topic_info()['Coherence'].mean()
            except KeyError:
                self.coherence_score = 0.5  # default fallback
                logger.warning("Coherence score column not found, using default 0.5")
            except Exception as e:
                self.coherence_score = 0.5
                logger.error(f"Failed to calculate coherence score: {str(e)}")
            
            # Log topic information
            topic_info = self.model.get_topic_info()
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                topic_words = [word for word, _ in self.model.get_topic(topic_id)]
                self.topic_keywords[topic_id] = topic_words
                self.topic_names[topic_id] = row['Name']
                mlflow.log_text("\n".join(topic_words), f"topic_{topic_id}_keywords.txt")
            
            # Save model
            model_path = f"{save_path}/bertopic_model"
            self.model.save(model_path)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                self.model,
                "bertopic_model",
                registered_model_name=f"bertopic_{datetime.now().strftime('%Y%m%d')}",
                pip_requirements=["bertopic"]
            )
            
            # Save coherence score
            joblib.dump(self.coherence_score, f"{save_path}/bertopic_coherence.pkl")
            
        except Exception as e:
            logger.error(f"BERTopic training failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize and train BERTopic model
    trainer = BERTopicTrainingService()
    results = trainer.train_model(
        data_path="data/processed_semantic_data.csv",
        text_column="Processed_Title",
        output_csv="data/topic_results.csv"
    )