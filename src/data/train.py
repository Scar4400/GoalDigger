import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import logging
from ..utils.config import MODEL_DIR, MODEL_PARAMS, PROCESSED_DATA_DIR
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Define target
        df['Target'] = (df['FTHG'] > df['FTAG']).astype(int)
        
        # Select features
        self.feature_columns = [col for col in df.columns if any(x in col for x in 
            ['Form', 'ScoringRate', 'ConcedingRate', 'H2H_', 'Season'])]
        
        X = df[self.feature_columns]
        y = df['Target']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, X_test, y_train, y_test):
        """Train the XGBoost model"""
        logger.info("Training model...")
        
        self.model = xgb.XGBClassifier(**MODEL_PARAMS)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Model Performance:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_scores': cv_scores
        }

    def save_model(self):
        """Save the trained model and feature columns"""
        if self.model is None:
            logger.error("No model to save!")
            return False
        
        model_path = os.path.join(MODEL_DIR, 'xgboost_model.joblib')
        feature_path = os.path.join(MODEL_DIR, 'feature_columns.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_columns, feature_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Feature columns saved to {feature_path}")
        return True

    def train(self):
        """Main training function"""
        try:
            # Load engineered features
            data_path = os.path.join(PROCESSED_DATA_DIR, 'engineered_features.csv')
            df = pd.read_csv(data_path)
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(df)
            
            # Train model and get metrics
            metrics = self.train_model(X_train, X_test, y_train, y_test)
            
            # Save model
            self.save_model()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = ModelTrainer()
    metrics = trainer.train()
    if metrics:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")
