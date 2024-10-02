import pandas as pd
import numpy as np
import joblib
import logging
from ..utils.config import MODEL_DIR, PROCESSED_DATA_DIR
import os

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature columns"""
        try:
            model_path = os.path.join(MODEL_DIR, 'xgboost_model.joblib')
            feature_path = os.path.join(MODEL_DIR, 'feature_columns.joblib')
            
            self.model = joblib.load(model_path)
            self.feature_columns = joblib.load(feature_path)
            
            logger.info("Model and features loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def predict_match(self, match_features):
        """Predict the outcome of a single match"""
        if self.model is None:
            logger.error("Model not loaded!")
            return None
        
        try:
            # Ensure all required features are present
            missing_features = [col for col in self.feature_columns if col not in match_features.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                return None
            
            # Select only the required features in the correct order
            X = match_features[self.feature_columns]
            
            # Make prediction
            prob = self.model.predict_proba(X)[0]
            prediction = self.model.predict(X)[0]
            
            return {
                'home_win_probability': prob[1],
                'away_win_probability': 1 - prob[1],
                'predicted_winner': 'Home Team' if prediction == 1 else 'Away Team'
            }
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None
    
    def predict_upcoming_fixtures(self, fixtures_file):
        """Predict outcomes for upcoming fixtures"""
        try:
            # Load upcoming fixtures
            fixtures_path = os.path.join(PROCESSED_DATA_DIR, fixtures_file)
            fixtures_df = pd.read_csv(fixtures_path)
            
            predictions = []
            for _, fixture in fixtures_df.iterrows():
                prediction = self.predict_match(pd.DataFrame([fixture]))
                if prediction:
                    predictions.append({
                        'home_team': fixture['HomeTeam'],
                        'away_team': fixture['AwayTeam'],
                        'date': fixture['Date'],
                        **prediction
                    })
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error predicting upcoming fixtures: {str(e)}")
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = Predictor()
    
    # Example: Predict upcoming fixtures
    predictions = predictor.predict_upcoming_fixtures('upcoming_fixtures_processed.csv')
    if predictions:
        for pred in predictions:
            logger.info(f"{pred['home_team']} vs {pred['away_team']}: {pred['predicted_winner']} ({pred['home_win_probability']:.2f})")
