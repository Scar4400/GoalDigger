import logging
import pandas as pd
from src.data.data_processor import DataProcessor
from src.data.feature_engineer import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.predict import Predictor
from src.utils.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # 1. Process raw data
        logger.info("Processing raw data...")
        processor = DataProcessor()
        processed_data = processor.process_data()
        if processed_data is None:
            logger.error("Data processing failed!")
            return
        
        # 2. Engineer features
        logger.info("Engineering features...")
        engineer = FeatureEngineer()
        engineered_data = engineer.engineer_features()
        if engineered_data is None:
            logger.error("Feature engineering failed!")
            return
        
        # 3. Train model
        logger.info("Training model...")
        trainer = ModelTrainer()
        metrics = trainer.train()
        if metrics is None:
            logger.error("Model training failed!")
            return
        
        # 4. Make predictions for upcoming fixtures
        logger.info("Making predictions...")
        predictor = Predictor()
        predictions = predictor.predict_upcoming_fixtures('upcoming_fixtures_processed.csv')
        
        if predictions:
            logger.info("Predictions for upcoming matches:")
            for pred in predictions:
                logger.info(f"{pred['home_team']} vs {pred['away_team']}")
                logger.info(f"Predicted winner: {pred['predicted_winner']}")
                logger.info(f"Home win probability: {pred['home_win_probability']:.2f}")
                logger.info("---")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
