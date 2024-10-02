import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from ..utils.config import *

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def calculate_form_features(self, df):
        """Calculate team form based on recent results"""
        df['HomeTeamWin'] = (df['FTHG'] > df['FTAG']).astype(int)
        df['AwayTeamWin'] = (df['FTHG'] < df['FTAG']).astype(int)
        df['Draw'] = (df['FTHG'] == df['FTAG']).astype(int)
        
        for window in ROLLING_WINDOWS:
            # Home team form
            df[f'HomeTeamForm_{window}'] = df.groupby('HomeTeam')['HomeTeamWin'].rolling(window=window).mean().reset_index(0, drop=True)
            # Away team form
            df[f'AwayTeamForm_{window}'] = df.groupby('AwayTeam')['AwayTeamWin'].rolling(window=window).mean().reset_index(0, drop=True)

        return df

    def add_goal_features(self, df):
        """Add features related to goals scored and conceded"""
        for window in ROLLING_WINDOWS:
            # Home team goals
            df[f'HomeTeamScoringRate_{window}'] = df.groupby('HomeTeam')['FTHG'].rolling(window=window).mean().reset_index(0, drop=True)
            df[f'HomeTeamConcedingRate_{window}'] = df.groupby('HomeTeam')['FTAG'].rolling(window=window).mean().reset_index(0, drop=True)
            
            # Away team goals
            df[f'AwayTeamScoringRate_{window}'] = df.groupby('AwayTeam')['FTAG'].rolling(window=window).mean().reset_index(0, drop=True)
            df[f'AwayTeamConcedingRate_{window}'] = df.groupby('AwayTeam')['FTHG'].rolling(window=window).mean().reset_index(0, drop=True)

        return df

    def add_head_to_head_features(self, df):
        """Add head-to-head statistics"""
        h2h_stats = df.groupby(['HomeTeam', 'AwayTeam']).agg({
            'FTHG': ['mean', 'std', 'count'],
            'FTAG': ['mean', 'std'],
            'HomeTeamWin': 'mean',
            'AwayTeamWin': 'mean',
            'Draw': 'mean'
        }).reset_index()
        
        h2h_stats.columns = ['HomeTeam', 'AwayTeam', 'H2H_HomeGoals_Mean', 'H2H_HomeGoals_Std', 'H2H_Matches',
                            'H2H_AwayGoals_Mean', 'H2H_AwayGoals_Std', 'H2H_HomeWinRate', 'H2H_AwayWinRate', 'H2H_DrawRate']
        
        return df.merge(h2h_stats, on=['HomeTeam', 'AwayTeam'], how='left')

    def add_season_stats(self, df):
        """Add season-level statistics"""
        df['Season'] = pd.to_datetime(df['Date']).dt.year
        
        season_stats = df.groupby(['Season', 'HomeTeam']).agg({
            'FTHG': 'mean',
            'HomeTeamWin': 'mean'
        }).reset_index()
        
        season_stats.columns = ['Season', 'Team', 'SeasonHomeGoals', 'SeasonHomeWinRate']
        
        # Merge for home teams
        df = df.merge(season_stats, left_on=['Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')
        df = df.drop('Team', axis=1)
        
        # Merge for away teams
        season_stats.columns = ['Season', 'Team', 'SeasonAwayGoals', 'SeasonAwayWinRate']
        df = df.merge(season_stats, left_on=['Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')
        df = df.drop('Team', axis=1)
        
        return df

    def engineer_features(self, input_file=None):
        """Main feature engineering function"""
        logger.info("Starting feature engineering process...")
        
        if input_file is None:
            input_file = os.path.join(PROCESSED_DATA_DIR, 'processed_historical_data.csv')
        
        try:
            df = pd.read_csv(input_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        except Exception as e:
            logger.error(f"Error loading input file: {str(e)}")
            return None
        
        # Add all features
        df = self.calculate_form_features(df)
        df = self.add_goal_features(df)
        df = self.add_head_to_head_features(df)
        df = self.add_season_stats(df)
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Select features for scaling
        feature_columns = [col for col in df.columns if any(x in col for x in 
            ['Form', 'ScoringRate', 'ConcedingRate', 'H2H_', 'Season'])]
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        # Save engineered features
        output_file = os.path.join(PROCESSED_DATA_DIR, 'engineered_features.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"Feature engineering completed. Saved to {output_file}")
        
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engineer = FeatureEngineer()
    engineered_data = engineer.engineer_features()
    if engineered_data is not None:
        logger.info(f"Feature engineering completed successfully. Shape: {engineered_data.shape}")
