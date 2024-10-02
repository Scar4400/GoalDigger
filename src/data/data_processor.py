import pandas as pd
import numpy as np
from ..utils.config import *
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.required_files = [
            (PREMIER_LEAGUE_2024_FILE, ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']),
            (PLAYER_STATS_FILE, ['Player', 'Team', 'Goals', 'Assists']),
            (PREMIER_22_23_FILE, ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']),
            (PREMIER_21_22_FILE, ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        ]

    def load_and_validate_data(self):
        """Load all CSV files and validate their structure"""
        datasets = {}
        for file_path, required_columns in self.required_files:
            try:
                df = pd.read_csv(file_path)
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.error(f"Missing columns in {file_path}: {missing_columns}")
                    continue
                datasets[os.path.basename(file_path)] = df
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        return datasets

    def standardize_team_names(self, df):
        """Standardize team names across different datasets"""
        team_name_mapping = {
            'Manchester United': 'Man United',
            'Manchester City': 'Man City',
            'Tottenham': 'Tottenham Hotspur',
            'Newcastle': 'Newcastle United',
            # Add more mappings as needed
        }
        if 'HomeTeam' in df.columns:
            df['HomeTeam'] = df['HomeTeam'].replace(team_name_mapping)
        if 'AwayTeam' in df.columns:
            df['AwayTeam'] = df['AwayTeam'].replace(team_name_mapping)
        if 'Team' in df.columns:
            df['Team'] = df['Team'].replace(team_name_mapping)
        return df

    def merge_historical_data(self, datasets):
        """Merge historical match data from different seasons"""
        historical_data = []
        for filename, df in datasets.items():
            if filename in ['Premier22_23.csv', 'Premier21-22.csv', 'PremierLeagueSeason2024.csv']:
                df = self.standardize_team_names(df)
                historical_data.append(df)
        
        merged_data = pd.concat(historical_data, ignore_index=True)
        merged_data['Date'] = pd.to_datetime(merged_data['Date'])
        merged_data = merged_data.sort_values('Date')
        return merged_data

    def add_player_stats(self, match_data, player_stats):
        """Add aggregated player statistics to match data"""
        player_stats = self.standardize_team_names(player_stats)
        team_stats = player_stats.groupby('Team').agg({
            'Goals': 'sum',
            'Assists': 'sum'
        }).reset_index()
        
        match_data = match_data.merge(team_stats, left_on='HomeTeam', right_on='Team', how='left')
        match_data = match_data.rename(columns={'Goals': 'HomeTeamGoals', 'Assists': 'HomeTeamAssists'})
        match_data = match_data.drop('Team', axis=1)
        
        match_data = match_data.merge(team_stats, left_on='AwayTeam', right_on='Team', how='left')
        match_data = match_data.rename(columns={'Goals': 'AwayTeamGoals', 'Assists': 'AwayTeamAssists'})
        match_data = match_data.drop('Team', axis=1)
        
        return match_data

    def process_data(self):
        """Main data processing function"""
        logger.info("Starting data processing...")
        
        # Load and validate all datasets
        datasets = self.load_and_validate_data()
        if not datasets:
            logger.error("No valid datasets found")
            return None
        
        # Merge historical match data
        historical_data = self.merge_historical_data(datasets)
        logger.info(f"Merged historical data shape: {historical_data.shape}")
        
        # Add player statistics
        if 'Player_Stats_22-23.csv' in datasets:
            historical_data = self.add_player_stats(historical_data, datasets['Player_Stats_22-23.csv'])
        
        # Save processed data
        output_file = os.path.join(PROCESSED_DATA_DIR, 'processed_historical_data.csv')
        historical_data.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
        
        return historical_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = DataProcessor()
    processed_data = processor.process_data()
    if processed_data is not None:
        logger.info(f"Data processing completed successfully. Shape: {processed_data.shape}")
