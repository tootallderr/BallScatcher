'''
NRFI Model Enhancement with Lineup Features

This script enhances the NRFI prediction model by integrating batting lineup data.
It adds lineup-based features to the existing model to improve prediction accuracy.

Key Features:
- Loads lineup data from historical and daily sources
- Calculates lineup strength metrics for top of the batting order
- Integrates batter-pitcher matchup statistics
- Enhances the NRFI model with lineup-based features
- Outputs predictions with improved accuracy

Usage:
    python enhance_nrfi_model.py [--train] [--predict]
    
Arguments:
    --train: Train a new enhanced model with lineup features
    --predict: Make predictions for upcoming games
'''

import os
import sys
import joblib
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add script directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NRFI_DIR, setup_logging

# Import lineup integration
from nrfi_lineup_integration import NRFILineupIntegration

# Setup logging
logger = setup_logging('NRFI_Enhanced')

# Define paths
PITCHERS_CSV = os.path.join(NRFI_DIR, "1stInningPitchersHistorical.csv")
BATTERS_CSV = os.path.join(NRFI_DIR, "1stInningBattersHistorical.csv")
LINEUPS_CSV = os.path.join(NRFI_DIR, "probable_lineups_historical.csv")
HISTORICAL_CSV = os.path.join(NRFI_DIR, "F1_historical_schedule.csv")
DAILY_STARTERS_CSV = os.path.join(NRFI_DIR, "F1_daily_starters.csv")
DAILY_LINEUPS_CSV = os.path.join(NRFI_DIR, "F1_daily_lineups.csv")
PROCESSED_BATTER_STATS = os.path.join(NRFI_DIR, "processed_batter_stats.csv")
PROCESSED_PITCHER_STATS = os.path.join(NRFI_DIR, "processed_pitcher_stats.csv")
ENHANCED_MODEL_PATH = os.path.join(NRFI_DIR, "nrfi_lineup_enhanced_model.joblib")
ENHANCED_SCALER_PATH = os.path.join(NRFI_DIR, "nrfi_lineup_enhanced_scaler.joblib")
ENHANCED_PREDICTIONS_PATH = os.path.join(NRFI_DIR, "nrfi_enhanced_predictions.csv")

def verify_files_exist():
    """Verify required files exist"""
    required_files = [
        (HISTORICAL_CSV, "Historical Schedule"),
        (DAILY_STARTERS_CSV, "Daily Starters")
    ]
    
    # Check if processed stats exist, if not check raw files
    if not (os.path.exists(PROCESSED_BATTER_STATS) and os.path.exists(PROCESSED_PITCHER_STATS)):
        required_files.extend([
            (PITCHERS_CSV, "Pitchers Historical"),
            (BATTERS_CSV, "Batters Historical"),
            (LINEUPS_CSV, "Historical Lineups")
        ])
    
    missing_files = []
    for file_path, file_name in required_files:
        if not os.path.exists(file_path):
            missing_files.append(f"{file_name}: {file_path}")
    
    if missing_files:
        logger.error("Missing required files:")
        for missing in missing_files:
            logger.error(f"  - {missing}")
        return False
        
    return True

def init_lineup_integration():
    """Initialize the lineup integration module"""
    logger.info("Initializing lineup integration module...")
    
    # First try to load preprocessed stats
    lineup_integration = NRFILineupIntegration(
        BATTERS_CSV, PITCHERS_CSV, LINEUPS_CSV, HISTORICAL_CSV
    )
    
    # Try to load processed stats
    if os.path.exists(PROCESSED_BATTER_STATS) and os.path.exists(PROCESSED_PITCHER_STATS):
        logger.info("Loading preprocessed stats...")
        if lineup_integration.load_processed_stats(NRFI_DIR):
            return lineup_integration
    
    logger.info("Generating new stats from raw data...")
    return lineup_integration

def load_and_preprocess_data():
    """Load and preprocess the historical game data"""
    # Import here to avoid circular imports
    from NRFI import load_and_preprocess_data as load_nrfi_data
    
    logger.info("Loading and preprocessing historical NRFI data...")
    df = load_nrfi_data(filepath=HISTORICAL_CSV)
    
    return df

def prepare_features(df, lineup_integration):
    """Add lineup features to historical data"""
    logger.info("Adding lineup features to historical data...")
    
    lineup_features = []
    
    # Use tqdm if available for progress tracking
    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="Processing games")
    except ImportError:
        iterator = df.iterrows()
    
    for idx, row in iterator:
        # Get lineup features for this game
        try:
            features = lineup_integration.generate_features(
                game_pk=row['game_pk'],
                game_date=row['date'],
                home_team_id=row['home_team_id'],
                away_team_id=row['away_team_id'],
                home_pitcher_id=row['home_pitcher_id'],
                away_pitcher_id=row['away_pitcher_id']
            )
            lineup_features.append(features)
        except Exception as e:
            logger.error(f"Error generating lineup features for game {row['game_pk']}: {e}")
            # Add default features
            lineup_features.append({
                'home_lineup_woba': 0.320,
                'home_lineup_slg': 0.400,
                'home_lineup_obp': 0.333,
                'home_lineup_barrel_rate': 0.060,
                'home_lineup_iso': 0.150,
                'away_lineup_woba': 0.320,
                'away_lineup_slg': 0.400,
                'away_lineup_obp': 0.333,
                'away_lineup_barrel_rate': 0.060,
                'away_lineup_iso': 0.150,
                'home_batter_advantage': 0.5,
                'away_batter_advantage': 0.5,
                'home_has_known_lineup': 0,
                'away_has_known_lineup': 0
            })
    
    # Convert to DataFrame and join with main dataset
    lineup_df = pd.DataFrame(lineup_features)
    enhanced_df = pd.concat([df, lineup_df], axis=1)
    
    return enhanced_df

def train_enhanced_model(df, lineup_integration):
    """Train an enhanced NRFI model with lineup features"""
    # Import here to avoid circular imports
    from NRFI import train_model as train_nrfi_model
    
    # Add lineup-based features
    enhanced_df = prepare_features(df, lineup_integration)
    
    # Define extended feature columns
    extended_feature_columns = [
        # Original features
        'temperature', 'wind_speed', 'is_dome', 'is_wind_favorable',
        'venue_nrfi_rate', 'venue_home_scoring', 'venue_away_scoring',
        'home_team_scoring_trend', 'away_team_scoring_trend',
        'home_pitcher_nrfi_rate', 'away_pitcher_nrfi_rate',
        'home_pitcher_runs_allowed', 'away_pitcher_runs_allowed',
        'pitcher_matchup_rating',
        
        # New lineup features
        'home_lineup_woba', 'home_lineup_slg', 'home_lineup_obp', 'home_lineup_barrel_rate',
        'away_lineup_woba', 'away_lineup_slg', 'away_lineup_obp', 'away_lineup_barrel_rate',
        'home_batter_advantage', 'away_batter_advantage'
    ]
    
    # Train enhanced model
    logger.info("Training enhanced NRFI model...")
    model, X_train, X_test, y_train, y_test, scaler = train_nrfi_model(
        enhanced_df, feature_columns=extended_feature_columns, 
        return_components=True
    )
    
    # Save the enhanced model and scaler
    logger.info(f"Saving enhanced model to {ENHANCED_MODEL_PATH}")
    joblib.dump(model, ENHANCED_MODEL_PATH)
    joblib.dump(scaler, ENHANCED_SCALER_PATH)
    
    return model, scaler, extended_feature_columns

def make_enhanced_predictions(model, scaler, feature_columns, lineup_integration):
    """Make predictions using the enhanced model"""
    logger.info("Making enhanced predictions for upcoming games...")
    
    if not os.path.exists(DAILY_STARTERS_CSV):
        logger.error(f"Daily starters file not found: {DAILY_STARTERS_CSV}")
        return None
    
    # Load daily starters
    upcoming_games = pd.read_csv(DAILY_STARTERS_CSV)
    
    # Import functions to preprocess daily data
    # This approach avoids importing circular dependencies
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NRFI", "Scripts"))
    from NRFI import create_temp_categories, get_venue_statistics, get_pitcher_statistics
    
    # Basic preprocessing from NRFI.py
    upcoming_games['is_dome'] = upcoming_games['condition'].fillna('').str.contains('Dome|Roof Closed').astype(int)
    upcoming_games['is_wind_favorable'] = upcoming_games['wind_direction'].fillna('').str.contains('In|Out').astype(int)
    upcoming_games['temp_category'] = create_temp_categories(upcoming_games)
    
    # Add venue stats
    venue_stats = get_venue_statistics()
    upcoming_games = upcoming_games.merge(venue_stats, on='venue_name', how='left')
    
    # Add pitcher stats
    for pitcher_type in ['home', 'away']:
        pitcher_stats = get_pitcher_statistics()
        upcoming_games = upcoming_games.merge(
            pitcher_stats, 
            left_on=f'{pitcher_type}_pitcher_id',
            right_on='player_id',
            how='left',
            suffixes=('', f'_{pitcher_type}_pitcher')
        )
        # Rename columns
        upcoming_games[f'{pitcher_type}_pitcher_nrfi_rate'] = upcoming_games['nrfi_rate']
        upcoming_games[f'{pitcher_type}_pitcher_runs_allowed'] = upcoming_games['runs_allowed_per_inning']
        
    # Calculate matchup rating
    upcoming_games['pitcher_matchup_rating'] = upcoming_games.apply(
        lambda x: (x['home_pitcher_nrfi_rate'] + x['away_pitcher_nrfi_rate']) / 2 
        if pd.notna(x.get('home_pitcher_nrfi_rate')) and pd.notna(x.get('away_pitcher_nrfi_rate'))
        else 0.5,
        axis=1
    )
    
    # Add lineup features
    lineup_features = []
    daily_lineups = None
    
    # Check if daily lineups exist
    if os.path.exists(DAILY_LINEUPS_CSV):
        try:
            daily_lineups = pd.read_csv(DAILY_LINEUPS_CSV)
            logger.info(f"Found daily lineups: {len(daily_lineups)} entries")
        except Exception as e:
            logger.warning(f"Error loading daily lineups: {e}")
    
    # Process each game
    for idx, row in upcoming_games.iterrows():
        try:
            # Get home and away lineups if available
            home_lineup = None
            away_lineup = None
            
            if daily_lineups is not None and not daily_lineups.empty:
                # Extract home lineup
                home_data = daily_lineups[
                    (daily_lineups['game_pk'] == row['game_pk']) & 
                    (daily_lineups['team'] == row['home_team_id'])
                ]
                
                if not home_data.empty:
                    home_lineup = {
                        row['batting_order']: row['player_id'] 
                        for _, row in home_data.iterrows()
                    }
                
                # Extract away lineup
                away_data = daily_lineups[
                    (daily_lineups['game_pk'] == row['game_pk']) & 
                    (daily_lineups['team'] == row['away_team_id'])
                ]
                
                if not away_data.empty:
                    away_lineup = {
                        row['batting_order']: row['player_id'] 
                        for _, row in away_data.iterrows()
                    }
            
            # Generate features
            features = lineup_integration.generate_features(
                game_pk=row['game_pk'],
                game_date=row['date'],
                home_team_id=row['home_team_id'],
                away_team_id=row['away_team_id'],
                home_pitcher_id=row['home_pitcher_id'],
                away_pitcher_id=row['away_pitcher_id'],
                known_home_lineup=home_lineup,
                known_away_lineup=away_lineup
            )
            
            lineup_features.append(features)
        except Exception as e:
            logger.error(f"Error processing game {row['game_pk']}: {e}")
            # Add default features
            lineup_features.append({
                'home_lineup_woba': 0.320,
                'home_lineup_slg': 0.400,
                'home_lineup_obp': 0.333,
                'home_lineup_barrel_rate': 0.060,
                'home_lineup_iso': 0.150,
                'away_lineup_woba': 0.320,
                'away_lineup_slg': 0.400,
                'away_lineup_obp': 0.333,
                'away_lineup_barrel_rate': 0.060,
                'away_lineup_iso': 0.150,
                'home_batter_advantage': 0.5,
                'away_batter_advantage': 0.5,
                'home_has_known_lineup': 0,
                'away_has_known_lineup': 0
            })
    
    # Add lineup features to upcoming games DataFrame
    lineup_df = pd.DataFrame(lineup_features)
    upcoming_games = pd.concat([upcoming_games, lineup_df], axis=1)
    
    # Prepare feature matrix for prediction
    features = upcoming_games[feature_columns].fillna(0)
    
    # Scale features (important for proper prediction)
    scaled_features = scaler.transform(features)
    
    # Make predictions
    probs = model.predict_proba(scaled_features)
    upcoming_games['nrfi_probability'] = probs[:, 1]  # Probability of NRFI (class 1)
    upcoming_games['runs_probability'] = probs[:, 0]  # Probability of runs (class 0)
    upcoming_games['prediction'] = model.predict(scaled_features)
    
    # Calculate confidence based on probability distance from 0.5
    upcoming_games['confidence'] = abs(upcoming_games['nrfi_probability'] - 0.5) * 2
    
    # Create prediction label
    upcoming_games['prediction_label'] = upcoming_games.apply(
        lambda x: f"{'NRFI' if x['prediction'] == 1 else 'YRFI'} ({x['confidence']:.1%} confidence)", 
        axis=1
    )
    
    # Create clearer prediction descriptions
    upcoming_games['prediction_desc'] = upcoming_games.apply(
        lambda x: f"Prediction: {'No Runs' if x['prediction'] == 1 else 'Runs Will Score'} in 1st inning\n"
                 f"NRFI Probability: {x['nrfi_probability']:.1%}\n"
                 f"YRFI Probability: {x['runs_probability']:.1%}\n"
                 f"Lineup Strength Factor: {(x['home_lineup_woba'] + x['away_lineup_woba'])/2:.3f}", 
        axis=1
    )
    
    # Sort by confidence and scheduled game time
    upcoming_games = upcoming_games.sort_values(['confidence', 'game_time'], ascending=[False, True])
    
    # Save predictions with key information
    output_columns = [
        'date', 'game_time', 'home_team', 'away_team', 
        'home_pitcher_name', 'away_pitcher_name',
        'venue_name', 'temperature', 'condition',
        'prediction_label', 'prediction_desc', 
        'nrfi_probability', 'runs_probability', 'confidence',
        'venue_nrfi_rate', 'pitcher_matchup_rating',
        'home_lineup_woba', 'away_lineup_woba',
        'home_batter_advantage', 'away_batter_advantage'
    ]
    
    # Save predictions
    logger.info(f"Saving enhanced predictions to {ENHANCED_PREDICTIONS_PATH}")
    upcoming_games[output_columns].to_csv(ENHANCED_PREDICTIONS_PATH, index=False)
    
    # Print summary
    print("\nEnhanced NRFI Prediction Summary:")
    print(f"Total games predicted: {len(upcoming_games)}")
    print(f"NRFI predictions: {(upcoming_games['prediction'] == 1).sum()}")
    print(f"YRFI predictions: {(upcoming_games['prediction'] == 0).sum()}")
    print(f"Average NRFI probability: {upcoming_games['nrfi_probability'].mean():.3f}")
    print(f"Average confidence: {upcoming_games['confidence'].mean():.3f}")
    
    return upcoming_games

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhance NRFI model with lineup features')
    parser.add_argument('--train', action='store_true', help='Train a new enhanced model')
    parser.add_argument('--predict', action='store_true', help='Make predictions for upcoming games')
    args = parser.parse_args()
    
    # Default to both if neither specified
    if not (args.train or args.predict):
        args.train = True
        args.predict = True
    
    # Verify required files exist
    if not verify_files_exist():
        sys.exit(1)
    
    # Initialize lineup integration
    lineup_integration = init_lineup_integration()
    
    model = None
    scaler = None
    feature_columns = None
    
    # Train enhanced model if requested
    if args.train:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Train model
        model, scaler, feature_columns = train_enhanced_model(df, lineup_integration)
        
        print("\nEnhanced NRFI model trained and saved successfully!")
    
    # Make predictions if requested
    if args.predict:
        # Load model if not trained in this session
        if model is None:
            if not os.path.exists(ENHANCED_MODEL_PATH) or not os.path.exists(ENHANCED_SCALER_PATH):
                logger.error("Enhanced model files not found. Train the model first.")
                sys.exit(1)
                
            try:
                logger.info("Loading enhanced model from files...")
                model = joblib.load(ENHANCED_MODEL_PATH)
                scaler = joblib.load(ENHANCED_SCALER_PATH)
                
                # Define feature columns (must match training)
                feature_columns = [
                    # Original features
                    'temperature', 'wind_speed', 'is_dome', 'is_wind_favorable',
                    'venue_nrfi_rate', 'venue_home_scoring', 'venue_away_scoring',
                    'home_team_scoring_trend', 'away_team_scoring_trend',
                    'home_pitcher_nrfi_rate', 'away_pitcher_nrfi_rate',
                    'home_pitcher_runs_allowed', 'away_pitcher_runs_allowed',
                    'pitcher_matchup_rating',
                    
                    # New lineup features
                    'home_lineup_woba', 'home_lineup_slg', 'home_lineup_obp', 'home_lineup_barrel_rate',
                    'away_lineup_woba', 'away_lineup_slg', 'away_lineup_obp', 'away_lineup_barrel_rate',
                    'home_batter_advantage', 'away_batter_advantage'
                ]
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                sys.exit(1)
        
        # Make predictions
        predictions = make_enhanced_predictions(model, scaler, feature_columns, lineup_integration)
        
        if predictions is not None:
            print(f"Enhanced predictions saved to {ENHANCED_PREDICTIONS_PATH}")

if __name__ == "__main__":
    main()
