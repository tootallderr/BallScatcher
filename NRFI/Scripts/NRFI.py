import os
import time
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

from default_tracker import DefaultTracker
# Initialize default tracker
default_tracker = DefaultTracker()

# Convenience functions for data quality tracking
def track_default(feature_name, default_count, total_count, group=None):
    default_tracker.track_default(feature_name, default_count, total_count, group)
    
def track_missing_file(file_path, group=None):
    default_tracker.track_missing_file(file_path, group)
    
def reset_defaults_tracking():
    default_tracker.reset()
    
def print_defaults_report():
    default_tracker.print_report()

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import multiprocessing  # to get the number of CPU cores
from datetime import datetime, timedelta
from tqdm import tqdm  # For progress bars
import random
import json
import ast  # For parsing string representations of lists
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import sys

# Add parent directory to Python path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NRFI_DIR, VISUALS_DIR
from config import (DATA_DIR, NRFI_DIR, VISUALS_DIR, setup_logging)

# Setup logging using centralized configuration
logger = setup_logging('NRFI_Model')

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Get optimal number of jobs based on available CPU cores
n_jobs_limit = max(1, multiprocessing.cpu_count() - 1)

# Define paths using config variables
PROCESSED_DATA_PATH = os.path.join(NRFI_DIR, "processed_mlb_data.csv")
MODEL_SAVE_PATH = os.path.join(NRFI_DIR, "nrfi_voting_classifier_model.joblib")
SCALER_SAVE_PATH = os.path.join(NRFI_DIR, "nrfi_scaler.joblib")
FEATURE_IMPORTANCE_PLOT_PATH = os.path.join(VISUALS_DIR, "feature_importance.png")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(VISUALS_DIR, "confusion_matrix.png")
ROC_CURVE_PLOT_PATH = os.path.join(VISUALS_DIR, "roc_curve.png")
PREDICTIONS_OUTPUT_PATH = os.path.join(NRFI_DIR, "F1_predictions.csv")

# Define paths for the new data files
BATTERS_FIRST_INNING_PATH = os.path.join(NRFI_DIR, "1stInningBattersHistorical.csv")
PITCHERS_FIRST_INNING_PATH = os.path.join(NRFI_DIR, "1stInningPitchersHistorical.csv")
HISTORICAL_SCHEDULE_PATH = os.path.join(NRFI_DIR, "F1_historical_schedule.csv") 
DAILY_STARTERS_PATH = os.path.join(NRFI_DIR, "F1_daily_starters.csv")
FIRST_INNING_SCORES_PATH = os.path.join(NRFI_DIR, "F1_inning_scores.csv")
LINEUPS_HISTORICAL_PATH = os.path.join(NRFI_DIR, "probable_lineups_historical.csv")

# Ensure necessary directories exist
os.makedirs(NRFI_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

# Calculate maximum jobs to use 70% of available cores
n_jobs_limit = max(1, int(multiprocessing.cpu_count() * 0.7))
print(f"Limiting n_jobs to {n_jobs_limit} (70% of available cores)")

def create_temp_categories(df):
    """Create temperature categories consistently"""
    return pd.cut(df['temperature'], 
                  bins=[0, 40, 60, 75, 100],
                  labels=['cold', 'mild', 'warm', 'hot'])

def track_processing_step(step_name, start_time=None):
    """Track processing steps with timing"""
    if (start_time):
        elapsed = time.time() - start_time
        print(f"\n{step_name} completed in {elapsed:.2f} seconds")
    print(f"\nStarting: {step_name}...")
    return time.time()

def track_feature_stats(df, features, step_name):
    """Track feature statistics at each step"""
    print(f"\nFeature statistics after {step_name}:")
    stats = df[features].describe()
    print(stats)
    
    # Track correlations with target if 'nrfi' exists
    if 'nrfi' in df.columns and features:
        correlations = df[features].corrwith(df['nrfi']).sort_values(ascending=False)
        print("\nFeature correlations with NRFI target:")
        print(correlations)
    
    return stats

def calculate_recent_performance(df, team_id, game_date, n_games=5):
    """Calculate team's performance in last N games"""
    start_time = track_processing_step(f"Calculating recent performance for team {team_id}")
    
    team_games = df[
        ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
        (df['date'] < game_date)
    ].sort_values('date', ascending=False).head(n_games)
    
    if team_games.empty:
        print(f"No recent games found for team {team_id}")
        return {
            'last_n_nrfi_rate': 0.5,
            'last_n_runs_scored': 0,
            'last_n_runs_allowed': 0
        }
    
    track_nan_counts(team_games, f"team {team_id} recent games")
    
    nrfi_rate = team_games['nrfi'].mean()
    runs_scored = []
    runs_allowed = []
    
    for _, game in team_games.iterrows():
        if game['home_team_id'] == team_id:
            runs_scored.append(game['home_inning_1_runs'])
            runs_allowed.append(game['away_inning_1_runs'])
        else:
            runs_scored.append(game['away_inning_1_runs'])
            runs_allowed.append(game['home_inning_1_runs'])
    
    track_processing_step("Recent performance calculation", start_time)
    return {
        'last_n_nrfi_rate': nrfi_rate,
        'last_n_runs_scored': np.mean(runs_scored),
        'last_n_runs_allowed': np.mean(runs_allowed)
    }

def calculate_head_to_head_stats(df, team1_id, team2_id, game_date, lookback_days=365):
    """Calculate historical matchup statistics between two teams"""
    game_date = pd.to_datetime(game_date)
    lookback_date = game_date - pd.Timedelta(days=lookback_days)
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    matchups = df[
        (df['date'] >= lookback_date) &
        (df['date'] < game_date) &
        (
            ((df['home_team_id'] == team1_id) & (df['away_team_id'] == team2_id)) |
            ((df['home_team_id'] == team2_id) & (df['away_team_id'] == team1_id))
        )
    ]
    
    if matchups.empty:
        return {
            'h2h_nrfi_rate': 0.5,
            'h2h_total_games': 0,
            'h2h_avg_total_runs': 0
        }
    
    total_games = len(matchups)
    nrfi_rate = matchups['nrfi'].mean()
    avg_total_runs = matchups.apply(
        lambda x: x['home_inning_1_runs'] + x['away_inning_1_runs'], axis=1
    ).mean()
    
    return {
        'h2h_nrfi_rate': nrfi_rate,
        'h2h_total_games': total_games,
        'h2h_avg_total_runs': avg_total_runs
    }

def calculate_streak_features(df, team_id, game_date):
    """Calculate streak-based features for a team"""
    team_games = df[
        ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
        (pd.to_datetime(df['date']) < pd.to_datetime(game_date))
    ].sort_values('date', ascending=False)
    
    if team_games.empty:
        return {
            'current_nrfi_streak': 0,
            'longest_nrfi_streak': 0,
            'nrfi_momentum': 0
        }
    
    # Calculate current NRFI streak
    current_streak = 0
    for _, game in team_games.iterrows():
        if game['nrfi'] == 1:
            current_streak += 1
        else:
            break
    
    # Calculate longest NRFI streak in last 20 games
    longest_streak = 0
    current_long = 0
    for _, game in team_games.head(20).iterrows():
        if game['nrfi'] == 1:
            current_long += 1
            longest_streak = max(longest_streak, current_long)
        else:
            current_long = 0
    
    # Calculate momentum (weighted average of last 5 games)
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # More recent games weighted higher
    momentum = 0
    last_5 = team_games.head(5)
    if not last_5.empty:
        momentum = sum(game.nrfi * weight for game, weight in zip(last_5.itertuples(), weights[:len(last_5)]))
    
    return {
        'current_nrfi_streak': current_streak,
        'longest_nrfi_streak': longest_streak,
        'nrfi_momentum': momentum
    }

def track_nan_counts(df, step_name):
    """Track NaN counts for each column at a given processing step"""
    nan_counts = df.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]
    if not nan_columns.empty:
        print(f"\nNaN counts after {step_name}:")
        for col, count in nan_columns.items():
            print(f"{col}: {count} NaNs ({(count/len(df))*100:.1f}% of data)")
    return nan_columns

def load_and_preprocess_data(filepath=HISTORICAL_SCHEDULE_PATH, cutoff_date=None):
    """Load and preprocess the historical game data with enhanced features from all available sources"""
    # Reset the default tracker for a fresh data load
    from default_tracker import default_tracker
    default_tracker.reset()
    
    print("Loading historical schedule data...")
    if not os.path.exists(filepath):
        print(f"ERROR: Historical schedule file not found at {filepath}")
        default_tracker.track_missing_file(filepath, group='main_data')
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    track_nan_counts(df, "initial load")
    
    # Check for critical fields
    critical_fields = ['home_inning_1_runs', 'away_inning_1_runs', 'date', 
                      'home_team_id', 'away_team_id', 'home_pitcher_id', 'away_pitcher_id']
    
    missing_fields = [field for field in critical_fields if field not in df.columns]
    if missing_fields:
        print(f"ERROR: Critical fields missing from historical data: {missing_fields}")
        for field in missing_fields:
            default_tracker.track_default(f'missing_{field}', 1, 1, group='main_data')
    
    print("\nConverting dates...")
    df['date'] = pd.to_datetime(df['date'])
    
    print("\nCalculating basic features...")
    # Basic feature calculations
    df['nrfi'] = ((df['home_inning_1_runs'] + df['away_inning_1_runs']) == 0).astype(int)
    
    # Track weather data completeness
    weather_fields = ['condition', 'wind_direction', 'temperature']
    weather_fields_exist = [field for field in weather_fields if field in df.columns]
    
    if weather_fields_exist:
        missing_weather = df[weather_fields_exist].isna().any(axis=1).sum()
        if missing_weather > 0:
            print(f"Warning: {missing_weather} games ({missing_weather/len(df):.1%}) missing weather data")
            default_tracker.track_default('weather_data_missing', missing_weather, len(df), group='weather_data')
    
    df['is_dome'] = df['condition'].fillna('').str.contains('Dome|Roof Closed').astype(int)
    df['is_wind_favorable'] = df['wind_direction'].fillna('').str.contains('In|Out').astype(int)
    df['temp_category'] = create_temp_categories(df)
    track_nan_counts(df, "basic feature calculation")
    
    # Load additional data sources
    print("\nLoading additional first inning data sources...")
    pitchers_df = load_pitchers_first_inning_data()
    batters_df = load_batters_first_inning_data()
    lineups_df = load_lineups_historical_data()
    scores_df = load_first_inning_scores()
    
    # Track data completeness
    data_files_status = {
        'pitcher_data': pitchers_df is not None,
        'batter_data': batters_df is not None,
        'lineup_data': lineups_df is not None,
        'scores_data': scores_df is not None
    }
    
    missing_data_files = [name for name, status in data_files_status.items() if not status]
    if missing_data_files:
        print(f"WARNING: Missing data sources: {missing_data_files}")
    
    # Calculate advanced statistics from these sources
    pitcher_stats = calculate_pitcher_first_inning_stats(pitchers_df) 
    batter_stats = calculate_batter_first_inning_stats(batters_df)
    
    # Calculate lineup strength if we have lineup and batter data
    if lineups_df is not None and batter_stats is not None:
        lineups_df = calculate_lineup_strength(lineups_df, batter_stats)
    else:
        print("WARNING: Cannot calculate lineup strength - missing lineup or batter data")
        default_tracker.track_default('lineup_strength_missing', 1, 1, group='feature_generation')
        
    # Create pitcher-batter matchup features and add to main dataframe
    if pitcher_stats is not None and batter_stats is not None and lineups_df is not None:
        print("\nCreating pitcher-batter matchup features...")
        df = create_pitcher_batter_matchup_features(df, pitcher_stats, batter_stats, lineups_df)
    else:
        print("WARNING: Cannot create pitcher-batter matchup features - missing required data")
        default_tracker.track_default('matchup_features_missing', 1, 1, group='feature_generation')
    
    # Print data quality report at the end of processing
    default_tracker.print_report()
    
    print("\nProcessing venue statistics...")
    # Venue statistics
    venue_stats = df.groupby('venue_name').agg({
        'nrfi': 'mean',
        'home_inning_1_runs': 'mean',
        'away_inning_1_runs': 'mean'
    }).reset_index()
    venue_stats.columns = ['venue_name', 'venue_nrfi_rate', 'venue_home_scoring', 'venue_away_scoring']
    df = df.merge(venue_stats, on='venue_name', how='left')
    track_nan_counts(df, "venue statistics merge")
    
    print("\nProcessing team statistics...")
    # Team statistics
    nan_counts_before = df.isna().sum()
    for team_type in ['home', 'away']:
        team_stats = []
        for team_id in df[f'{team_type}_team_id'].unique():
            team_df = df[df[f'{team_type}_team_id'] == team_id].sort_values('date')
            rolling_stats = pd.DataFrame({
                'date': team_df['date'],
                f'{team_type}_team_id': team_id,
                f'{team_type}_last_5_nrfi_rate': team_df['nrfi'].rolling(5, min_periods=1).mean(),
                f'{team_type}_last_5_runs_scored': team_df[f'{team_type}_inning_1_runs'].rolling(5, min_periods=1).mean(),
                f'{team_type}_last_5_runs_allowed': team_df[f'{team_type}_inning_1_runs'].rolling(5, min_periods=1).mean(),
                f'{team_type}_team_scoring_trend': team_df[f'{team_type}_inning_1_runs'].rolling(10, min_periods=1).mean()
            })
            team_stats.append(rolling_stats)
        
        team_stats_df = pd.concat(team_stats)
        df = df.merge(team_stats_df, on=['date', f'{team_type}_team_id'], how='left')
        
        # Track new NaNs introduced by this merge
        nan_counts_after = df.isna().sum()
        new_nans = nan_counts_after - nan_counts_before
        if (new_nans > 0).any():
            print(f"\nNew NaNs introduced in {team_type} team statistics merge:")
            print(new_nans[new_nans > 0])
        nan_counts_before = nan_counts_after.copy()
    
    track_nan_counts(df, "team statistics processing")
    
    print("\nProcessing pitcher statistics...")
    # Pitcher statistics
    for pitcher_type in ['home', 'away']:
        pitcher_stats = df.groupby(f'{pitcher_type}_pitcher_id').agg({
            'nrfi': 'mean',
            f'{pitcher_type}_inning_1_runs': 'mean'
        }).reset_index()
        pitcher_stats.columns = [
            f'{pitcher_type}_pitcher_id',
            f'{pitcher_type}_pitcher_nrfi_rate',
            f'{pitcher_type}_pitcher_runs_allowed'
        ]
        before_merge = df.isna().sum()
        df = df.merge(pitcher_stats, on=f'{pitcher_type}_pitcher_id', how='left')
        after_merge = df.isna().sum()
        new_nans = after_merge - before_merge
        if (new_nans > 0).any():
            print(f"\nNew NaNs introduced in {pitcher_type} pitcher merge:")
            print(new_nans[new_nans > 0])
    
    track_nan_counts(df, "pitcher statistics processing")
    
    print("\nCalculating matchup features...")
    df['pitcher_matchup_rating'] = (df['home_pitcher_nrfi_rate'] + df['away_pitcher_nrfi_rate']) / 2
    track_nan_counts(df, "matchup features calculation")
    
    print("\nHandling missing values...")
    # Count NaNs before filling
    print("\nNaN counts before filling:")
    print(df.isna().sum()[df.isna().sum() > 0])
    
    # Handle missing values for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Verify no NaNs remain
    remaining_nans = df[numeric_columns].isna().sum()
    if remaining_nans.any():
        print("\nWarning: NaNs still remain after filling:")
        print(remaining_nans[remaining_nans > 0])
    else:
        print("\nAll numeric NaN values have been filled")
    
    print("\nData preprocessing complete!")
    return df

def prepare_features(df):
    """Prepare feature matrix for modeling"""
    start_time = track_processing_step("Feature preparation")
    
    # Create temperature categories if they don't exist
    if 'temp_category' not in df.columns:
        print("Creating temperature categories...")
        df['temp_category'] = create_temp_categories(df)
    
    # Track NaN values before feature creation
    track_nan_counts(df, "before feature preparation")
    
    # Define basic feature columns
    basic_features = [
        'temperature', 'wind_speed', 'is_dome', 'is_wind_favorable',
        'venue_nrfi_rate', 'venue_home_scoring', 'venue_away_scoring',
        'home_team_scoring_trend', 'away_team_scoring_trend',
        'home_pitcher_nrfi_rate', 'away_pitcher_nrfi_rate',
        'home_pitcher_runs_allowed', 'away_pitcher_runs_allowed',
        'pitcher_matchup_rating'
    ]
    
    # Add enhanced features if they exist
    enhanced_features = [
        'home_pitcher_strength', 'away_pitcher_strength',
        'home_lineup_strength', 'away_lineup_strength',
        'home_pitcher_vs_lineup', 'away_pitcher_vs_lineup',
        'overall_matchup_score'
    ]
    
    # Check which enhanced features exist in the dataframe
    existing_enhanced = [col for col in enhanced_features if col in df.columns]
    
    # Combine all features
    feature_columns = basic_features + existing_enhanced
    
    # Track feature statistics
    track_feature_stats(df, feature_columns, "basic features")
    
    # Add dummy variables for temperature category
    print("\nCreating temperature category dummies...")
    temp_dummies = pd.get_dummies(df['temp_category'], prefix='temp')
    X = pd.concat([df[feature_columns], temp_dummies], axis=1)
    
    # Track final feature matrix statistics
    track_feature_stats(X, X.columns, "final feature matrix")
    
    # Track processing time
    track_processing_step("Feature preparation", start_time)
    
    return X

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix using seaborn"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH)
    plt.close('all')  # Explicitly close all figures

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH)
    plt.close('all')  # Explicitly close all figures
    return importances

def plot_calibration_curve(y_true, y_prob, filename):
    """Plot calibration curve to visualize model calibration quality"""
    plt.figure(figsize=(10, 6))
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    
    # Plot calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration curve")
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure to file with the provided filename
    plt.savefig(os.path.join(VISUALS_DIR, filename))
    plt.close()

def train_nrfi_ensemble_model(df, tune_hyperparameters=True):
    """Train an ensemble NRFI prediction model using a VotingClassifier"""
    start_time = track_processing_step("Model training")
    
    print("\nPreparing features...")
    X = prepare_features(df)
    y = df['nrfi']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Define base classifiers
    print("\nInitializing base classifiers...")
    clf_rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs_limit)
    clf_gb = GradientBoostingClassifier(random_state=42)
    clf_et = ExtraTreesClassifier(random_state=42, n_jobs=n_jobs_limit)
    
    # Create ensemble
    print("\nCreating ensemble...")
    ensemble = VotingClassifier(estimators=[
        ('rf', clf_rf),
        ('gb', clf_gb),
        ('et', clf_et)
    ], voting='soft', n_jobs=n_jobs_limit)
    
    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', ensemble)
    ])
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_start = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=n_jobs_limit)
    print(f"Cross-validation completed in {time.time() - cv_start:.2f} seconds")
    print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Optionally perform hyperparameter tuning on the ensemble
    if tune_hyperparameters:
        param_grid = {
            'ensemble__rf__n_estimators': [100, 200],
            'ensemble__rf__max_depth': [5, 10],
            'ensemble__gb__n_estimators': [100, 200],
            'ensemble__et__n_estimators': [100, 200],
        }
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=n_jobs_limit
        )
        grid_search.fit(X_train, y_train)
        print("\nBest parameters from grid search for ensemble:")
        print(grid_search.best_params_)
        best_model = grid_search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline
    
    print("\nCalibrating probabilities using isotonic regression...")
    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\nEnsemble Model Performance on Test Set:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    
    # Brier score
    brier = brier_score_loss(y_test, y_pred_proba)
    print(f"Brier score: {brier:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # For feature importance, we extract from one of the base models (e.g., the Random Forest)
    feature_importance = plot_feature_importance(best_model.named_steps['ensemble'].estimators_[0], X.columns)
    print("\nTop 5 Most Important Features from Random Forest in Ensemble:")
    print(feature_importance.head())
    
    # Save the entire pipeline
    os.makedirs('data/First Inning NRFI', exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)
    
    # Calibration plot
    plot_calibration_curve(y_test, calibrated_model.predict_proba(X_test)[:, 1], 'calibration_curve.png')
    
    track_processing_step("Model training", start_time)
    return best_model

def predict_upcoming_games(model, daily_starters_path=DAILY_STARTERS_PATH):
    """Make predictions for upcoming games using the trained pipeline"""
    start_time = track_processing_step("Prediction for upcoming games")
    
    if not os.path.exists(daily_starters_path):
        print(f"Daily starters file not found at {daily_starters_path}")
        return None
    
    # Load and preprocess upcoming games
    print("\nLoading upcoming games data...")
    upcoming_games = pd.read_csv(daily_starters_path)
    track_nan_counts(upcoming_games, "initial upcoming games load")
    
    # Load additional data sources
    print("\nLoading additional first inning data for predictions...")
    pitchers_df = load_pitchers_first_inning_data()
    batters_df = load_batters_first_inning_data()
    lineups_df = load_lineups_historical_data()
    
    # Handle dates consistently in YYYY-MM-DD format
    if 'game_time' in upcoming_games.columns:
        # First convert to datetime preserving timezone info (if it exists)
        upcoming_games['game_time'] = pd.to_datetime(upcoming_games['game_time'])
        # Then convert to date only in YYYY-MM-DD format
        upcoming_games['date'] = upcoming_games['game_time'].dt.date
    else:
        # If no game_time, just convert date to datetime and then to date
        upcoming_games['date'] = pd.to_datetime(upcoming_games['date']).dt.date
    
    # Convert date back to string format YYYY-MM-DD for consistency
    upcoming_games['date'] = upcoming_games['date'].astype(str)
    
    # Determine the cutoff date (latest date in upcoming games)
    cutoff_date = pd.to_datetime(upcoming_games['date']).max()
    
    # Load historical data up to the cutoff date
    historical_df = load_and_preprocess_data(cutoff_date=cutoff_date)
    
    # Print column comparison to debug
    print("\nColumns in upcoming_games but not in historical_df:")
    upcoming_only = [col for col in upcoming_games.columns if col not in historical_df.columns]
    print(upcoming_only)
    
    print("\nColumns in historical_df but not in upcoming_games:")
    historical_only = [col for col in historical_df.columns if col not in upcoming_games.columns]
    print(historical_only)
    
    # Convert pitcher IDs to integers
    upcoming_games['home_pitcher_id'] = upcoming_games['home_pitcher_id'].fillna(0).astype(int)
    upcoming_games['away_pitcher_id'] = upcoming_games['away_pitcher_id'].fillna(0).astype(int)
    
    # Ensure all numeric columns are actually numeric
    # Define expected numeric columns from historical data
    expected_numeric_cols = [
        'home_team_id', 'away_team_id', 
        'home_inning_1_runs', 'home_inning_1_hits', 'home_inning_1_errors', 'home_inning_1_leftOnBase',
        'away_inning_1_runs', 'away_inning_1_hits', 'away_inning_1_errors', 'away_inning_1_leftOnBase',
        'temperature', 'wind_speed'
    ]
    
    # Convert string columns to numeric where needed
    for col in expected_numeric_cols:
        if col in upcoming_games.columns:
            upcoming_games[col] = pd.to_numeric(upcoming_games[col], errors='coerce')
    
    # Create basic weather features
    upcoming_games['is_dome'] = upcoming_games['condition'].fillna('').astype(str).str.contains('Dome|Roof Closed').astype(int)
    upcoming_games['is_wind_favorable'] = upcoming_games['wind_direction'].fillna('').astype(str).str.contains('In|Out').astype(int)
    
    # Create venue features from historical data
    venue_stats = historical_df.groupby('venue_name').agg({
        'nrfi': 'mean',
        'home_inning_1_runs': 'mean',
        'away_inning_1_runs': 'mean'
    }).reset_index()
    venue_stats.columns = ['venue_name', 'venue_nrfi_rate', 'venue_home_scoring', 'venue_away_scoring']
    upcoming_games = upcoming_games.merge(venue_stats, on='venue_name', how='left')
    
    # Create team scoring trends from historical data
    for team_type in ['home', 'away']:
        team_stats = historical_df.groupby(f'{team_type}_team_id')[f'{team_type}_inning_1_runs'].mean().reset_index()
        team_stats.columns = [f'{team_type}_team_id', f'{team_type}_team_scoring_trend']
        upcoming_games = upcoming_games.merge(team_stats, on=f'{team_type}_team_id', how='left')
    
    # Convert pitcher IDs to consistently across all DataFrames
    for pitcher_type in ['home', 'away']:
        # Convert historical pitcher IDs to int64
        historical_df[f'{pitcher_type}_pitcher_id'] = pd.to_numeric(
            historical_df[f'{pitcher_type}_pitcher_id'], 
            errors='coerce'
        ).fillna(0).astype('int64')
        
        # Convert upcoming games pitcher IDs to int64
        upcoming_games[f'{pitcher_type}_pitcher_id'] = pd.to_numeric(
            upcoming_games[f'{pitcher_type}_pitcher_id'], 
            errors='coerce'
        ).fillna(0).astype('int64')
        
        # Calculate pitcher stats with consistent ID type
        pitcher_stats = historical_df.groupby(f'{pitcher_type}_pitcher_id').agg({
            'nrfi': 'mean',
            f'{pitcher_type}_inning_1_runs': 'mean'
        }).reset_index()
        
        # Ensure pitcher stats IDs are also int64
        pitcher_stats[f'{pitcher_type}_pitcher_id'] = pitcher_stats[f'{pitcher_type}_pitcher_id'].astype('int64')
        
        pitcher_stats.columns = [
            f'{pitcher_type}_pitcher_id',
            f'{pitcher_type}_pitcher_nrfi_rate',
            f'{pitcher_type}_pitcher_runs_allowed'
        ]
        
        # Merge with consistent types
        upcoming_games = upcoming_games.merge(
            pitcher_stats,
            on=f'{pitcher_type}_pitcher_id',
            how='left'
        )    # Create matchup features
    upcoming_games['pitcher_matchup_rating'] = (
        upcoming_games['home_pitcher_nrfi_rate'].fillna(0.5) + 
        upcoming_games['away_pitcher_nrfi_rate'].fillna(0.5)
    ) / 2
    
    # Calculate advanced statistics from additional data sources
    pitcher_stats = calculate_pitcher_first_inning_stats(pitchers_df) 
    batter_stats = calculate_batter_first_inning_stats(batters_df)
    
    # Calculate lineup strength if we have lineup and batter data
    if lineups_df is not None and batter_stats is not None:
        lineups_df = calculate_lineup_strength(lineups_df, batter_stats)
    
    # Add advanced matchup features if possible
    if pitcher_stats is not None and batter_stats is not None and lineups_df is not None:
        print("\nCreating pitcher-batter matchup features for predictions...")
        upcoming_games = create_pitcher_batter_matchup_features(
            upcoming_games, pitcher_stats, batter_stats, lineups_df)
    
    # Handle missing numeric values
    numeric_columns = upcoming_games.select_dtypes(include=[np.number]).columns
    
    # Fill missing numeric values
    for col in numeric_columns:
        if col in historical_df.columns and np.issubdtype(historical_df[col].dtype, np.number):
            # Use historical mean for columns that exist in historical data
            upcoming_games[col] = upcoming_games[col].fillna(historical_df[col].mean())
        else:
            # For columns that don't exist in historical_df or aren't numeric there,
            # use the mean of the column itself or a default value
            if upcoming_games[col].count() > 0:  # If we have some non-NA values
                upcoming_games[col] = upcoming_games[col].fillna(upcoming_games[col].mean())
            else:
                # If all values are NA, use a default
                if 'runs' in col.lower():
                    upcoming_games[col] = upcoming_games[col].fillna(0)
                elif 'rate' in col.lower() or 'prob' in col.lower():
                    upcoming_games[col] = upcoming_games[col].fillna(0.5)
                else:
                    upcoming_games[col] = upcoming_games[col].fillna(0)
    
    # Prepare features (ensure same columns as training)
    upcoming_features = prepare_features(upcoming_games)
    
    # Use the pipeline to scale features and make predictions
    predictions = model.predict_proba(upcoming_features)
    
    # Add predictions and confidence scores
    upcoming_games['nrfi_probability'] = predictions[:, 1]
    # Add runs probability (YRFI) which is the inverse of NRFI
    upcoming_games['runs_probability'] = 1 - upcoming_games['nrfi_probability']
    
    # Make prediction based on which probability is higher
    upcoming_games['prediction'] = (upcoming_games['nrfi_probability'] > 0.5).astype(int)
    
    # Calculate raw confidence (0-1 scale, higher means more confident)
    upcoming_games['confidence'] = abs(upcoming_games['nrfi_probability'] - 0.5) * 2
    
    # Create clearer prediction labels that show the prediction and confidence
    upcoming_games['prediction_label'] = upcoming_games.apply(
        lambda x: f"{'NRFI' if x['prediction'] == 1 else 'YRFI'} ({x['confidence']:.1%} confidence)", 
        axis=1
    )
    
    # Create clearer prediction descriptions
    upcoming_games['prediction_desc'] = upcoming_games.apply(
        lambda x: f"Prediction: {'No Runs' if x['prediction'] == 1 else 'Runs Will Score'} in 1st inning\n"
                 f"NRFI Probability: {x['nrfi_probability']:.1%}\n"
                 f"YRFI Probability: {x['runs_probability']:.1%}", 
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
        'venue_nrfi_rate', 'pitcher_matchup_rating'
    ]
    
    predictions_file = PREDICTIONS_OUTPUT_PATH
    upcoming_games[output_columns].to_csv(predictions_file, index=False)
    
    # Track prediction statistics
    print("\nPrediction Statistics:")
    print(f"Total games predicted: {len(predictions)}")
    print(f"Average NRFI probability: {upcoming_games['nrfi_probability'].mean():.3f}")
    print(f"Average YRFI probability: {upcoming_games['runs_probability'].mean():.3f}")
    print(f"Average confidence: {upcoming_games['confidence'].mean():.3f}")
    print("\nPrediction distribution:")
    print(upcoming_games['prediction'].value_counts(normalize=True))
    
    # Print a few example predictions with clearer interpretation
    print("\nExample predictions:")
    examples = upcoming_games[['home_team', 'away_team', 'nrfi_probability', 'runs_probability', 'prediction_label']].head(3)
    print(examples)
    
    track_processing_step("Prediction completion", start_time)
    return upcoming_games

def calculate_calibrated_confidence(df, nrfi_probabilities, X_test, test_data):
    """
    Calculate a more meaningful confidence score that's calibrated to historical outcomes
    """
    # 1. Probability calibration - add epsilon to ensure 1.0 falls in last bin
    bins = np.linspace(0, 1.0 + 1e-10, 11)  # Create 10 probability bins with slight padding
    bin_indices = np.clip(np.digitize(nrfi_probabilities, bins) - 1, 0, 9)  # Clip to valid range
    
    # 2. Create a historical accuracy map for each probability bin
    historical_accuracy = []
    for bin_idx in range(len(bins)-1):  # Only iterate through actual bin spaces
        bin_mask = (bin_indices == bin_idx)
        if np.sum(bin_mask) > 0:
            # Calculate how accurate predictions in this probability range have been
            bin_accuracy = np.mean((nrfi_probabilities[bin_mask] >= 0.5).astype(int) == 
                                   test_data['nrfi'].values[bin_mask])
            historical_accuracy.append(bin_accuracy)
        else:
            historical_accuracy.append(0.5)  # Default when no data
    
    # 3. Context-specific confidence
    context_confidence = []
    for i in range(len(test_data)):
        # Find similar historical games based on venue and matchups
        venue = test_data.iloc[i]['venue_name']
        home_pitcher = test_data.iloc[i]['home_pitcher_id']
        away_pitcher = test_data.iloc[i]['away_pitcher_id']
        
        # Look for games with same venue or same pitchers
        venue_matches = df[df['venue_name'] == venue]
        home_pitcher_matches = df[df['home_pitcher_id'] == home_pitcher]
        away_pitcher_matches = df[df['away_pitcher_id'] == away_pitcher]
        
        # Combine context matches with weighted importance
        context_matches = pd.concat([
            venue_matches,           # Venue is important
            home_pitcher_matches,    # Home pitcher performance matters
            away_pitcher_matches     # Away pitcher performance matters
        ]).drop_duplicates()
        
        if len(context_matches) >= 5:
            similar_accuracy = context_matches['nrfi'].mean()
            context_confidence.append(abs(similar_accuracy - 0.5) * 2)
        else:
            context_confidence.append(0.5)
    
    # 4. Consistency metrics using current batch predictions
    consistency_scores = []
    for i, prob in enumerate(nrfi_probabilities):
        prediction = 1 if prob >= 0.5 else 0
        
        # Find similar probability ranges in current predictions
        prob_range_mask = (nrfi_probabilities >= prob-0.05) & (nrfi_probabilities <= prob+0.05)
        similar_probs = test_data[prob_range_mask]
        
        if len(similar_probs) >= 10:
            correct_rate = np.mean(similar_probs['nrfi'] == prediction)
            consistency_scores.append(correct_rate)
        else:
            consistency_scores.append(0.5)
    
    # 5. Calculate final calibrated confidence score
    calibrated_confidence = np.zeros(len(nrfi_probabilities))
    
    for i in range(len(nrfi_probabilities)):
        # Raw statistical confidence (distance from 0.5)
        raw_confidence = abs(nrfi_probabilities[i] - 0.5) * 2
        
        # Get the bin accuracy for this prediction's probability range
        bin_accuracy = historical_accuracy[bin_indices[i]]
        
        # Blend the different confidence metrics
        calibrated_confidence[i] = (
            0.30 * raw_confidence +              # Statistical confidence
            0.30 * bin_accuracy +                # Historical accuracy for this probability range
            0.20 * context_confidence[i] +       # Context-specific performance
            0.20 * consistency_scores[i]         # Consistency of similar predictions
        )
    
    return calibrated_confidence

def backtest_model(df, time_splits=5, output_path='data/First Inning NRFI/F1_backtesting_results.csv'):
    """
    Perform backtesting on historical data to evaluate model performance over time.
    Uses time-based splits to simulate making predictions at different points in time.
    """
    start_time = track_processing_step("Backtesting model")
    
    # Sort data by date for proper time-based splitting
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create time-based splits
    dates = sorted(df['date'].unique())
    split_dates = [dates[i] for i in range(0, len(dates), len(dates) // time_splits)]
    
    # Ensure we have the last date
    if split_dates[-1] != dates[-1]:
        split_dates.append(dates[-1])
    
    print(f"\nSplit dates: {split_dates}")
    
    backtest_results = []
    
    # Track overall performance metrics
    all_true_values = []
    all_predictions = []
    all_probabilities = []
    
    # For each time period, train on data before that period and predict for that period
    for i in range(len(split_dates) - 1):
        split_start = split_dates[i]
        split_end = split_dates[i+1]
        
        print(f"\nBacktesting period {i+1}/{len(split_dates)-1}: {split_start} to {split_end}")
        
        # Split data into train (before current period) and test (current period)
        train_data = df[df['date'] < split_start].copy()
        test_data = df[(df['date'] >= split_start) & (df['date'] < split_end)].copy()
        
        if len(train_data) < 100:
            print("Warning: Not enough training data for this period, skipping")
            continue
        
        # Prepare features and targets
        X_train = prepare_features(train_data)
        y_train = train_data['nrfi']
        
        X_test = prepare_features(test_data)
        y_test = test_data['nrfi']
        
        # Train model and get NRFI probabilities
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100, n_jobs=n_jobs_limit))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Calculate predictions and probabilities
        nrfi_probabilities = pipeline.predict_proba(X_test)[:, 1]
        yrfi_probabilities = 1 - nrfi_probabilities

        # Make predictions (1 for NRFI, 0 for YRFI)
        y_pred = (nrfi_probabilities >= 0.5).astype(int)

        # Calculate confidence (how far from 0.5 in either direction)
        standard_confidence = np.abs(nrfi_probabilities - 0.5) * 2

        # Calculate calibrated confidence
        calibrated_confidence = calculate_calibrated_confidence(
            df, nrfi_probabilities, X_test, test_data)

        # Create more detailed results
        period_results = test_data.copy()
        period_results['nrfi_probability'] = nrfi_probabilities
        period_results['yrfi_probability'] = yrfi_probabilities
        period_results['predicted_outcome'] = np.where(y_pred == 1, 'NRFI', 'YRFI')
        period_results['actual_outcome'] = np.where(y_test == 1, 'NRFI', 'YRFI')
        period_results['standard_confidence'] = standard_confidence
        period_results['calibrated_confidence'] = calibrated_confidence
        period_results['confidence'] = calibrated_confidence  # Use calibrated as primary
        period_results['prediction_type'] = period_results['predicted_outcome']  # What we predicted
        period_results['correct_prediction'] = (y_pred == y_test)  # Simple equality check

        # Add prediction strength categories based on confidence
        period_results['prediction_strength'] = pd.cut(
            calibrated_confidence,
            bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
            labels=['Very Weak', 'Weak', 'Moderate', 'Strong', 'Very Strong']
        )
        
        # Collect metrics
        all_true_values.extend(y_test)
        all_predictions.extend(y_pred)
        all_probabilities.extend(nrfi_probabilities)
        
        # Calculate period metrics
        period_accuracy = np.mean(period_results['correct_prediction'])
        period_roc_auc = roc_auc_score(y_test, nrfi_probabilities) if len(set(y_test)) > 1 else 0
        
        print(f"Period metrics - Accuracy: {period_accuracy:.3f}, ROC-AUC: {period_roc_auc:.3f}")
        
        # Create results dataframe for this period
        period_results['backtest_period'] = i + 1
        period_results['backtest_start_date'] = split_start
        period_results['backtest_end_date'] = split_end
        period_results['period_accuracy'] = period_accuracy
        period_results['period_roc_auc'] = period_roc_auc
        period_results['predicted_outcome'] = np.where(nrfi_probabilities >= 0.5, 'NRFI', 'YRFI')
        period_results['actual_outcome'] = np.where(y_test == 1, 'NRFI', 'YRFI')
        period_results['prediction_correct'] = period_results['predicted_outcome'] == period_results['actual_outcome']
        
        backtest_results.append(period_results)
    
    # Combine all results
    if not backtest_results:
        print("No backtest results generated. Check your data and split settings.")
        return None
        
    all_results = pd.concat(backtest_results)
    
    # Calculate final metrics
    overall_accuracy = accuracy_score(all_true_values, all_predictions)
    overall_precision = precision_score(all_true_values, all_predictions, zero_division=0)
    overall_recall = recall_score(all_true_values, all_predictions, zero_division=0)
    overall_f1 = f1_score(all_true_values, all_predictions, zero_division=0)
    overall_roc_auc = roc_auc_score(all_true_values, all_probabilities)
    
    print("\nOverall Backtesting Metrics:")
    print(f"Accuracy: {overall_accuracy:.3f}")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall: {overall_recall:.3f}")
    print(f"F1 Score: {overall_f1:.3f}")
    print(f"ROC-AUC: {overall_roc_auc:.3f}")
    
    # Select relevant columns for output
    output_columns = [
        'date', 'home_team', 'away_team', 
        'home_pitcher_name', 'away_pitcher_name',
        'venue_name', 'temperature', 'condition',
        'backtest_period', 'nrfi', 'prediction',
        'nrfi_probability', 'runs_probability', 'confidence',
        'correct_prediction', 'period_accuracy', 'period_roc_auc'
    ]
    
    # Make sure we have all required columns in the output
    output_columns = [col for col in output_columns if col in all_results.columns]
    
    # Add overall metrics to the dataframe for reference
    all_results['overall_accuracy'] = overall_accuracy
    all_results['overall_precision'] = overall_precision
    all_results['overall_recall'] = overall_recall
    all_results['overall_f1'] = overall_f1
    all_results['overall_roc_auc'] = overall_roc_auc
    
    # Save results to CSV
    print(f"\nSaving backtesting results to {output_path}")
    all_results.to_csv(output_path, index=False)
    
    # Generate summary by confidence level
    print("\nPerformance by Confidence Level:")
    confidence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    all_results['confidence_bin'] = pd.cut(all_results['confidence'], bins=confidence_bins)
    confidence_summary = all_results.groupby('confidence_bin', observed=True).agg({
        'correct_prediction': ['mean', 'count'],
        'nrfi': 'mean'
    })
    print(confidence_summary)
    
    # Generate summary by prediction type
    print("\nPerformance by Prediction Type:")
    prediction_summary = all_results.groupby('predicted_outcome', observed=True).agg({
        'correct_prediction': ['mean', 'count'],
        'nrfi': 'mean'
    })
    print(prediction_summary)
    
    # Plot performance over time
    try:
        plt.figure(figsize=(10, 6))
        period_summary = all_results.groupby('backtest_period').agg({
            'correct_prediction': 'mean',
            'period_accuracy': 'first',
            'period_roc_auc': 'first'
        })
        
        plt.plot(period_summary.index, period_summary['correct_prediction'], marker='o', label='Prediction Accuracy')
        plt.plot(period_summary.index, period_summary['period_roc_auc'], marker='s', label='ROC-AUC')
        plt.xlabel('Backtest Period')
        plt.ylabel('Score')
        plt.title('Model Performance Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/First Inning NRFI/F1_backtest_performance.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting performance over time: {e}")
    
    track_processing_step("Backtesting completion", start_time)
    
    analyze_confidence_correlation(all_results)
    
    compare_confidence_metrics(all_results)
    
    return all_results

def analyze_confidence_correlation(all_results):
    """
    Analyze how well confidence scores correlate with prediction correctness
    """
    # Create confidence bins
    bins = np.linspace(0, 1, 11)
    all_results['confidence_bin'] = pd.cut(all_results['confidence'], bins)
    
    # Group by confidence bin and calculate accuracy
    confidence_accuracy = all_results.groupby('confidence_bin', observed=True)['correct_prediction'].mean()
    
    # Calculate correlation between confidence and correctness
    correlation = all_results['confidence'].corr(all_results['correct_prediction'])
    
    print(f"\nCorrelation between confidence and correctness: {correlation:.4f}")
    print("\nAccuracy by confidence level:")
    print(confidence_accuracy)
    
    # Plot the relationship
    plt.figure(figsize=(10, 6))
    sns.barplot(x=confidence_accuracy.index.astype(str), 
                y=confidence_accuracy.values)
    plt.title('Prediction Accuracy by Confidence Level')
    plt.xlabel('Confidence Level')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/First Inning NRFI/F1_confidence_calibration.png')
    plt.close()
    
    return confidence_accuracy

def expected_calibration_error(y_true, y_prob, n_bins=15, strategy='quantile'):
    """
    Compute Expected Calibration Error (ECE) for a set of predictions.
    """
    from sklearn.calibration import calibration_curve
    import numpy as np
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    bin_sizes = np.histogram(y_prob, bins=n_bins)[0]
    ece = np.sum(bin_sizes * np.abs(frac_pos - mean_pred)) / np.sum(bin_sizes)
    return ece

# Add this function to compare the two confidence metrics
def compare_confidence_metrics(all_results):
    """
    Compare standard confidence vs calibrated confidence with improved calibration visualization and ECE reporting
    """
    import numpy as np
    from sklearn.calibration import calibration_curve
    
    # Calculate correlations
    std_corr = all_results['standard_confidence'].corr(all_results['correct_prediction'])
    cal_corr = all_results['calibrated_confidence'].corr(all_results['correct_prediction'])
    print(f"\nStandard confidence correlation with correctness: {std_corr:.4f}")
    print(f"Calibrated confidence correlation with correctness: {cal_corr:.4f}")
    print(f"Improvement: {(cal_corr - std_corr) / std_corr:.2%}")

    # Improved: Use quantile binning and more bins
    n_bins = 15
    # Standard confidence
    all_results['std_conf_bin'] = pd.qcut(all_results['standard_confidence'], q=n_bins, duplicates='drop')
    std_acc = all_results.groupby('std_conf_bin', observed=True)['correct_prediction'].agg(['mean', 'count'])
    std_x = [interval.mid for interval in std_acc.index]
    # Calibrated confidence
    all_results['cal_conf_bin'] = pd.qcut(all_results['calibrated_confidence'], q=n_bins, duplicates='drop')
    cal_acc = all_results.groupby('cal_conf_bin', observed=True)['correct_prediction'].agg(['mean', 'count'])
    cal_x = [interval.mid for interval in cal_acc.index]

    # Error bars (Wilson score interval)
    from statsmodels.stats.proportion import proportion_confint
    std_err_low, std_err_upp = proportion_confint(std_acc['mean']*std_acc['count'], std_acc['count'], method='wilson')
    cal_err_low, cal_err_upp = proportion_confint(cal_acc['mean']*cal_acc['count'], cal_acc['count'], method='wilson')

    # Plot both for comparison, only using bins that have data
    plt.figure(figsize=(12, 7))
    plt.errorbar(std_x, std_acc['mean'], yerr=[std_acc['mean']-std_err_low, std_err_upp-std_acc['mean']], fmt='o-', label='Standard Confidence', capsize=3)
    plt.errorbar(cal_x, cal_acc['mean'], yerr=[cal_acc['mean']-cal_err_low, cal_err_upp-cal_acc['mean']], fmt='s-', label='Calibrated Confidence', capsize=3)
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration')
    plt.xlabel('Confidence Score (binned, quantiles)')
    plt.ylabel('Actual Accuracy')
    plt.title('Confidence Calibration Comparison (with Error Bars)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/First Inning NRFI/F1_confidence_comparison.png')
    plt.close()

    # ECE calculation for both metrics
    ece_std = expected_calibration_error(all_results['correct_prediction'], all_results['standard_confidence'], n_bins=n_bins, strategy='quantile')
    ece_cal = expected_calibration_error(all_results['correct_prediction'], all_results['calibrated_confidence'], n_bins=n_bins, strategy='quantile')
    print(f"Standard Confidence ECE: {ece_std:.4f}")
    print(f"Calibrated Confidence ECE: {ece_cal:.4f}")
    print(f"ECE Improvement: {ece_std - ece_cal:.4f}")

def load_pitchers_first_inning_data():
    """Load and preprocess the first inning pitcher statistics"""
    print("Loading first inning pitcher data...")
    
    # Check if file exists first
    if not os.path.exists(PITCHERS_FIRST_INNING_PATH):
        print(f"ERROR: First inning pitcher data file not found at {PITCHERS_FIRST_INNING_PATH}")
        default_tracker.track_missing_file(PITCHERS_FIRST_INNING_PATH, group='pitcher_data')
        return None
    
    # Load the data
    pitchers_df = pd.read_csv(PITCHERS_FIRST_INNING_PATH)
    
    # Select key performance metrics that are most relevant for NRFI prediction
    key_metrics = [
        'player_id', 'player_name', 'game_date', 'game_pk',
        'ba', 'slg', 'woba', 'xwoba', 'k_percent', 'bb_percent',
        'whiffs', 'swings', 'hardhit_percent', 'barrels_per_bbe_percent',
        'pitcher_run_value_per_100' # Negative is good for pitchers
    ]
    
    # Filter to only include key metrics if they exist
    existing_metrics = [col for col in key_metrics if col in pitchers_df.columns]
    missing_metrics = [col for col in key_metrics if col not in existing_metrics and col not in ['player_id', 'player_name', 'game_date', 'game_pk']]
    
    if missing_metrics:
        print(f"Warning: Missing pitcher metrics: {missing_metrics}")
        default_tracker.track_default('pitcher_missing_metrics', len(missing_metrics), 
                                      len(key_metrics) - 4, group='pitcher_data')
    
    pitchers_df = pitchers_df[existing_metrics]
    
    # Check for essential statistics and count missing values
    essential_stats = ['ba', 'k_percent', 'bb_percent', 'woba']
    available_stats = [col for col in essential_stats if col in pitchers_df.columns]
    
    if available_stats:
        missing_values = pitchers_df[available_stats].isna().sum()
        for stat, count in missing_values.items():
            if count > 0:
                print(f"Warning: {count}/{len(pitchers_df)} missing values for {stat} ({count/len(pitchers_df):.1%})")
                default_tracker.track_default(f'pitcher_{stat}', count, len(pitchers_df), group='pitcher_data')
    
    # Convert date to datetime for easier processing
    if 'game_date' in pitchers_df.columns:
        pitchers_df['game_date'] = pd.to_datetime(pitchers_df['game_date'])
    
    print(f"Loaded {len(pitchers_df)} first inning pitcher records")
    return pitchers_df

def calculate_pitcher_first_inning_stats(pitchers_df):
    """Calculate aggregate first inning statistics for pitchers"""
    if pitchers_df is None or len(pitchers_df) == 0:
        print("ERROR: No pitcher data available to calculate statistics")
        default_tracker.track_default('pitcher_data_missing', 1, 1, group='pitcher_stats')
        return pd.DataFrame()
    
    # Group by pitcher_id and calculate aggregate stats
    pitcher_stats = pitchers_df.groupby('player_id').agg({
        'player_name': 'first',  # Keep first occurrence of name
        'ba': 'mean',
        'slg': 'mean',
        'woba': 'mean',
        'xwoba': 'mean',
        'k_percent': 'mean',
        'bb_percent': 'mean',
        'hardhit_percent': 'mean',
        'barrels_per_bbe_percent': 'mean',
        'pitcher_run_value_per_100': 'mean',
        'game_pk': 'count'  # Count of games for sample size
    }).reset_index()
    
    # Rename count column to games_played
    pitcher_stats = pitcher_stats.rename(columns={'game_pk': 'games_played'})
    
    # Check for small sample sizes
    small_sample_pitchers = (pitcher_stats['games_played'] < 5).sum()
    if small_sample_pitchers > 0:
        pct_small_sample = small_sample_pitchers / len(pitcher_stats)
        print(f"Warning: {small_sample_pitchers} pitchers ({pct_small_sample:.1%}) have small sample sizes (<5 games)")
        default_tracker.track_default('pitcher_small_sample', small_sample_pitchers, len(pitcher_stats), group='pitcher_stats')
    
    # Track defaulted values for metrics used in the NRFI score
    scoring_metrics = {
        'ba': {'default': 0.260, 'weight': 0.2},
        'k_percent': {'default': 0.20, 'weight': 0.2},
        'bb_percent': {'default': 0.08, 'weight': 0.15},
        'woba': {'default': 0.320, 'weight': 0.2},
        'hardhit_percent': {'default': 0.35, 'weight': 0.1},
        'barrels_per_bbe_percent': {'default': 0.06, 'weight': 0.15}
    }
    
    # Track each metric used in the score calculation
    for metric, info in scoring_metrics.items():
        if metric in pitcher_stats.columns:
            missing_count = pitcher_stats[metric].isna().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(pitcher_stats)
                print(f"Warning: {missing_count} pitchers ({missing_pct:.1%}) missing {metric} data")
                default_tracker.track_default(f'pitcher_{metric}_default', missing_count, len(pitcher_stats), 
                                             group='pitcher_nrfi_score')
    
    # Calculate a pitcher NRFI score (higher is better for NRFI)
    # This formula weighs key metrics that contribute to preventing runs
    pitcher_stats['pitcher_nrfi_score'] = (
        # Lower batting average against is better
        (1 - pitcher_stats['ba'].fillna(0.260)) * 0.2 +  
        # Higher K% is better
        pitcher_stats['k_percent'].fillna(0.20) * 0.2 + 
        # Lower BB% is better
        (1 - pitcher_stats['bb_percent'].fillna(0.08)) * 0.15 + 
        # Lower wOBA is better
        (1 - pitcher_stats['woba'].fillna(0.320)) * 0.2 + 
        # Lower hard hit% is better
        (1 - pitcher_stats['hardhit_percent'].fillna(0.35)) * 0.1 + 
        # Lower barrels% is better
        (1 - pitcher_stats['barrels_per_bbe_percent'].fillna(0.06)) * 0.15
    )
    
    # Normalize score between 0 and 1
    if len(pitcher_stats) > 1:  # Only normalize if we have more than one pitcher
        min_score = pitcher_stats['pitcher_nrfi_score'].min()
        max_score = pitcher_stats['pitcher_nrfi_score'].max()
        if max_score > min_score:  # Avoid division by zero
            pitcher_stats['pitcher_nrfi_score'] = (pitcher_stats['pitcher_nrfi_score'] - min_score) / (max_score - min_score)
    
    return pitcher_stats

def load_batters_first_inning_data():
    """Load and preprocess the first inning batter statistics"""
    print("Loading first inning batter data...")
    
    # Check if file exists first
    if not os.path.exists(BATTERS_FIRST_INNING_PATH):
        print(f"ERROR: First inning batter data file not found at {BATTERS_FIRST_INNING_PATH}")
        default_tracker.track_missing_file(BATTERS_FIRST_INNING_PATH, group="batter_data")
        return None
    
    # Load the data
    batters_df = pd.read_csv(BATTERS_FIRST_INNING_PATH)
    
    # Select key performance metrics that are most relevant for NRFI prediction
    key_metrics = [
        'player_id', 'player_name', 'game_date', 'game_pk',
        'ba', 'slg', 'woba', 'xwoba', 'k_percent', 'bb_percent',
        'hardhit_percent', 'barrels_per_bbe_percent',
        'batter_run_value_per_100' # Positive is good for batters
    ]
    
    # Filter to only include key metrics if they exist
    existing_metrics = [col for col in key_metrics if col in batters_df.columns]
    missing_metrics = [col for col in key_metrics if col not in existing_metrics and col not in ['player_id', 'player_name', 'game_date', 'game_pk']]
    
    if missing_metrics:
        print(f"Warning: Missing batter metrics: {missing_metrics}")
        default_tracker.track_default('batter_missing_metrics', len(missing_metrics), 
                                     len(key_metrics) - 4, group='batter_data')
    
    batters_df = batters_df[existing_metrics]
    
    # Count missing values in essential statistics
    essential_stats = ['ba', 'slg', 'woba', 'k_percent']
    available_stats = [stat for stat in essential_stats if stat in batters_df.columns]
    
    if available_stats:
        for stat in available_stats:
            missing_count = batters_df[stat].isna().sum()
            if missing_count > 0:
                print(f"Warning: {missing_count}/{len(batters_df)} missing values for {stat} ({missing_count/len(batters_df):.1%})")
                default_tracker.track_default(f'batter_{stat}', missing_count, len(batters_df), group='batter_data')
    
    # Convert date to datetime for easier processing
    if 'game_date' in batters_df.columns:
        batters_df['game_date'] = pd.to_datetime(batters_df['game_date'])
    
    print(f"Loaded {len(batters_df)} first inning batter records")
    return batters_df

def calculate_batter_first_inning_stats(batters_df):
    """Calculate aggregate first inning statistics for batters"""
    if batters_df is None or len(batters_df) == 0:
        print("ERROR: No batter data available to calculate statistics")
        default_tracker.track_default('batter_data_missing', 1, 1, group='batter_stats')
        return pd.DataFrame()
    
    # Group by batter_id and calculate aggregate stats
    batter_stats = batters_df.groupby('player_id').agg({
        'player_name': 'first',  # Keep first occurrence of name
        'ba': 'mean',
        'slg': 'mean',
        'woba': 'mean',
        'xwoba': 'mean',
        'k_percent': 'mean',
        'bb_percent': 'mean',
        'hardhit_percent': 'mean',
        'barrels_per_bbe_percent': 'mean',
        'batter_run_value_per_100': 'mean',
        'game_pk': 'count'  # Count of games for sample size
    }).reset_index()
      # Rename count column to games_played
    batter_stats = batter_stats.rename(columns={'game_pk': 'games_played'})
    
    # Check for small sample sizes (fewer than 5 games)
    small_sample_batters = (batter_stats['games_played'] < 5).sum()
    if small_sample_batters > 0:
        pct_small_sample = small_sample_batters / len(batter_stats)
        print(f"Warning: {small_sample_batters} batters ({pct_small_sample:.1%}) have small sample sizes (<5 games)")
        default_tracker.track_default('batter_small_sample', small_sample_batters, len(batter_stats), group='batter_stats')
    
    # Define metrics used in scoring threat calculation with their default values
    scoring_metrics = {
        'ba': 0.240,
        'slg': 0.400,
        'woba': 0.320,
        'bb_percent': 0.08,
        'k_percent': 0.22,
        'hardhit_percent': 0.35,
        'barrels_per_bbe_percent': 0.06
    }
      # Track default usage for each metric in the score calculation
    for metric, default_val in scoring_metrics.items():
        if metric in batter_stats.columns:
            missing_count = batter_stats[metric].isna().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(batter_stats)
                print(f"Warning: {missing_count} batters ({missing_pct:.1%}) missing {metric} data")
                default_tracker.track_default(f'batter_{metric}_default', missing_count, len(batter_stats), 
                                           group='batter_scoring_threat')
    
    # Calculate a batter scoring threat score (higher means more likely to score)
    # This formula weighs key metrics that contribute to scoring runs
    batter_stats['batter_scoring_threat'] = (
        # Higher batting average is better
        batter_stats['ba'].fillna(scoring_metrics['ba']) * 0.2 +
        # Higher SLG is better
        batter_stats['slg'].fillna(scoring_metrics['slg']) * 0.2 +
        # Higher wOBA is better  
        batter_stats['woba'].fillna(scoring_metrics['woba']) * 0.2 +
        # Higher BB% is better for getting on base
        batter_stats['bb_percent'].fillna(scoring_metrics['bb_percent']) * 0.1 +
        # Lower K% is better (less likely to make an out)
        (1 - batter_stats['k_percent'].fillna(scoring_metrics['k_percent'])) * 0.1 +
        # Higher hard hit% is better
        batter_stats['hardhit_percent'].fillna(scoring_metrics['hardhit_percent']) * 0.1 +
        # Higher barrel% is better for extra base hits
        batter_stats['barrels_per_bbe_percent'].fillna(scoring_metrics['barrels_per_bbe_percent']) * 0.1
    )
    
    # Normalize score between 0 and 1
    if len(batter_stats) > 1:  # Only normalize if we have more than one batter
        min_score = batter_stats['batter_scoring_threat'].min()
        max_score = batter_stats['batter_scoring_threat'].max()
        if max_score > min_score:  # Avoid division by zero
            batter_stats['batter_scoring_threat'] = (batter_stats['batter_scoring_threat'] - min_score) / (max_score - min_score)
    
    return batter_stats

def load_lineups_historical_data():
    """Load and preprocess the historical lineup data"""
    print("Loading historical lineup data...")
    
    # Check if file exists first
    if not os.path.exists(LINEUPS_HISTORICAL_PATH):
        print(f"ERROR: Historical lineup data file not found at {LINEUPS_HISTORICAL_PATH}")
        default_tracker.track_missing_file(LINEUPS_HISTORICAL_PATH, group="lineup_data")
        return None
    
    # Load the data
    lineups_df = pd.read_csv(LINEUPS_HISTORICAL_PATH)
      # Check for required columns
    required_columns = ['Date', 'Home Team', 'Away Team', 'Home Lineup', 'Away Lineup']
    missing_columns = [col for col in required_columns if col not in lineups_df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns in lineup data: {missing_columns}")
        for col in missing_columns:
            default_tracker.track_default(f'lineup_missing_{col}', 1, 1, group='lineup_data')
    
    # Convert date to datetime for easier processing
    if 'Date' in lineups_df.columns:
        lineups_df['Date'] = pd.to_datetime(lineups_df['Date'])
      # Track and convert string representations of lists to actual lists
    if 'Home Lineup' in lineups_df.columns:
        invalid_entries = 0
        for i, lineup in enumerate(lineups_df['Home Lineup']):
            try:
                if isinstance(lineup, str):
                    lineups_df.at[i, 'Home Lineup'] = ast.literal_eval(lineup)
            except (ValueError, SyntaxError):
                invalid_entries += 1
                lineups_df.at[i, 'Home Lineup'] = []
        if invalid_entries > 0:
            print(f"Warning: {invalid_entries} invalid Home Lineup entries")
            default_tracker.track_default('invalid_home_lineups', invalid_entries, len(lineups_df), group='lineup_data')
    if 'Away Lineup' in lineups_df.columns:
        invalid_entries = 0
        for i, lineup in enumerate(lineups_df['Away Lineup']):
            try:
                if isinstance(lineup, str):
                    lineups_df.at[i, 'Away Lineup'] = ast.literal_eval(lineup)
            except (ValueError, SyntaxError):
                invalid_entries += 1
                lineups_df.at[i, 'Away Lineup'] = []
        if invalid_entries > 0:
            print(f"Warning: {invalid_entries} invalid Away Lineup entries")
            default_tracker.track_default('invalid_away_lineups', invalid_entries, len(lineups_df), group='lineup_data')
      # Count empty lineups
    if 'Home Lineup' in lineups_df.columns and 'Away Lineup' in lineups_df.columns:
        empty_home = sum(1 for lineup in lineups_df['Home Lineup'] if not lineup)
        empty_away = sum(1 for lineup in lineups_df['Away Lineup'] if not lineup)
        if empty_home > 0:
            print(f"Warning: {empty_home} empty Home Lineup entries ({empty_home/len(lineups_df):.1%})")
            default_tracker.track_default('empty_home_lineups', empty_home, len(lineups_df), group='lineup_data')
            
        if empty_away > 0:
            print(f"Warning: {empty_away} empty Away Lineup entries ({empty_away/len(lineups_df):.1%})")
            default_tracker.track_default('empty_away_lineups', empty_away, len(lineups_df), group='lineup_data')
    
    print(f"Loaded {len(lineups_df)} historical lineup records")
    return lineups_df

def load_first_inning_scores():
    """Load and preprocess the first inning scores data"""
    print("Loading first inning scores data...")
    
    # Check if file exists first
    if not os.path.exists(FIRST_INNING_SCORES_PATH):
        print(f"Warning: First inning scores file not found at {FIRST_INNING_SCORES_PATH}")
        return None
    
    # Load the data
    scores_df = pd.read_csv(FIRST_INNING_SCORES_PATH)
    
    # Ensure the gamePk column name is standardized
    if 'gamePk' in scores_df.columns:
        scores_df = scores_df.rename(columns={'gamePk': 'game_pk'})
    
    print(f"Loaded {len(scores_df)} first inning score records")
    return scores_df

def calculate_lineup_strength(lineups_df, batter_stats):
    """Calculate the offensive strength of each lineup using first inning batter stats"""
    if lineups_df is None or batter_stats is None or len(lineups_df) == 0:
        print("No lineup data or batter stats available to calculate lineup strength")
        return lineups_df
    
    # Make a copy to avoid modifying the original
    lineups = lineups_df.copy()
    
    # Create a dictionary for faster lookup of batter stats
    batter_dict = batter_stats.set_index('player_id').to_dict('index')
    
    # Function to calculate lineup strength (focus on first 4 batters who are most likely to bat in the 1st inning)
    def calculate_top_lineup_strength(lineup):
        if not isinstance(lineup, list) or not lineup:
            return 0.5  # Default if lineup is missing
        
        # Use only the first 4 batters (or fewer if lineup is shorter)
        top_batters = lineup[:min(4, len(lineup))]
        
        # Calculate average scoring threat for top batters
        threats = []
        for batter_id in top_batters:
            if batter_id in batter_dict:
                threats.append(batter_dict[batter_id].get('batter_scoring_threat', 0.5))
            else:
                threats.append(0.5)  # Default if batter not found
        
        # Return mean threat score, or default if no valid scores
        return np.mean(threats) if threats else 0.5
    
    # Calculate lineup strengths
    lineups['home_lineup_strength'] = lineups['Home Lineup'].apply(calculate_top_lineup_strength)
    lineups['away_lineup_strength'] = lineups['Away Lineup'].apply(calculate_top_lineup_strength)
    
    return lineups

def create_pitcher_batter_matchup_features(df, pitchers, batters, lineups):
    """Create features that represent the matchup between pitchers and opposing lineups"""
    if pitchers is None or batters is None or lineups is None:
        print("Missing data required for pitcher-batter matchup analysis")
        return df
    
    # Make a copy to avoid modifying the original
    enhanced_df = df.copy()
    
    # Create dictionaries for pitcher and lineup strengths
    pitcher_dict = {}
    if 'player_id' in pitchers.columns and 'pitcher_nrfi_score' in pitchers.columns:
        pitcher_dict = pitchers.set_index('player_id').to_dict('index')
    
    lineup_dict = {}
    
    # Create a lookup dictionary for lineups
    for _, row in lineups.iterrows():
        date_key = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
        home_team = row['Home Team']
        away_team = row['Away Team']
        
        # Store home lineup info
        home_key = f"{date_key}_{home_team}"
        lineup_dict[home_key] = {
            'lineup_strength': row.get('home_lineup_strength', 0.5),
            'lineup': row.get('Home Lineup', [])
        }
        
        # Store away lineup info
        away_key = f"{date_key}_{away_team}"
        lineup_dict[away_key] = {
            'lineup_strength': row.get('away_lineup_strength', 0.5),
            'lineup': row.get('Away Lineup', [])
        }
    
    # Calculate matchup scores for each game
    matchup_scores = []
    
    print("Creating pitcher-batter matchup features...")
    for i, row in tqdm(enhanced_df.iterrows(), total=len(enhanced_df)):
        # Get date and teams
        game_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Get home pitcher data
        home_pitcher_id = row.get('home_pitcher_id')
        home_pitcher_score = 0.5  # Default
        if home_pitcher_id and home_pitcher_id in pitcher_dict:
            home_pitcher_score = pitcher_dict[home_pitcher_id].get('pitcher_nrfi_score', 0.5)
        
        # Get away pitcher data
        away_pitcher_id = row.get('away_pitcher_id')
        away_pitcher_score = 0.5  # Default
        if away_pitcher_id and away_pitcher_id in pitcher_dict:
            away_pitcher_score = pitcher_dict[away_pitcher_id].get('pitcher_nrfi_score', 0.5)
        
        # Get lineup strengths
        home_key = f"{game_date}_{home_team}"
        away_key = f"{game_date}_{away_team}"
        
        home_lineup_strength = 0.5
        away_lineup_strength = 0.5
        
        if home_key in lineup_dict:
            home_lineup_strength = lineup_dict[home_key].get('lineup_strength', 0.5)
        
        if away_key in lineup_dict:
            away_lineup_strength = lineup_dict[away_key].get('lineup_strength', 0.5)
        
        # Calculate matchup scores
        home_pitcher_vs_lineup = (home_pitcher_score - away_lineup_strength + 1) / 2
        away_pitcher_vs_lineup = (away_pitcher_score - home_lineup_strength + 1) / 2
        
        # Overall matchup score (higher is better for NRFI)
        overall_matchup = (home_pitcher_vs_lineup + away_pitcher_vs_lineup) / 2
        
        matchup_scores.append({
            'home_pitcher_strength': home_pitcher_score,
            'away_pitcher_strength': away_pitcher_score,
            'home_lineup_strength': home_lineup_strength,
            'away_lineup_strength': away_lineup_strength,
            'home_pitcher_vs_lineup': home_pitcher_vs_lineup,
            'away_pitcher_vs_lineup': away_pitcher_vs_lineup,
            'overall_matchup_score': overall_matchup
        })
    
    # Convert to DataFrame and join with enhanced_df
    matchup_df = pd.DataFrame(matchup_scores)
    enhanced_df = pd.concat([enhanced_df.reset_index(drop=True), matchup_df], axis=1)
    
    return enhanced_df

def calculate_first_inning_stats(scores_df):
    """Calculate first inning statistics from the scores data"""
    if scores_df is None or len(scores_df) == 0:
        print("No first inning scores data available")
        return pd.DataFrame()
    
    # Create NRFI column (1 if no runs in first inning, 0 otherwise)
    scores_df['nrfi'] = ((scores_df['home_runs'] + scores_df['away_runs']) == 0).astype(int)
    
    # Calculate additional statistics
    scores_stats = scores_df.groupby('game_pk').agg({
        'nrfi': 'first',
        'home_runs': 'sum',
        'away_runs': 'sum',
        'home_hits': 'sum',
        'away_hits': 'sum',
        'home_lob': 'sum',
        'away_lob': 'sum',
        'home_errors': 'sum',
        'away_errors': 'sum'
    }).reset_index()
    
    # Calculate total runs in first inning
    scores_stats['total_runs'] = scores_stats['home_runs'] + scores_stats['away_runs']
    scores_stats['total_hits'] = scores_stats['home_hits'] + scores_stats['away_hits']
    scores_stats['total_lob'] = scores_stats['home_lob'] + scores_stats['away_lob']
    
    return scores_stats

def main():
    """Main function to train ensemble model, run backtests, and make predictions"""
    print("Loading and preprocessing historical data with enhanced first inning statistics...")
    df = load_and_preprocess_data()
    
    # Save the enhanced processed data
    print("\nSaving enhanced processed data...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Enhanced data saved to {PROCESSED_DATA_PATH}")
    
    print("\nSummary of enhanced features:")
    enhanced_features = [
        'home_pitcher_strength', 'away_pitcher_strength',
        'home_lineup_strength', 'away_lineup_strength',
        'home_pitcher_vs_lineup', 'away_pitcher_vs_lineup',
        'overall_matchup_score'
    ]
    existing_enhanced = [col for col in enhanced_features if col in df.columns]
    print(f"Available enhanced features: {existing_enhanced}")
    
    print("\nRunning backtesting to evaluate historical performance...")
    # Perform backtesting with 5 time periods (adjust as needed)
    backtest_results = backtest_model(df, time_splits=5, 
                                     output_path=os.path.join(NRFI_DIR, 'F1_backtesting_results.csv'))
    
    print("\nTraining Ensemble NRFI model on full dataset...")
    # Set tune_hyperparameters=True to perform grid search tuning on the ensemble
    model = train_nrfi_ensemble_model(df, tune_hyperparameters=True)
    
    print("\nMaking predictions for upcoming games...")
    predictions = predict_upcoming_games(model)
    
    if predictions is not None:
        print(f"\nPredictions saved to {PREDICTIONS_OUTPUT_PATH}")
        print("\nTop NRFI Opportunities:")
        top_predictions = predictions[['date', 'home_team', 'away_team', 
                                       'nrfi_probability', 'confidence']].head()
        print(top_predictions)
    
    print(f"\nBacktest results saved to {os.path.join(NRFI_DIR, 'F1_backtesting_results.csv')}")
    print(f"Backtest performance visualization saved to {os.path.join(VISUALS_DIR, 'F1_backtest_performance.png')}")
    
    # Print summary of model performance
    if 'home_pitcher_vs_lineup' in df.columns and 'away_pitcher_vs_lineup' in df.columns:
        print("\nModel leverages enhanced first inning statistics including:")
        print("- Pitcher first inning performance metrics")
        print("- Batter first inning performance metrics") 
        print("- Lineup strength analysis")
        print("- Pitcher-batter matchup analysis")

if __name__ == "__main__":
    main()
