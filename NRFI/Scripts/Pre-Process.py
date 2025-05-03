import pandas as pd
import os
import sys
from datetime import datetime
import logging
from enum import Enum, auto
import time

# Add parent directory to Python path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, TEMP_DIR, setup_logging, MLB_SEASON_DATES, 
                   NRFI_DIR, LOGS_DIR, PITCHERS_CSV, BATTERS_CSV)

class ProcessingStage(Enum):
    RAW_DATA = auto()
    CLEANED_IDS = auto()
    FILTERED_DATES = auto()
    MERGED_BASIC = auto()
    PITCHER_STATS = auto()
    FINAL_PROCESSED = auto()

# Setup logging using centralized configuration
logger = setup_logging('Pre-Process')

# Update file paths to use NRFI_DIR and LOGS_DIR
DEBUG_LOG_FILE = os.path.join(LOGS_DIR, "data_processing_debug.log")
PROCESSED_DATA_PATH = os.path.join(NRFI_DIR, "processed_mlb_data.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(NRFI_DIR, exist_ok=True)

# Setup additional file logging for data processing debug info
file_handler = logging.FileHandler(DEBUG_LOG_FILE, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

def log_dataframe_sample(df, stage_name, sample_size=10):
    """Log a sample of the dataframe with detailed column information"""
    header = f"""
{'='*80}
PROCESSING STAGE: {stage_name}
{'='*80}"""
    logging.info(header)
    
    # Log basic dataframe shape info
    logging.info(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Group columns by type
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    
    # Log column type distribution
    logging.info("\nColumn Types:")
    logging.info(f"- Numeric: {len(numeric_cols)}")
    logging.info(f"- Categorical: {len(categorical_cols)}")
    logging.info(f"- DateTime: {len(datetime_cols)}")
    
    # Get a consistent random sample
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    
    # Log missing values summary for the sample
    missing_summary = sample_df.isnull().sum()
    if missing_summary.any():
        logging.info("\nMissing Values in Sample:")
        for col, count in missing_summary[missing_summary > 0].items():
            logging.info(f"- {col}: {count}/{sample_size}")
    
    # Log sample data with better formatting
    logging.info("\nRandom Sample (Same seed across stages):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    logging.info("\n" + sample_df.to_string())
    
    # Log numeric summary if applicable
    if len(numeric_cols) > 0:
        logging.info("\nNumeric Columns Summary (Sample):")
        logging.info(sample_df[numeric_cols].describe().to_string())
    
    logging.info("\n" + "="*80 + "\n")

def analyze_missing_columns(df, threshold=0.02):
    """
    Analyze and return columns that have missing values above the threshold
    Args:
        df: pandas DataFrame
        threshold: float, maximum allowed percentage of missing values (default 0.02 or 2%)
    Returns:
        dict with statistics about missing values
    """
    total_rows = len(df)
    missing_stats = {}
    
    # Calculate missing values for each column
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / total_rows) * 100
    
    # Get columns exceeding threshold
    columns_to_drop = missing_percentages[missing_percentages > threshold * 100].index.tolist()
    completely_empty = missing_counts[missing_counts == total_rows].index.tolist()
    
    missing_stats = {
        'total_columns': len(df.columns),
        'columns_exceeding_threshold': columns_to_drop,
        'completely_empty_columns': completely_empty,
        'column_missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
    }
    
    return missing_stats

def clean_missing_columns(df, threshold=0.02):
    """
    Remove columns that have missing values above the threshold
    Args:
        df: pandas DataFrame
        threshold: float, maximum allowed percentage of missing values (default 0.02 or 2%)
    Returns:
        cleaned DataFrame
    """
    stats = analyze_missing_columns(df, threshold)
    columns_to_drop = stats['columns_exceeding_threshold']
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        logging.info(f"Dropped {len(columns_to_drop)} columns exceeding {threshold*100}% missing values threshold")
        logging.info("Dropped columns: " + ", ".join(columns_to_drop))
    
    return df, stats

def load_data(data_dir=NRFI_DIR):
    """
    Load data from CSV files in the specified data directory
    """
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found")

    # Define file paths
    historical_schedule_path = os.path.join(data_dir, "F1_historical_schedule.csv")
    first_inning_scores_path = os.path.join(data_dir, "F1_inning_scores.csv")
    pitchers_historical_path = PITCHERS_CSV
    batters_historical_path = BATTERS_CSV

    try:
        # Load data
        schedule_df = pd.read_csv(historical_schedule_path)
        scores_df = pd.read_csv(first_inning_scores_path)
        pitchers_df = pd.read_csv(pitchers_historical_path)
        batters_df = pd.read_csv(batters_historical_path)

        # Convert dates to datetime
        schedule_df['date'] = pd.to_datetime(schedule_df['date'])
        current_date = pd.Timestamp.now()
        
        # Filter out future games
        initial_games = len(schedule_df)
        schedule_df = schedule_df[schedule_df['date'] < current_date]
        future_games = initial_games - len(schedule_df)
        logging.info(f"\nRemoved {future_games} future/scheduled games")
        
        # Filter out postponed games
        postponed_count = len(schedule_df[schedule_df['status'] == 'Postponed'])
        schedule_df = schedule_df[schedule_df['status'] != 'Postponed']
        logging.info(f"Removed {postponed_count} postponed games")
        logging.info(f"Remaining historical games: {len(schedule_df)}")
        
        # Analyze and clean missing values
        logging.info("\nAnalyzing Pitchers Historical Data:")
        pitchers_df, pitchers_stats = clean_missing_columns(pitchers_df)
        logging.info(f"Original columns: {pitchers_stats['total_columns']}")
        logging.info("Missing value percentages by column:")
        for col, pct in pitchers_stats['column_missing_percentages'].items():
            logging.info(f"- {col}: {pct:.2f}%")
            
        logging.info("\nAnalyzing Batters Historical Data:")
        batters_df, batters_stats = clean_missing_columns(batters_df)
        logging.info(f"Original columns: {batters_stats['total_columns']}")
        logging.info("Missing value percentages by column:")
        for col, pct in batters_stats['column_missing_percentages'].items():
            logging.info(f"- {col}: {pct:.2f}%")
            
        # Log completely empty columns
        if pitchers_stats['completely_empty_columns']:
            logging.info("\nCompletely empty columns in Pitchers data:")
            logging.info(", ".join(pitchers_stats['completely_empty_columns']))
            
        if batters_stats['completely_empty_columns']:
            logging.info("\nCompletely empty columns in Batters data:")
            logging.info(", ".join(batters_stats['completely_empty_columns']))
        
        return schedule_df, scores_df, pitchers_df, batters_df
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading data files: {str(e)}")


def convert_to_int(value):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None

def log_operation_stats(df, operation_name, previous_count=None):
    """Log statistics about a data operation"""
    current_count = len(df)
    if previous_count is not None:
        removed = previous_count - current_count
        percent = (removed / previous_count * 100) if previous_count > 0 else 0
        print(f"{operation_name}:")
        print(f"  - Rows before: {previous_count:,}")
        print(f"  - Rows after: {current_count:,}")
        print(f"  - Rows removed: {removed:,} ({percent:.2f}%)")
    return current_count

def count_missing_values(df, cols=None):
    """Count missing values in specified columns or all columns"""
    missing = df[cols if cols else df.columns].isnull().sum()
    total = len(df)
    missing_stats = missing[missing > 0]
    if len(missing_stats) > 0:
        print("\nMissing values by column:")
        for col, count in missing_stats.items():
            percent = (count / total * 100)
            print(f"  - {col}: {count:,} ({percent:.2f}%)")
    return missing_stats

def analyze_missing_data_impact(df, key_identifier='game_pk'):
    """
    Analyze missing data patterns and their impact on game-level integrity
    Args:
        df: pandas DataFrame
        key_identifier: column that uniquely identifies each game (default: game_pk)
    Returns:
        dict with missing data analysis
    """
    total_games = df[key_identifier].nunique() if key_identifier in df.columns else len(df)
    total_columns = len(df.columns)
    
    # Calculate missing values by column
    missing_by_col = df.isnull().sum()
    missing_pct_by_col = (missing_by_col / len(df)) * 100
    
    # Identify columns with any missing values
    cols_with_missing = missing_by_col[missing_by_col > 0].index.tolist()
    
    # Group columns by missingness severity
    critical_cols = []
    warning_cols = []
    acceptable_cols = []
    
    for col in cols_with_missing:
        pct_missing = missing_pct_by_col[col]
        if pct_missing > 5:
            critical_cols.append((col, pct_missing))
        elif pct_missing > 2:
            warning_cols.append((col, pct_missing))
        else:
            acceptable_cols.append((col, pct_missing))
    
    # Analyze game-level completeness
    games_with_missing = df[df.isnull().any(axis=1)][key_identifier].nunique() if key_identifier in df.columns else df.isnull().any(axis=1).sum()
    pct_games_affected = (games_with_missing / total_games) * 100
    
    # Group columns by data type for context
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Calculate impact on key metrics if they exist
    key_stats_cols = ['home_runs', 'away_runs', 'NRFI']
    available_key_stats = [col for col in key_stats_cols if col in df.columns]
    key_stats_missing = df[available_key_stats].isnull().sum() if available_key_stats else pd.Series()
    
    # Analyze missing patterns in pitching statistics
    pitching_cols = [col for col in df.columns if any(stat in col.lower() for stat in ['pitch', 'strike', 'ball', 'whiff', 'swing'])]
    pitching_missing = {col: missing_pct_by_col[col] for col in pitching_cols if col in missing_pct_by_col.index}
    
    return {
        'total_games': total_games,
        'games_with_missing': games_with_missing,
        'pct_games_affected': pct_games_affected,
        'critical_columns': critical_cols,
        'warning_columns': warning_cols,
        'acceptable_columns': acceptable_cols,
        'key_stats_missing': key_stats_missing.to_dict(),
        'numeric_cols_count': len(numeric_cols),
        'categorical_cols_count': len(categorical_cols),
        'pitching_stats_missing': pitching_missing
    }

def log_missing_data_analysis(df, analysis_results):
    """Log the results of missing data analysis"""
    logging.info("\nMissing Data Impact Analysis:")
    logging.info(f"Total unique games/rows: {analysis_results['total_games']:,}")
    logging.info(f"Records affected by missing data: {analysis_results['games_with_missing']:,} ({analysis_results['pct_games_affected']:.2f}%)")
    
    if analysis_results['critical_columns']:
        logging.info("\nCritical Columns (>5% missing):")
        for col, pct in analysis_results['critical_columns']:
            logging.info(f"  - {col}: {pct:.2f}%")
    
    if analysis_results['warning_columns']:
        logging.info("\nWarning Columns (2-5% missing):")
        for col, pct in analysis_results['warning_columns']:
            logging.info(f"  - {col}: {pct:.2f}%")
    
    if analysis_results['key_stats_missing']:
        logging.info("\nKey Statistics Missing Values:")
        for stat, count in analysis_results['key_stats_missing'].items():
            logging.info(f"  - {stat}: {count:,}")
    
    if analysis_results['pitching_stats_missing']:
        logging.info("\nPitching Statistics Missing Values:")
        for stat, pct in analysis_results['pitching_stats_missing'].items():
            logging.info(f"  - {stat}: {pct:.2f}%")
    
    # Analyze correlation between missing values if we have critical or warning columns
    if analysis_results['critical_columns'] or analysis_results['warning_columns']:
        critical_warning_cols = [col for col, _ in analysis_results['critical_columns'] + analysis_results['warning_columns']]
        if critical_warning_cols:
            missing_correlation = df[critical_warning_cols].isnull().corr()
            logging.info("\nMissing Value Correlation Pattern:")
            logging.info(missing_correlation.to_string())

def process_data(schedule_df, scores_df, pitchers_df):
    """Process and clean the loaded data with detailed statistics"""
    print("\nStarting data processing...")
    
    # Initial data analysis
    initial_missing_analysis = analyze_missing_data_impact(pitchers_df)
    log_missing_data_analysis(pitchers_df, initial_missing_analysis)
    
    # Log initial state
    log_dataframe_sample(schedule_df, ProcessingStage.RAW_DATA.name)
    
    # Setup output paths
    output_file = PROCESSED_DATA_PATH
    temp_file = os.path.join(TEMP_DIR, f"processed_mlb_data_{int(time.time())}.temp")

    # Initial counts
    initial_schedule = len(schedule_df)
    initial_scores = len(scores_df)
    initial_pitchers = len(pitchers_df)
    print(f"\nInitial data counts:")
    print(f"  - Schedule data: {initial_schedule:,} rows")
    print(f"  - Scores data: {initial_scores:,} rows")
    print(f"  - Pitchers data: {initial_pitchers:,} rows")

    # Convert IDs to integers
    print("\nConverting IDs to integers...")
    schedule_df['home_pitcher_id'] = schedule_df['home_pitcher_id'].apply(convert_to_int)
    schedule_df['away_pitcher_id'] = schedule_df['away_pitcher_id'].apply(convert_to_int)
    pitchers_df['player_id'] = pitchers_df['player_id'].apply(convert_to_int)
    
    log_dataframe_sample(schedule_df, ProcessingStage.CLEANED_IDS.name)

    # Count missing IDs after conversion
    print("\nAnalyzing missing pitcher IDs:")
    count_missing_values(schedule_df, ['home_pitcher_id', 'away_pitcher_id'])

    # Get postponed games count
    postponed_games = schedule_df[schedule_df['status'] == 'Postponed']
    print(f"\nPostponed games: {len(postponed_games):,} ({(len(postponed_games)/len(schedule_df))*100:.2f}%)")

    # Filter out dates outside of MLB seasons
    current_count = len(schedule_df)
    schedule_df['date'] = pd.to_datetime(schedule_df['date'])
    
    # Define season date ranges
    season_ranges = {
        2018: ('2018-03-29', '2018-10-28'),
        2019: ('2019-03-20', '2019-10-30'),
        2020: ('2020-07-23', '2020-10-27'),
        2021: ('2021-04-01', '2021-11-02'),
        2022: ('2022-04-07', '2022-11-05'),
        2023: ('2023-03-30', '2023-11-04'),
        2024: ('2024-03-28', '2024-11-02'),
        2025: ('2025-03-27', '2025-11-02')
    }

    # Filter data by season dates
    valid_dates = pd.Series(False, index=schedule_df.index)
    for year, (start, end) in season_ranges.items():
        year_mask = (schedule_df['date'] >= pd.to_datetime(start)) & (schedule_df['date'] <= pd.to_datetime(end))
        valid_dates = valid_dates | year_mask

    schedule_df = schedule_df[valid_dates]
    log_operation_stats(schedule_df, "Date range filtering", current_count)
    log_dataframe_sample(schedule_df, ProcessingStage.FILTERED_DATES.name)

    # Print games by season
    print("\nGames by season:")
    season_counts = schedule_df.groupby(schedule_df['date'].dt.year).size()
    for year, count in season_counts.items():
        print(f"  - {year}: {count:,} games")

    # Merge schedule and scores data
    print("\nMerging schedule and scores data...")
    current_count = len(schedule_df)
    merged_df = pd.merge(
        schedule_df,
        scores_df,
        left_on='game_pk',
        right_on='gamePk',
        how='left'
    )
    log_operation_stats(merged_df, "Schedule-Scores merge", current_count)
    log_dataframe_sample(merged_df, ProcessingStage.MERGED_BASIC.name)

    # Only keep games that have been played (have scores)
    current_count = len(merged_df)
    merged_df = merged_df.dropna(subset=['home_runs', 'away_runs'])
    log_operation_stats(merged_df, "Removing unplayed games", current_count)
    log_dataframe_sample(merged_df, "After Removing Unplayed Games")

    # Merge pitcher stats
    print("\nMerging pitcher statistics...")
    current_count = len(merged_df)
    home_pitcher_stats = pitchers_df.rename(columns={'player_id': 'home_pitcher_id', 'player_name': 'home_pitcher_name'})
    merged_df = pd.merge(
        merged_df,
        home_pitcher_stats,
        how='left',
        on=['game_pk', 'home_pitcher_id'],
        suffixes=('', '_home_pitcher')
    )
    log_operation_stats(merged_df, "Home pitcher stats merge", current_count)
    log_dataframe_sample(merged_df, ProcessingStage.PITCHER_STATS.name)

    # Merge away pitcher stats
    current_count = len(merged_df)
    away_pitcher_stats = pitchers_df.rename(columns={'player_id': 'away_pitcher_id', 'player_name': 'away_pitcher_name'})
    merged_df = pd.merge(
        merged_df,
        away_pitcher_stats,
        how='left',
        on=['game_pk', 'away_pitcher_id'],
        suffixes=('', '_away_pitcher')
    )
    log_operation_stats(merged_df, "Away pitcher stats merge", current_count)
    log_dataframe_sample(merged_df, "After Away Pitcher Merge")

    # Calculate NRFI (No Runs First Inning)
    merged_df['NRFI'] = ((merged_df['home_runs'] == 0) & (merged_df['away_runs'] == 0)).astype(int)
    nrfi_count = merged_df['NRFI'].sum()
    nrfi_percent = (nrfi_count / len(merged_df) * 100)
    print(f"\nNRFI Statistics:")
    print(f"  - Total NRFI games: {nrfi_count:,} ({nrfi_percent:.2f}%)")

    # Sort by date for chronological order
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df.sort_values(['date', 'game_pk'])
    
    # Identify numeric columns
    numeric_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Fill any remaining missing values with zeros
    merged_df[numeric_columns] = merged_df[numeric_columns].fillna(0)
    
    # After filling missing values
    log_dataframe_sample(merged_df, ProcessingStage.FINAL_PROCESSED.name)

    # Round all numeric columns to 3 decimal places
    merged_df[numeric_columns] = merged_df[numeric_columns].round(3)

    # Save to temp file first
    try:
        merged_df.to_csv(temp_file, index=False)
        # Move temp file to final location
        os.replace(temp_file, output_file)
        logger.info(f"Successfully saved processed data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise
    
    return merged_df

def main():
    try:
        # Create required directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Load data using DATA_DIR paths
        schedule_df, scores_df, pitchers_df, batters_df = load_data()
        
        # Process data
        processed_df = process_data(schedule_df, scores_df, pitchers_df)
        logger.info(f"Processed {len(processed_df):,} rows of data successfully.")
        
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()