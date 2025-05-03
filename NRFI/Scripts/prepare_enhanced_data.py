"""
Prepare Enhanced NRFI Data Files
--------------------------------
This script prepares the enhanced data files for the NRFI model.
It processes the first inning batter and pitcher stats and lineup data.
"""

import os
import sys
import time
import traceback
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, NRFI_DIR, setup_logging

# Setup logging
logger = setup_logging('NRFI_Data_Prep')

# Define paths - Use the subdirectory structure as seen in NRFI.py
BATTERS_SOURCE_PATH = os.path.join(DATA_DIR, "First Inning NRFI", "1stInningBattersHistorical.csv") 
PITCHERS_SOURCE_PATH = os.path.join(DATA_DIR, "First Inning NRFI", "1stInningPitchersHistorical.csv")
LINEUPS_SOURCE_PATH = os.path.join(DATA_DIR, "First Inning NRFI", "probable_lineups_historical.csv")

# Target paths in the NRFI directory
BATTERS_TARGET_PATH = os.path.join(NRFI_DIR, "1stInningBattersHistorical.csv")
PITCHERS_TARGET_PATH = os.path.join(NRFI_DIR, "1stInningPitchersHistorical.csv")

def ensure_directories():
    """Ensure all necessary directories exist"""
    print(f"Ensuring directory exists: {NRFI_DIR}")
    os.makedirs(NRFI_DIR, exist_ok=True)

def process_batter_stats():
    """Process first inning batter stats"""
    print("\nProcessing first inning batter stats...")
    try:
        # Check if source file exists
        if not os.path.exists(BATTERS_SOURCE_PATH):
            print(f"Source file not found: {BATTERS_SOURCE_PATH}")
            return False
            
        df = pd.read_csv(BATTERS_SOURCE_PATH)
        print(f"Loaded {len(df)} batter records")
        
        # Convert date to datetime
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Create derived metrics
        if all(col in df.columns for col in ['hits', 'abs']):
            df['hit_rate'] = df['hits'] / df['abs'].replace(0, 1)  # Avoid division by zero
        
        if all(col in df.columns for col in ['barrels_total', 'bip']):
            df['barrel_rate'] = df['barrels_total'] / df['bip'].replace(0, 1)
        
        if all(col in df.columns for col in ['swings', 'takes']):
            df['swing_rate'] = df['swings'] / (df['swings'] + df['takes']).replace(0, 1)
            
        if all(col in df.columns for col in ['whiffs', 'swings']):
            df['whiff_rate'] = df['whiffs'] / df['swings'].replace(0, 1)
        
        # Save processed file
        df.to_csv(BATTERS_TARGET_PATH, index=False)
        print(f"Saved processed batter stats to {BATTERS_TARGET_PATH}")
        return True
    
    except Exception as e:
        print(f"Error processing batter stats: {e}")
        return False

def process_pitcher_stats():
    """Process first inning pitcher stats"""
    print("\nProcessing first inning pitcher stats...")
    try:
        # Check if source file exists
        if not os.path.exists(PITCHERS_SOURCE_PATH):
            print(f"Source file not found: {PITCHERS_SOURCE_PATH}")
            return False
        
        # Load data
        df = pd.read_csv(PITCHERS_SOURCE_PATH)
        print(f"Loaded {len(df)} pitcher records")
        
        # Convert date to datetime
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
          # Create derived metrics
        # Ensure key columns exist for model
        missing_columns = []
        defaulted_columns = {}
        calculated_columns = {}
        mapped_columns = {}
        
        # Check for key columns
        for col in ['hardhit_percent', 'k_percent', 'spin_rate', 'velocity', 'whiffs']:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"WARNING: The following key columns are missing and will be addressed: {', '.join(missing_columns)}")
            print("This may affect model accuracy! Consider adding these columns to your source data.")
        
        # Create missing columns with default values or calculate them from available data
        # hardhit_percent handling
        if 'hardhit_percent' not in df.columns:
            if 'hard_hit_percent' in df.columns:
                print("Mapping 'hard_hit_percent' to 'hardhit_percent'")
                df['hardhit_percent'] = df['hard_hit_percent']
                mapped_columns['hardhit_percent'] = 'hard_hit_percent'
            elif 'hard_hit_against_percent' in df.columns:
                print("Mapping 'hard_hit_against_percent' to 'hardhit_percent'")
                df['hardhit_percent'] = df['hard_hit_against_percent']
                mapped_columns['hardhit_percent'] = 'hard_hit_against_percent'
            else:
                print("WARNING: Creating default hardhit_percent column (30.0% - league average)")
                df['hardhit_percent'] = 30.0  # Average hardhit_percent
                defaulted_columns['hardhit_percent'] = 30.0
        
        # k_percent handling
        if 'k_percent' not in df.columns:
            if 'so' in df.columns and 'abs' in df.columns and df['abs'].sum() > 0:
                print("Calculating k_percent from strikeouts and at-bats")
                df['k_percent'] = df['so'] / df['abs'].replace(0, 1)
                calculated_columns['k_percent'] = 'so / abs'
            elif 'strikeOuts' in df.columns and 'batters_faced' in df.columns and df['batters_faced'].sum() > 0:
                print("Calculating k_percent from strikeouts and batters faced")
                df['k_percent'] = df['strikeOuts'] / df['batters_faced'].replace(0, 1)
                calculated_columns['k_percent'] = 'strikeOuts / batters_faced'
            else:
                print("WARNING: Creating default k_percent column (0.22 - league average K%)")
                df['k_percent'] = 0.22  # League average K%
                defaulted_columns['k_percent'] = 0.22
        
        # spin_rate handling
        if 'spin_rate' not in df.columns:
            print("WARNING: Creating default spin_rate column (2200 RPM - league average)")
            df['spin_rate'] = 2200.0  # League average spin rate
            defaulted_columns['spin_rate'] = 2200.0
        
        # velocity handling
        if 'velocity' not in df.columns:
            if 'effective_speed' in df.columns:
                print("Mapping 'effective_speed' to 'velocity'")
                df['velocity'] = df['effective_speed']
                mapped_columns['velocity'] = 'effective_speed'
            else:
                print("WARNING: Creating default velocity column (92.0 mph - league average)")
                df['velocity'] = 92.0  # League average velocity
                defaulted_columns['velocity'] = 92.0
        
        # whiffs handling
        if 'whiffs' not in df.columns:
            if 'swings' in df.columns and 'whiff_rate' in df.columns:
                print("Calculating whiffs from swings and whiff_rate")
                df['whiffs'] = df['swings'] * df['whiff_rate']
                calculated_columns['whiffs'] = 'swings * whiff_rate'
            else:
                print("WARNING: Creating default whiffs column (5.0 - estimated average)")
                df['whiffs'] = 5.0  # Default whiff count
                defaulted_columns['whiffs'] = 5.0
          # Save processed file but first provide a summary of what was done
        print("\n===== PITCHER STATS PROCESSING SUMMARY =====")
        print(f"Total records processed: {len(df)}")
        
        # Report on mappings
        if mapped_columns:
            print("\nColumns mapped from alternate sources:")
            for target, source in mapped_columns.items():
                print(f"  - {target} ← {source}")
        
        # Report on calculated values
        if calculated_columns:
            print("\nColumns calculated from available data:")
            for target, formula in calculated_columns.items():
                print(f"  - {target} = {formula}")
        
        # Report on defaulted values
        if defaulted_columns:
            print("\nWARNING: Columns created with default values (may affect model accuracy):")
            for col, default_val in defaulted_columns.items():
                print(f"  - {col} = {default_val}")
                # Calculate percentage of records using the default
                pct = 100.0
                print(f"    Default applied to 100% of records ({len(df)} rows)")
        
        # Display a sample of the processed data
        print("\nSample of processed pitcher data:")
        sample_cols = ['player_id', 'player_name', 'game_date', 'velocity', 'spin_rate', 
                       'k_percent', 'hardhit_percent', 'whiffs']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(3).to_string(index=False))
        
        df.to_csv(PITCHERS_TARGET_PATH, index=False)
        print(f"\nSaved processed pitcher stats to {PITCHERS_TARGET_PATH}")
        
        return True
    
    except Exception as e:
        print(f"Error processing pitcher stats: {e}")
        return False

def ensure_lineup_data():
    """Ensure lineup data is available in the NRFI directory"""
    print("\nChecking lineup data...")
    try:
        # Check if source file exists
        if not os.path.exists(LINEUPS_SOURCE_PATH):
            print(f"Lineup file not found: {LINEUPS_SOURCE_PATH}")
            
            # Look for alternative lineup files in the data directory
            alternative_paths = [
                os.path.join(DATA_DIR, "mlb_game_logs_lineups.csv"),
                os.path.join(DATA_DIR, "mlb", "lineups.csv"),
                os.path.join(DATA_DIR, "lineups.csv")
            ]
            
            found_alternative = False
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Found alternative lineup file: {alt_path}")
                    found_alternative = True
                    
                    # Copy this file to the NRFI directory
                    nrfi_lineups_path = os.path.join(NRFI_DIR, "probable_lineups_historical.csv")
                    print(f"Copying alternative lineup data to {nrfi_lineups_path}")
                    
                    # Read and save the file to create a copy
                    df = pd.read_csv(alt_path)
                    df.to_csv(nrfi_lineups_path, index=False)
                    print(f"Alternative lineup data copied successfully")
                    return True
            
            if not found_alternative:
                # Create a minimal empty lineup file as a placeholder
                print("No alternative lineup files found. Creating a minimal placeholder file.")
                nrfi_lineups_path = os.path.join(NRFI_DIR, "probable_lineups_historical.csv")
                
                # Create minimal structure with required columns
                minimal_df = pd.DataFrame({
                    'game_pk': [],
                    'game_date': [],
                    'team': [],
                    'home_away': [],
                    'batting_order': [],
                    'player_id': [],
                    'player_name': []
                })
                
                # Save the placeholder file
                minimal_df.to_csv(nrfi_lineups_path, index=False)
                print(f"Created empty placeholder lineup file at {nrfi_lineups_path}")
                print("WARNING: Model functionality that depends on lineup data will be limited.")
                return True
                
        else:
            # Source file exists, proceed normally
            nrfi_lineups_path = os.path.join(NRFI_DIR, "probable_lineups_historical.csv")
            if not os.path.exists(nrfi_lineups_path):
                # Link or copy the file to NRFI directory
                print(f"Copying lineup data to {nrfi_lineups_path}")
                
                # Read and save the file to create a copy
                df = pd.read_csv(LINEUPS_SOURCE_PATH)
                df.to_csv(nrfi_lineups_path, index=False)
                
                print(f"Lineup data copied successfully")
            else:
                print(f"Lineup data already exists at {nrfi_lineups_path}")
            
            return True
    
    except Exception as e:
        print(f"Error ensuring lineup data: {e}")
        traceback.print_exc()
        
        # In case of any error, create the minimal placeholder file
        try:
            print("Creating minimal placeholder lineup file due to error.")
            nrfi_lineups_path = os.path.join(NRFI_DIR, "probable_lineups_historical.csv")
            
            # Create minimal structure with required columns
            minimal_df = pd.DataFrame({
                'game_pk': [],
                'game_date': [],
                'team': [],
                'home_away': [],
                'batting_order': [],
                'player_id': [],
                'player_name': []
            })
            
            # Save the placeholder file
            minimal_df.to_csv(nrfi_lineups_path, index=False)
            print(f"Created empty placeholder lineup file at {nrfi_lineups_path}")
            print("WARNING: Model functionality that depends on lineup data will be limited.")
            return True
            
        except Exception as inner_e:
            print(f"Failed to create placeholder file: {inner_e}")
            return False

def main():
    print("Starting enhanced data preparation for NRFI model...")
    print("=" * 50)
    
    # Create directories
    print("\n[1/4] Creating necessary directories...")
    ensure_directories()
    
    # Process batter stats
    print("\n[2/4] Processing batter statistics...")
    batter_start_time = time.time()
    batter_success = process_batter_stats()
    batter_time = time.time() - batter_start_time
    
    # Process pitcher stats
    print("\n[3/4] Processing pitcher statistics...")
    pitcher_start_time = time.time()
    pitcher_success = process_pitcher_stats()
    pitcher_time = time.time() - pitcher_start_time
    
    # Process lineup data
    print("\n[4/4] Processing lineup data...")
    lineup_start_time = time.time()
    lineup_success = ensure_lineup_data()
    lineup_time = time.time() - lineup_start_time
    
    # Calculate overall time
    total_time = batter_time + pitcher_time + lineup_time
      # Report results
    print("\n" + "=" * 50)
    print("DATA PREPARATION SUMMARY")
    print("=" * 50)
    print(f"Batter stats:  {'✓ Success' if batter_success else '✗ Failed'} ({batter_time:.1f} seconds)")
    print(f"Pitcher stats: {'✓ Success' if pitcher_success else '✗ Failed'} ({pitcher_time:.1f} seconds)")
    print(f"Lineup data:   {'✓ Success' if lineup_success else '⚠️ Warning: Using placeholder'} ({lineup_time:.1f} seconds)")
    print("-" * 50)
    print(f"Total processing time: {total_time:.1f} seconds")
    
    # Check if we have the minimum required data (batters and pitchers)
    min_requirements_met = batter_success and pitcher_success
    
    # Overall status - lineup data is helpful but not critical for the model
    if min_requirements_met:
        if not lineup_success:
            print("\n⚠️ Basic data preparation completed with warnings:")
            print("   - Lineup data is missing or using a placeholder")
            print("   - This may limit certain model features but will not prevent model operation")
        else:
            print("\n✅ All data preparation steps completed successfully!")
        
        # Provide a final data quality report
        print("\nDATA QUALITY ASSESSMENT:")
        print("-" * 50)
        
        # Read in the processed files to check their quality
        try:
            batters_df = pd.read_csv(BATTERS_TARGET_PATH)
            pitchers_df = pd.read_csv(PITCHERS_TARGET_PATH)
            
            # Check for null percentages in key columns
            batter_nulls = batters_df[['player_id', 'game_date', 'whiff_rate', 'hit_rate']].isnull().mean() * 100
            pitcher_nulls = pitchers_df[['player_id', 'game_date', 'velocity', 'spin_rate', 'k_percent', 'hardhit_percent', 'whiffs']].isnull().mean() * 100
            
            print("Missing data percentages:")
            print("BATTERS:")
            for col, pct in batter_nulls.items():
                status = "✓ Good" if pct < 5 else "⚠️ Warning" if pct < 20 else "❌ Critical"
                print(f"  {col}: {pct:.1f}% missing - {status}")
                
            print("\nPITCHERS:")
            for col, pct in pitcher_nulls.items():
                status = "✓ Good" if pct < 5 else "⚠️ Warning" if pct < 20 else "❌ Critical"
                print(f"  {col}: {pct:.1f}% missing - {status}")
                
            # Check for defaulted values
            defaulted_cols = ['velocity', 'spin_rate', 'k_percent', 'hardhit_percent', 'whiffs']
            default_values = {
                'velocity': 92.0,
                'spin_rate': 2200.0,
                'k_percent': 0.22,
                'hardhit_percent': 30.0,
                'whiffs': 5.0
            }
            
            print("\nChecking for defaulted values:")
            for col, default_val in default_values.items():
                if col in pitchers_df.columns:
                    default_pct = (pitchers_df[col] == default_val).mean() * 100
                    status = "✓ Good" if default_pct < 5 else "⚠️ Warning" if default_pct < 20 else "❌ Critical"
                    print(f"  {col}: {default_pct:.1f}% using default value {default_val} - {status}")
            
        except Exception as e:
            print(f"\nError during data quality assessment: {e}")
    else:
        print("\n❌ Some data preparation steps failed. Please check the errors above.")
        print("   Model accuracy may be compromised due to missing or defaulted data.")

if __name__ == "__main__":
    main()
