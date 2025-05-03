'''
NRFI Lineup Integration Module

This module enhances the NRFI prediction model by incorporating batting lineup data.
It utilizes historical batter performance in first innings and predicted lineups to
create features that capture the strength of batting orders and matchup advantages.

Key Features:
- Creates first-inning performance metrics for batters
- Calculates lineup strength focused on top of the order
- Generates batter-pitcher matchup statistics
- Handles missing lineups by using historical patterns
- Integrates seamlessly with the existing NRFI model

Dependencies:
- pandas: For data manipulation
- numpy: For numerical operations
- datetime: For date handling

Usage:
    lineup_integration = NRFILineupIntegration(
        batters_file, pitchers_file, lineups_file, historical_file
    )
    
    features = lineup_integration.generate_features(
        game_pk=game_pk,
        game_date=game_date,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_pitcher_id=home_pitcher_id,
        away_pitcher_id=away_pitcher_id
    )
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import setup_logging

# Setup logging
logger = setup_logging('NRFI_Lineup')

class NRFILineupIntegration:
    def __init__(self, batters_file, pitchers_file, lineups_file, historical_schedule_file):
        """Initialize with required data files"""
        logger.info("Initializing NRFI Lineup Integration...")
        
        try:
            self.batters_df = pd.read_csv(batters_file)
            logger.info(f"Loaded batters data: {len(self.batters_df)} records")
        except Exception as e:
            logger.error(f"Failed to load batters file: {e}")
            self.batters_df = pd.DataFrame()
            
        try:
            self.pitchers_df = pd.read_csv(pitchers_file)
            logger.info(f"Loaded pitchers data: {len(self.pitchers_df)} records")
        except Exception as e:
            logger.error(f"Failed to load pitchers file: {e}")
            self.pitchers_df = pd.DataFrame()
            
        try:
            self.lineups_df = pd.read_csv(lineups_file)
            logger.info(f"Loaded lineups data: {len(self.lineups_df)} records")
        except Exception as e:
            logger.error(f"Failed to load lineups file: {e}")
            self.lineups_df = pd.DataFrame()
            
        try:
            self.historical_df = pd.read_csv(historical_schedule_file)
            logger.info(f"Loaded historical schedule: {len(self.historical_df)} records")
        except Exception as e:
            logger.error(f"Failed to load historical schedule: {e}")
            self.historical_df = pd.DataFrame()
        
        # Preprocess data
        self._preprocess_data()
        
        # Add a set to track teams we've already logged as missing lineups
        self.logged_missing_teams = set()
        
    def _preprocess_data(self):
        """Clean and prepare the data for analysis"""
        logger.info("Preprocessing data...")
        
        # Convert dates to datetime
        for df_name, df in [
            ("batters_df", self.batters_df), 
            ("pitchers_df", self.pitchers_df), 
            ("lineups_df", self.lineups_df), 
            ("historical_df", self.historical_df)
        ]:
            if not df.empty:
                date_col = 'game_date' if 'game_date' in df.columns else 'date' if 'date' in df.columns else None
                if date_col:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        logger.debug(f"Converted {date_col} to datetime in {df_name}")
                    except Exception as e:
                        logger.warning(f"Failed to convert dates in {df_name}: {e}")
                
        # Create key metrics for batters and pitchers
        if not self.batters_df.empty:
            self._create_batter_metrics()
        else:
            self.batter_stats = pd.DataFrame()
            
        if not self.pitchers_df.empty:
            self._create_pitcher_metrics()
        else:
            self.pitcher_stats = pd.DataFrame()
        
        logger.info("Data preprocessing complete")
        
    def _create_batter_metrics(self):
        """Create first inning performance metrics for each batter"""
        logger.info("Creating batter metrics...")
        
        # Handle missing columns gracefully
        required_columns = ['player_id', 'pa', 'hits', 'abs', 'hrs', 'doubles', 'triples', 'so', 'bb']
        missing_columns = [col for col in required_columns if col not in self.batters_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns in batters data: {missing_columns}")
            # Add missing columns with default values
            for col in missing_columns:
                self.batters_df[col] = 0
        
        # Handle optional columns
        for col in ['slg', 'woba', 'barrel_rate', 'whiff_rate']:
            if col not in self.batters_df.columns:
                self.batters_df[col] = 0.0
        
        try:
            # Group by player and calculate first inning stats
            batter_stats = self.batters_df.groupby('player_id').agg({
                'pa': 'sum',
                'hits': 'sum',
                'abs': 'sum',
                'hrs': 'sum',
                'doubles': 'sum',
                'triples': 'sum',
                'so': 'sum',
                'bb': 'sum',
                'slg': 'mean',
                'woba': 'mean',
                'whiff_rate': 'mean',
                'barrel_rate': 'mean'
            }).reset_index()
            
            # Calculate batting average, OBP, etc. safely
            batter_stats['ba'] = batter_stats.apply(
                lambda x: x['hits'] / x['abs'] if x['abs'] > 0 else 0.250, axis=1
            )
            batter_stats['obp'] = batter_stats.apply(
                lambda x: (x['hits'] + x['bb']) / x['pa'] if x['pa'] > 0 else 0.320, axis=1
            )
            
            # Calculate first inning power metrics
            batter_stats['iso'] = batter_stats.apply(
                lambda x: (x['doubles'] + 2*x['triples'] + 3*x['hrs']) / x['abs'] if x['abs'] > 0 else 0.150,
                axis=1
            )
            
            # Fill missing values with league averages
            batter_stats.fillna({
                'ba': 0.250,
                'obp': 0.320,
                'slg': 0.400,
                'woba': 0.320,
                'barrel_rate': 0.060,
                'whiff_rate': 0.240,
                'iso': 0.150
            }, inplace=True)
            
            self.batter_stats = batter_stats
            logger.info(f"Created stats for {len(batter_stats)} batters")
        except Exception as e:
            logger.error(f"Error creating batter metrics: {e}")
            # Create empty DataFrame with required columns
            self.batter_stats = pd.DataFrame(columns=[
                'player_id', 'pa', 'hits', 'abs', 'hrs', 'doubles', 'triples', 'so', 'bb',
                'ba', 'obp', 'slg', 'woba', 'barrel_rate', 'whiff_rate', 'iso'
            ])
    
    def _create_pitcher_metrics(self):
        """Create first inning performance metrics for each pitcher"""
        logger.info("Creating pitcher metrics...")
        
        # Handle missing columns gracefully
        required_columns = ['player_id', 'pa', 'hits', 'abs', 'hrs', 'so', 'bb']
        missing_columns = [col for col in required_columns if col not in self.pitchers_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns in pitchers data: {missing_columns}")
            # Add missing columns with default values
            for col in missing_columns:
                self.pitchers_df[col] = 0
                
        # Handle optional columns
        for col in ['whiff_rate', 'k_percent', 'bb_percent', 'first_inning_whiff_rate', 'first_inning_contact_rate']:
            if col not in self.pitchers_df.columns:
                self.pitchers_df[col] = 0.0
        
        try:
            pitcher_stats = self.pitchers_df.groupby('player_id').agg({
                'pa': 'sum',
                'hits': 'sum',
                'abs': 'sum',
                'hrs': 'sum',
                'so': 'sum',
                'bb': 'sum',
                'whiff_rate': 'mean',
                'k_percent': 'mean',
                'bb_percent': 'mean',
                'first_inning_whiff_rate': 'mean',
                'first_inning_contact_rate': 'mean'
            }).reset_index()
            
            # Safely calculate derived metrics
            pitcher_stats['hits_per_pa'] = pitcher_stats.apply(
                lambda x: x['hits'] / x['pa'] if x['pa'] > 0 else 0.230, 
                axis=1
            )
            
            pitcher_stats['first_inning_nrfi_rate'] = pitcher_stats.apply(
                lambda x: max(0.5, 1 - (x['hits_per_pa'] * 2)),
                axis=1
            )
            
            # Fill missing values with reasonable defaults
            pitcher_stats.fillna({
                'whiff_rate': 0.240,
                'k_percent': 0.220,
                'bb_percent': 0.080,
                'first_inning_whiff_rate': 0.240,
                'first_inning_contact_rate': 0.760,
                'first_inning_nrfi_rate': 0.500
            }, inplace=True)
            
            self.pitcher_stats = pitcher_stats
            logger.info(f"Created stats for {len(pitcher_stats)} pitchers")
        except Exception as e:
            logger.error(f"Error creating pitcher metrics: {e}")
            # Create empty DataFrame with required columns
            self.pitcher_stats = pd.DataFrame(columns=[
                'player_id', 'pa', 'hits', 'abs', 'hrs', 'so', 'bb',
                'whiff_rate', 'k_percent', 'bb_percent', 
                'first_inning_whiff_rate', 'first_inning_contact_rate',
                'hits_per_pa', 'first_inning_nrfi_rate'
            ])
        
    def get_recent_lineup(self, team, date, n_games=10):
        """Get recent lineup for a team"""
        logger.debug(f"Getting recent lineup for {team} before {date}")
        
        if self.lineups_df.empty:
            logger.warning(f"No lineup data available")
            return None
        
        try:
            # Convert date to datetime if it's not already
            cutoff_date = pd.to_datetime(date) if not isinstance(date, pd.Timestamp) else date
            
            # Find team identifier field (could be team, team_id, or name)
            team_field = 'team' if 'team' in self.lineups_df.columns else 'team_id' if 'team_id' in self.lineups_df.columns else None
            if not team_field:
                logger.warning("Could not find team identifier field in lineups data")
                return None
                
            # Date field could be game_date or date
            date_field = 'game_date' if 'game_date' in self.lineups_df.columns else 'date' if 'date' in self.lineups_df.columns else None
            if not date_field:
                logger.warning("Could not find date field in lineups data")
                return None
            
            # Filter lineups for the team before the cutoff date
            team_lineups = self.lineups_df[
                (self.lineups_df[team_field] == team) & 
                (self.lineups_df[date_field] < cutoff_date)
            ].sort_values(date_field, ascending=False).head(n_games)
            
            if team_lineups.empty:
                logger.info(f"No recent lineups found for team {team}")
                return None
                
            # Analyze the most common batting order
            common_lineup = {}
            for pos in range(1, 10):  # 9 batting positions
                pos_players = team_lineups[team_lineups['batting_order'] == pos]
                if not pos_players.empty:
                    # Get most common player in this position
                    common_player = pos_players['player_id'].value_counts().idxmax()
                    common_lineup[pos] = common_player
            
            logger.debug(f"Found lineup with {len(common_lineup)} positions for team {team}")
            return common_lineup
        except Exception as e:
            logger.error(f"Error getting recent lineup: {e}")
            return None
        
    def calculate_lineup_strength(self, lineup, focus_positions=4):
        """Calculate strength of a lineup based on first X batters"""
        logger.debug(f"Calculating lineup strength for {focus_positions} top positions")
        
        if not lineup:
            logger.debug("No lineup provided, using default values")
            return {
                'lineup_woba': 0.320,
                'lineup_slg': 0.400,
                'lineup_obp': 0.333,
                'lineup_barrel_rate': 0.060,
                'lineup_iso': 0.150
            }
            
        # Focus on top of the order
        top_lineup = {k: v for k, v in lineup.items() if k <= focus_positions}
        
        metrics = {
            'lineup_woba': 0,
            'lineup_slg': 0,
            'lineup_obp': 0,
            'lineup_barrel_rate': 0,
            'lineup_iso': 0
        }
        
        # Sum metrics for available players
        valid_players = 0
        
        if self.batter_stats.empty:
            logger.warning("No batter stats available for lineup strength calculation")
            return {
                'lineup_woba': 0.320,
                'lineup_slg': 0.400,
                'lineup_obp': 0.333,
                'lineup_barrel_rate': 0.060,
                'lineup_iso': 0.150
            }
        
        for pos, player_id in top_lineup.items():
            player_stats = self.batter_stats[self.batter_stats['player_id'] == player_id]
            if not player_stats.empty:
                metrics['lineup_woba'] += player_stats['woba'].values[0]
                metrics['lineup_slg'] += player_stats['slg'].values[0]
                metrics['lineup_obp'] += player_stats['obp'].values[0]
                metrics['lineup_barrel_rate'] += player_stats['barrel_rate'].values[0]
                metrics['lineup_iso'] += player_stats['iso'].values[0]
                valid_players += 1
                
        # Calculate averages
        if valid_players > 0:
            for key in metrics:
                metrics[key] /= valid_players
        else:
            # Use league average values if no valid players
            metrics = {
                'lineup_woba': 0.320,
                'lineup_slg': 0.400,
                'lineup_obp': 0.333,
                'lineup_barrel_rate': 0.060,
                'lineup_iso': 0.150
            }
                
        logger.debug(f"Calculated lineup strength: wOBA={metrics['lineup_woba']:.3f}, SLG={metrics['lineup_slg']:.3f}")
        return metrics
        
    def calculate_batter_pitcher_matchups(self, lineup, pitcher_id):
        """Calculate matchup statistics between specific lineup and pitcher"""
        logger.debug(f"Calculating matchups for pitcher {pitcher_id}")
        
        if not lineup or not pitcher_id or self.pitcher_stats.empty:
            logger.debug("Missing data for matchup calculation, using default values")
            return {
                'top_order_advantage': 0.5  # Neutral when unknown
            }
            
        # Get pitcher stats
        pitcher_stats = self.pitcher_stats[self.pitcher_stats['player_id'] == pitcher_id]
        if pitcher_stats.empty:
            logger.debug(f"No stats found for pitcher {pitcher_id}")
            return {'top_order_advantage': 0.5}
            
        pitcher_k_rate = pitcher_stats['k_percent'].values[0]
        pitcher_whiff_rate = pitcher_stats['first_inning_whiff_rate'].values[0] if 'first_inning_whiff_rate' in pitcher_stats.columns else 0.240
        
        # Look at top 4 batters
        top_lineup = {k: v for k, v in lineup.items() if k <= 4}
        
        # Calculate advantage metrics
        batter_whiff_rates = []
        batter_obp_values = []
        
        if self.batter_stats.empty:
            logger.warning("No batter stats available for matchup calculation")
            return {'top_order_advantage': 0.5}
            
        for pos, player_id in top_lineup.items():
            batter_stats = self.batter_stats[self.batter_stats['player_id'] == player_id]
            if not batter_stats.empty:
                batter_whiff_rates.append(batter_stats['whiff_rate'].values[0])
                batter_obp_values.append(batter_stats['obp'].values[0])
                
        if not batter_whiff_rates:
            logger.debug("No batter whiff rates found, using default advantage")
            return {'top_order_advantage': 0.5}
            
        # Top order has advantage when their whiff rate is lower than pitcher's
        avg_batter_whiff = sum(batter_whiff_rates) / len(batter_whiff_rates)
        avg_batter_obp = sum(batter_obp_values) / len(batter_obp_values) if batter_obp_values else 0.333
        
        # Calculate advantage: higher values favor pitcher, lower values favor batters
        top_order_advantage = pitcher_whiff_rate / avg_batter_whiff if avg_batter_whiff > 0 else 0.5
        
        # Adjust by OBP vs K-rate
        obp_k_factor = (avg_batter_obp / 0.333) / (pitcher_k_rate / 0.220)
        top_order_advantage = (top_order_advantage * 0.7) + (0.5 / obp_k_factor * 0.3)
        
        result = {
            'top_order_advantage': min(max(0.2, top_order_advantage), 0.8)  # Bound between 0.2-0.8
        }
        
        logger.debug(f"Calculated matchup advantage: {result['top_order_advantage']:.3f}")
        return result
        
    def generate_features(self, game_pk, game_date, home_team_id, away_team_id, 
                          home_pitcher_id, away_pitcher_id, 
                          known_home_lineup=None, known_away_lineup=None):
        """Generate lineup-related features for NRFI model"""
        logger.info(f"Generating lineup features for game {game_pk}: {home_team_id} vs {away_team_id}")
        
        # Get lineups (known or predicted)
        home_lineup = known_home_lineup or self.get_recent_lineup(home_team_id, game_date)
        away_lineup = known_away_lineup or self.get_recent_lineup(away_team_id, game_date)
        
        # Calculate lineup strength
        home_strength = self.calculate_lineup_strength(home_lineup)
        away_strength = self.calculate_lineup_strength(away_lineup)
        
        # Calculate matchups
        home_batter_vs_away_pitcher = self.calculate_batter_pitcher_matchups(home_lineup, away_pitcher_id)
        away_batter_vs_home_pitcher = self.calculate_batter_pitcher_matchups(away_lineup, home_pitcher_id)
        
        # Combine all features
        features = {
            'home_lineup_woba': home_strength['lineup_woba'],
            'home_lineup_slg': home_strength['lineup_slg'],
            'home_lineup_obp': home_strength['lineup_obp'],
            'home_lineup_barrel_rate': home_strength['lineup_barrel_rate'],
            'home_lineup_iso': home_strength['lineup_iso'],
            'away_lineup_woba': away_strength['lineup_woba'],
            'away_lineup_slg': away_strength['lineup_slg'],
            'away_lineup_obp': away_strength['lineup_obp'],
            'away_lineup_barrel_rate': away_strength['lineup_barrel_rate'],
            'away_lineup_iso': away_strength['lineup_iso'],
            'home_batter_advantage': 1 - away_batter_vs_home_pitcher['top_order_advantage'],
            'away_batter_advantage': 1 - home_batter_vs_away_pitcher['top_order_advantage'],
            'home_has_known_lineup': 1 if known_home_lineup else 0,
            'away_has_known_lineup': 1 if known_away_lineup else 0,
        }
        
        logger.info(f"Generated {len(features)} lineup features for game {game_pk}")
        return features

    def save_processed_stats(self, output_dir=None):
        """Save processed batter and pitcher stats for faster loading"""
        if output_dir is None:
            output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        try:
            batter_output = os.path.join(output_dir, "processed_batter_stats.csv")
            pitcher_output = os.path.join(output_dir, "processed_pitcher_stats.csv")
            
            self.batter_stats.to_csv(batter_output, index=False)
            self.pitcher_stats.to_csv(pitcher_output, index=False)
            
            logger.info(f"Saved processed stats to {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Error saving processed stats: {e}")
            return False
            
    def load_processed_stats(self, stats_dir=None):
        """Load preprocessed stats instead of calculating from scratch"""
        if stats_dir is None:
            stats_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        try:
            batter_file = os.path.join(stats_dir, "processed_batter_stats.csv")
            pitcher_file = os.path.join(stats_dir, "processed_pitcher_stats.csv")
            
            if os.path.exists(batter_file) and os.path.exists(pitcher_file):
                self.batter_stats = pd.read_csv(batter_file)
                self.pitcher_stats = pd.read_csv(pitcher_file)
                logger.info(f"Loaded preprocessed stats from {stats_dir}")
                return True
        except Exception as e:
            logger.error(f"Error loading preprocessed stats: {e}")
            
        logger.info("Could not load preprocessed stats, will recalculate")
        return False

    def add_lineup_features(self, upcoming_df, for_prediction=False):
        """Add lineup features to upcoming games dataframe"""
        logger.info(f"Adding lineup features to {len(upcoming_df)} games")
        
        # Reset the logged teams set at the beginning of a new prediction batch
        self.logged_missing_teams = set()
        
        # Prepare columns for features
        feature_columns = [
            'home_lineup_woba', 'home_lineup_slg', 'home_lineup_obp', 
            'home_lineup_barrel_rate', 'home_lineup_iso',
            'away_lineup_woba', 'away_lineup_slg', 'away_lineup_obp', 
            'away_lineup_barrel_rate', 'away_lineup_iso',
            'home_batter_advantage', 'away_batter_advantage',
            'home_has_known_lineup', 'away_has_known_lineup'
        ]
        
        # Add default columns if missing
        for col in feature_columns:
            if col not in upcoming_df.columns:
                upcoming_df[col] = 0.0
        
        # Process each game
        for idx, game in upcoming_df.iterrows():
            game_pk = game['game_pk']
            game_date = game['game_date']
            home_team_id = game['home_team_id']
            away_team_id = game['away_team_id']
            home_pitcher_id = game['home_pitcher_id'] if 'home_pitcher_id' in game else None
            away_pitcher_id = game['away_pitcher_id'] if 'away_pitcher_id' in game else None
            
            # Skip if game ID is not valid
            if pd.isna(game_pk) or pd.isna(game_date) or pd.isna(home_team_id) or pd.isna(away_team_id):
                logger.warning(f"Skipping invalid game data at index {idx}")
                continue
            
            # Generate features
            features = self.generate_features(
                game_pk=game_pk,
                game_date=game_date,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_pitcher_id=home_pitcher_id,
                away_pitcher_id=away_pitcher_id
            )
            
            # Update the upcoming_df with new features
            for col, value in features.items():
                upcoming_df.at[idx, col] = value
        
        logger.info("Lineup features added to upcoming games dataframe")
        return upcoming_df
