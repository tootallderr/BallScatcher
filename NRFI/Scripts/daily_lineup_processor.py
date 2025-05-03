'''
Daily Lineup Processor for NRFI Predictions

This script fetches the latest MLB lineups for today's games and prepares them for use
in NRFI prediction. It connects to the MLB Stats API to retrieve confirmed lineups
whenever possible, falling back to predicted lineups based on recent patterns.

Usage:
    python daily_lineup_processor.py [--force]
    
Arguments:
    --force: Force fresh lineup data even if today's data exists
'''

import os
import sys
import aiohttp
import asyncio
import pandas as pd
import argparse
from datetime import datetime, timedelta

# Add script directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NRFI_DIR, DATA_DIR, setup_logging, USER_AGENT

# Setup logging
logger = setup_logging('NRFI_Lineups')

# Configure event loop policy for Windows
if sys.platform.startswith('win'):
    import asyncio
    if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def fetch_with_retry(session, url, retries=3):
    """Fetch data from URL with retry logic"""
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to fetch {url}: {str(e)}")
                return {}
            await asyncio.sleep(1 + attempt)


async def fetch_schedule(session, date=None):
    """Fetch MLB schedule for a given date"""
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
        
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher,team"
    logger.debug(f"Fetching schedule for date: {date}")
    return await fetch_with_retry(session, url)


async def fetch_lineups(session, game_pk):
    """Fetch lineups for a specific game"""
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    logger.debug(f"Fetching lineups for game: {game_pk}")
    return await fetch_with_retry(session, url)


def extract_lineup(boxscore_data, team_side, game_pk, game_date, team_name, team_id):
    """Extract lineup data from boxscore"""
    lineup = []
    team_data = boxscore_data.get('teams', {}).get(team_side, {})
    players = team_data.get('players', {})
    
    # Check if lineup is available
    if not players:
        logger.debug(f"No {team_side} lineup available for game {game_pk}")
        return []
    
    # Try to extract batting order
    batting_order = {}
    for player_id, player_data in players.items():
        # Only include players in the batting order (not all players)
        if 'battingOrder' in player_data:
            # Convert from string like "100" to position 1
            order_value = int(player_data['battingOrder']) / 100
            position = player_data.get('position', {}).get('abbreviation', '')
            name = player_data.get('person', {}).get('fullName', '')
            pid = player_data.get('person', {}).get('id', '')
            
            batting_order[order_value] = {
                'player_id': pid,
                'player_name': name,
                'position': position
            }
    
    # Convert to list of records
    for order in sorted(batting_order.keys()):
        player = batting_order[order]
        lineup.append({
            'game_pk': game_pk,
            'game_date': game_date,
            'team': team_id,
            'team_name': team_name,
            'home_away': team_side,
            'batting_order': order,
            'player_id': player['player_id'],
            'player_name': player['player_name'],
            'position': player['position']
        })
    
    return lineup


async def process_games(date=None):
    """Process all games for a given date"""
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    headers = {
        'User-Agent': USER_AGENT or 'NRFI Lineup Processor/1.0'
    }
    
    all_lineups = []
    
    async with aiohttp.ClientSession(headers=headers) as session:
        # Fetch schedule
        logger.info(f"Fetching schedule for {date}")
        schedule_data = await fetch_schedule(session, date)
        
        # Check if any games found
        dates = schedule_data.get('dates', [])
        if not dates:
            logger.info(f"No games scheduled for {date}")
            return []
        
        games = dates[0].get('games', [])
        logger.info(f"Found {len(games)} games scheduled for {date}")
        
        # Process each game
        for game in games:
            game_pk = game.get('gamePk')
            game_date = date
            status = game.get('status', {}).get('detailedState', '')
            
            # Skip if game is postponed
            if status == 'Postponed':
                logger.info(f"Game {game_pk} is postponed, skipping")
                continue
                
            # Get team info
            home_team = game.get('teams', {}).get('home', {}).get('team', {})
            away_team = game.get('teams', {}).get('away', {}).get('team', {})
            
            home_team_name = home_team.get('name', '')
            home_team_id = home_team.get('id', '')
            away_team_name = away_team.get('name', '')
            away_team_id = away_team.get('id', '')
            
            logger.info(f"Processing game {game_pk}: {away_team_name} @ {home_team_name}")
            
            # Fetch lineups
            boxscore_data = await fetch_lineups(session, game_pk)
            
            # Extract lineups if available
            home_lineup = extract_lineup(
                boxscore_data, 'home', game_pk, game_date, home_team_name, home_team_id
            )
            away_lineup = extract_lineup(
                boxscore_data, 'away', game_pk, game_date, away_team_name, away_team_id
            )
            
            # Add to collection
            all_lineups.extend(home_lineup)
            all_lineups.extend(away_lineup)
            
            # Be nice to the API
            await asyncio.sleep(0.5)
            
    return all_lineups


def main():
    """Main function to process daily lineups"""
    parser = argparse.ArgumentParser(description='Process MLB lineups for NRFI predictions')
    parser.add_argument('--force', action='store_true', 
                        help='Force fresh lineup data even if today\'s data exists')
    parser.add_argument('--date', type=str, 
                        help='Specific date to process (YYYY-MM-DD format)')
    args = parser.parse_args()
    
    # Determine the target date
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
        
    # Set output file path
    daily_lineups_path = os.path.join(NRFI_DIR, "F1_daily_lineups.csv")
    
    # Check if we already have today's data
    if not args.force and os.path.exists(daily_lineups_path):
        try:
            existing_data = pd.read_csv(daily_lineups_path)
            if not existing_data.empty and 'game_date' in existing_data.columns:
                latest_date = pd.to_datetime(existing_data['game_date']).max().strftime('%Y-%m-%d')
                if latest_date == target_date:
                    logger.info(f"Already have lineup data for {target_date}. Use --force to refresh.")
                    return
        except Exception as e:
            logger.warning(f"Error reading existing lineup data: {e}")
    
    # Process the lineups
    try:
        logger.info(f"Processing lineups for {target_date}")
        lineups = asyncio.run(process_games(target_date))
        
        if not lineups:
            logger.warning(f"No lineup data found for {target_date}")
            return
            
        # Convert to DataFrame
        lineups_df = pd.DataFrame(lineups)
        
        # Save to CSV
        mode = 'w'  # Always overwrite for daily lineups
        lineups_df.to_csv(daily_lineups_path, index=False, mode=mode)
        
        logger.info(f"Saved {len(lineups_df)} lineup entries to {daily_lineups_path}")
        print(f"Successfully processed lineups for {target_date}: {len(lineups_df)} entries")
        
    except Exception as e:
        logger.error(f"Error processing lineups: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
