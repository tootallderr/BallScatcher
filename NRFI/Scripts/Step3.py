'''
Step3.py - First Inning Score Collector

This script collects detailed first inning statistics for MLB games using the MLB Stats API.
It processes both historical and daily game data, focusing on first inning performance.

Key Features:
- Asynchronous data collection with aiohttp
- Retry mechanism with exponential backoff
- Batch processing to manage API load
- Incremental updates with change tracking
- Detailed first inning statistics including:
    * Runs scored
    * Hits
    * Errors
    * Left on base
    * Starting pitchers

Output Files:
    - data/first_inning_scores.csv: Contains first inning statistics with:
        * Game identifiers and dates
        * Home/Away team performance
        * Starting pitcher information
        * Detailed first inning stats

Dependencies:
    - aiohttp: Async HTTP requests
    - requests: Standard HTTP requests
    - pandas: Data processing
    - tqdm: Progress tracking
    - asyncio: Asynchronous operations

Configuration:
    Required config.py variables:
    - START_SEASON: Initial season to collect from
    - DATA_DIR: Directory for storing CSV files
'''



import asyncio
import sys
import aiohttp
from itertools import islice
import os
import random
import json
import csv
from datetime import datetime, timedelta, date
import requests
from tqdm import tqdm
from functools import wraps
import time

if sys.platform == 'win32':
    # Set the event loop policy to use the SelectorEventLoop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add parent directory to Python path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, START_DATE, START_SEASON, USER_AGENT, setup_logging,
                   MLB_SEASON_DATES, TEMP_DIR, NRFI_DIR) # Import NRFI_DIR

# Setup logging using centralized configuration
logger = setup_logging('Step3')

# Update file paths to use NRFI_DIR instead of DATA_DIR
FIRST_INNING_SCORES_FILE = os.path.join(NRFI_DIR, "F1_inning_scores.csv")

def is_valid_mlb_date(date_obj):
    """
    Check if a given date falls within an MLB season (regular season or postseason).
    
    Args:
        date_obj: datetime object or date object to check
    
    Returns:
        bool: True if date is within an MLB season, False otherwise
    """
    # Convert to date object if it's a datetime
    if isinstance(date_obj, datetime):
        date_to_check = date_obj.date()
    else:
        date_to_check = date_obj
    
    year = date_to_check.year
    if year not in MLB_SEASON_DATES:
        logger.debug(f"No MLB season data available for year {year}")
        return False
        
    season = MLB_SEASON_DATES[year]
    regular_start, regular_end = [datetime.strptime(d, '%Y-%m-%d').date() for d in season['regular_season']]
    post_start, post_end = [datetime.strptime(d, '%Y-%m-%d').date() for d in season['postseason']]
    
    is_valid = (regular_start <= date_to_check <= regular_end or 
                post_start <= date_to_check <= post_end)
    
    if not is_valid:
        logger.debug(f"Date {date_to_check} is outside MLB season for year {year}")
    
    return is_valid

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x + 
                            random.uniform(0, 1))
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def create_data_directory():
    os.makedirs(NRFI_DIR, exist_ok=True) # Use NRFI_DIR
    return NRFI_DIR

@retry_with_backoff()
def fetch_schedule(date):
    schedule_url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}'
    response = requests.get(schedule_url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Failed to retrieve schedule for {date}')
        return None

def extract_game_pks(schedule):
    if schedule and 'dates' in schedule and len(schedule['dates']) > 0:
        return [game['gamePk'] for game in schedule['dates'][0]['games']]
    else:
        print('No games found for the given date.')
        return []

@retry_with_backoff()
def fetch_linescore(game_pk):
    linescore_url = f'https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore'
    response = requests.get(linescore_url)
    if response.status_code == 200:
        data = response.json()
        # Debug the raw response
        print(f"Raw linescore response for game {game_pk}:")
        print(json.dumps(data, indent=2))
        return data
    else:
        print(f'Failed to retrieve linescore for gamePk: {game_pk}. Status: {response.status_code}')
        print(f'Response: {response.text}')
        return None

@retry_with_backoff()
def fetch_game_data(game_pk):
    """Fetch game data including starting pitchers"""
    url = f'https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Failed to retrieve game data for gamePk: {game_pk}')
        return None

def extract_first_inning_data(linescore):
    """Extract first inning scores and additional stats with better error handling"""
    if not linescore or 'innings' not in linescore:
        print('No linescore data available')
        return None, None

    innings = linescore.get('innings', [])
    if not innings:
        print('No innings data found')
        return None, None

    first_inning = innings[0]
    
    # Add debugging to see raw data
    print(f"First inning data: {json.dumps(first_inning, indent=2)}")
    
    # Handle different possible data structures
    if isinstance(first_inning, dict):
        away_stats = first_inning.get('away', {})
        home_stats = first_inning.get('home', {})
        
        away_data = {
            'runs': away_stats.get('runs', 0),
            'hits': away_stats.get('hits', 0),
            'errors': away_stats.get('errors', 0),
            'leftOnBase': away_stats.get('leftOnBase', 0)
        }
        
        home_data = {
            'runs': home_stats.get('runs', 0),
            'hits': home_stats.get('hits', 0),
            'errors': home_stats.get('errors', 0),
            'leftOnBase': home_stats.get('leftOnBase', 0)
        }
        
        # Debug the extracted values
        print(f"Extracted away stats: {away_data}")
        print(f"Extracted home stats: {home_data}")
        
        return away_data, home_data
    else:
        print(f'Unexpected first inning data format: {type(first_inning)}')
        return None, None

def get_existing_games():
    """Load existing game data and return set of game PKs with dates"""
    csv_file = FIRST_INNING_SCORES_FILE # Use updated variable
    existing_games = {}
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_games[row['gamePk']] = {
                    'away_runs': row['away_runs'],
                    'away_hits': row['away_hits'],
                    'away_errors': row['away_errors'],
                    'away_lob': row['away_lob'],
                    'home_runs': row['home_runs'],
                    'home_hits': row['home_hits'],
                    'home_errors': row['home_errors'],
                    'home_lob': row['home_lob']
                }
    return existing_games

def save_game_data_to_csv(new_game_data):
    """Save game data with additional statistics to CSV while preserving existing data"""
    # Use TEMP_DIR from config instead of system temp
    temp_filename = f'first_inning_scores_{int(time.time())}.temp'
    temp_file = os.path.join(TEMP_DIR, temp_filename)
    final_file = FIRST_INNING_SCORES_FILE
    
    try:
        # Ensure the temp directory exists
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Load existing data if available
        existing_data = []
        existing_game_pks = set()
        if os.path.exists(final_file):
            with open(final_file, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    existing_data.append(row)
                    existing_game_pks.add(row['gamePk'])
        
        # Write all data to temp file
        with open(temp_file, mode='w', newline='') as file:
            fieldnames = [
                'gamePk', 
                'away_runs', 'away_hits', 'away_errors', 'away_lob',
                'home_runs', 'home_hits', 'home_errors', 'home_lob'
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write existing data first
            for row in existing_data:
                writer.writerow(row)
            
            # Write new games
            for game in new_game_data:
                writer.writerow({
                    'gamePk': game['gamePk'],
                    'away_runs': game['away_stats']['runs'],
                    'away_hits': game['away_stats']['hits'],
                    'away_errors': game['away_stats']['errors'],
                    'away_lob': game['away_stats']['leftOnBase'],
                    'home_runs': game['home_stats']['runs'],
                    'home_hits': game['home_stats']['hits'],
                    'home_errors': game['home_stats']['errors'],
                    'home_lob': game['home_stats']['leftOnBase']
                })
        
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(final_file), exist_ok=True)
        
        # Try to remove the target file first if it exists
        if os.path.exists(final_file):
            os.remove(final_file)
            
        # Move temp file to final location
        os.rename(temp_file, final_file)
        logger.info(f"Successfully added {len(new_game_data)} new games to {final_file}")
        
    except Exception as e:
        logger.error(f"Error saving game data: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

def generate_date_range(start_year, end_date=None, start_date=None):
    """Generate a list of dates from the start of the specified season to tomorrow's date,
    filtering only MLB season dates using MLB_SEASON_DATES"""
    if end_date is None:
        end_date = (datetime.now() + timedelta(days=1)).date()
    
    if start_date is None:
        # Use MLB_SEASON_DATES to determine start date
        if start_year in MLB_SEASON_DATES:
            regular_season_start = datetime.strptime(MLB_SEASON_DATES[start_year]['regular_season'][0], '%Y-%m-%d').date()
            start_date = regular_season_start
        else:
            logger.warning(f"No MLB season data for year {start_year}, using default date")
            start_date = date(start_year, 3, 28)  # Default to late March for unknown years
    
    current_date = start_date
    date_list = []
    skipped_dates = 0
    
    while current_date <= end_date:
        if is_valid_mlb_date(current_date):
            date_list.append(current_date.strftime('%Y-%m-%d'))
        else:
            skipped_dates += 1
        
        current_date += timedelta(days=1)
    
    logger.info(f"Generated {len(date_list)} valid MLB dates, skipped {skipped_dates} non-MLB dates")
    return date_list

async def fetch_schedule_async(session, date):
    """Async version of fetch_schedule"""
    schedule_url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}'
    async with session.get(schedule_url) as response:
        if response.status == 200:
            return await response.json()
        return None

async def fetch_linescore_async(session, game_pk):
    """Async version of fetch_linescore"""
    linescore_url = f'https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore'
    async with session.get(linescore_url) as response:
        if response.status == 200:
            return await response.json()
        return None

async def process_game(session, game_pk, existing_games, is_today_or_future, game_date=None):
    """Process a single game asynchronously"""
    game_pk_str = str(game_pk)
    
    # Change the reprocessing logic to include recent past dates
    current_date = datetime.now().date()
    last_processed_cutoff = current_date - timedelta(days=3)  # Look back 3 days
    
    # Always process if:
    # 1. Game doesn't exist in our data
    # 2. Game is from last 3 days
    # 3. Game is today or in the future
    should_process = (
        game_pk_str not in existing_games or
        is_today_or_future or
        (game_pk_str in existing_games and 
         game_date and game_date >= last_processed_cutoff)
    )
    
    if not should_process:
        return None

    # Fetch both linescore and boxscore
    linescore = await fetch_linescore_async(session, game_pk)
    boxscore = await session.get(f'https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore')
    boxscore_data = await boxscore.json() if boxscore.status == 200 else None

    if linescore and boxscore_data:
        away_stats, home_stats = extract_first_inning_data(linescore)
        
        # Extract starting pitchers with safer access
        away_pitcher = "Unknown"
        home_pitcher = "Unknown"
        
        try:
            teams = boxscore_data.get('teams', {})
            away_pitchers = teams.get('away', {}).get('pitchers', [])
            home_pitchers = teams.get('home', {}).get('pitchers', [])
            
            if away_pitchers:
                away_pitcher_id = away_pitchers[0]
                away_pitcher = teams.get('away', {}).get('players', {}).get(f'ID{away_pitcher_id}', {}).get('person', {}).get('fullName', 'Unknown')
            
            if home_pitchers:
                home_pitcher_id = home_pitchers[0]
                home_pitcher = teams.get('home', {}).get('players', {}).get(f'ID{home_pitcher_id}', {}).get('person', {}).get('fullName', 'Unknown')
                
        except Exception as e:
            print(f"Error extracting pitcher data for game {game_pk}: {str(e)}")

        if away_stats is not None and home_stats is not None:
            return {
                'gamePk': game_pk,
                'away_stats': away_stats,
                'home_stats': home_stats,
                'away_pitcher': away_pitcher,
                'home_pitcher': home_pitcher
            }
    return None

async def process_date(session, date_str, existing_games, pbar):
    """Process all games for a given date asynchronously"""
    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Check if date is within MLB season
    if not is_valid_mlb_date(date_obj):
        logger.debug(f"Skipping {date_str} - not within MLB season")
        pbar.update(1)
        return []
        
    schedule = await fetch_schedule_async(session, date_str)
    if not schedule:
        pbar.update(1)
        return []

    game_pks = extract_game_pks(schedule)
    if not game_pks:
        pbar.update(1)
        return []

    is_today_or_future = date_obj >= datetime.now().date()

    tasks = [process_game(session, game_pk, existing_games, is_today_or_future, date_obj) 
             for game_pk in game_pks]
    results = await asyncio.gather(*tasks)
    
    pbar.update(1)
    pbar.set_postfix({'Games': len(game_pks)})
    
    return [r for r in results if r is not None]

async def process_all_seasons_async():
    """Async version of process_all_seasons"""
    data_dir = create_data_directory()
    existing_games = get_existing_games()
    
    last_sync_file = os.path.join(data_dir, '.last_sync')
    start_date = None
    if os.path.exists(last_sync_file):
        with open(last_sync_file, 'r') as f:
            last_sync = f.read().strip()
            start_date = datetime.strptime(last_sync, '%Y-%m-%d').date()
    
    if not start_date:
        start_date = date(START_SEASON, 3, 15)
    
    dates = generate_date_range(start_date.year, start_date=start_date)
    
    print(f"Processing data from {start_date} to present...")
    new_game_data = []
    
    async with aiohttp.ClientSession() as session:
        with tqdm(total=len(dates), desc="Processing dates", position=0, leave=True) as pbar:
            # Process dates in chunks to avoid overwhelming the API
            chunk_size = 10
            for i in range(0, len(dates), chunk_size):
                date_chunk = dates[i:i + chunk_size]
                tasks = [process_date(session, date_str, existing_games, pbar) 
                        for date_str in date_chunk]
                chunk_results = await asyncio.gather(*tasks)
                
                for results in chunk_results:
                    new_game_data.extend(results)
                    
                    if len(new_game_data) >= 50:
                        save_game_data_to_csv(new_game_data)  
                        new_game_data = []
    
    if new_game_data:
        save_game_data_to_csv(new_game_data)  
    
    with open(last_sync_file, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d'))

def process_all_seasons():
    """Wrapper to run async code"""
    asyncio.run(process_all_seasons_async())

def main(date=None, days_ahead=3):
    data_dir = create_data_directory()
    
    if date == "all":
        process_all_seasons()
        return
    
    if date is None:
        # You could use today instead of yesterday as the starting point
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Process each day from the specified date through X days ahead
    start_date = datetime.strptime(date, '%Y-%m-%d')
    for day_offset in range(days_ahead + 1):  # +1 to include the start date
        target_date = start_date + timedelta(days=day_offset)
        
        # Skip dates outside MLB season
        if not is_valid_mlb_date(target_date):
            logger.info(f"Skipping {target_date.strftime('%Y-%m-%d')} - not within MLB season")
            continue
            
        target_date_str = target_date.strftime('%Y-%m-%d')
        logger.info(f"Processing date: {target_date_str}")
        schedule = fetch_schedule(target_date_str)
        game_pks = extract_game_pks(schedule)
        all_game_data = []

        if game_pks:
            logger.info(f"Processing {len(game_pks)} games from {target_date_str}")
            for game_pk in tqdm(game_pks, desc="Fetching game data"):
                linescore = fetch_linescore(game_pk)
                if linescore:
                    away_stats, home_stats = extract_first_inning_data(linescore)
                    if away_stats is not None and home_stats is not None:
                        all_game_data.append({
                            'gamePk': game_pk,
                            'away_stats': away_stats,
                            'home_stats': home_stats
                        })

        if all_game_data:
            save_game_data_to_csv(all_game_data)

if __name__ == '__main__':
    # To process all seasons from START_SEASON, use:
    main("all")
    
    # To process a specific date or default to yesterday, use:
    # main()  # or main("2024-05-15")
