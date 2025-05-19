'''
Step4.py - MLB Lineup and Probable Pitcher Collector

This script fetches and maintains historical MLB lineup and probable pitcher data.
It uses the MLB Stats API to collect detailed team and player information before games.

Key Features:
- Asynchronous API requests with aiohttp
- Smart date handling for season ranges
- Incremental data collection and updates
- Progress tracking and periodic saves
- Comprehensive lineup data including:
    * Confirmed lineup status
    * Starting pitchers
    * Batting orders
    * Team matchups

Output Files:
    - data/probable_lineups_historical.csv: Contains lineup data with:
        * Game dates and teams
        * Starting pitchers
        * Lineup confirmation status
        * Complete batting orders
        * Team matchup details

Dependencies:
    - aiohttp: Async HTTP requests
    - pandas: Data processing and storage
    - asyncio: Asynchronous operations
    - tqdm: Progress tracking

Configuration:
    - Handles MLB seasons from 2024 onwards
    - Includes spring training through post-season
    - Auto-detects and continues from last processed date
    - Saves progress after each date processed
'''



import sys
import aiohttp
import asyncio
import pandas as pd
import os
from tqdm import tqdm
from aiohttp import ClientTimeout
from typing import List, Dict
from datetime import datetime, timedelta
import gc

# Configure event loop policy for Windows
if sys.platform.startswith('win'):
    import platform
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add parent directory to Python path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (START_SEASON, setup_logging, MLB_SEASON_DATES, NRFI_DIR)

# Setup logging using centralized configuration
logger = setup_logging('Step4')

# Update file paths to use NRFI_DIR instead of DATA_DIR
LINEUPS_FILE = os.path.join(NRFI_DIR, "probable_lineups_historical.csv")

MAX_CONCURRENT_REQUESTS = 50  # Limit concurrent connections
CHUNK_SIZE = 10  # Number of games to process in each batch

async def fetch_with_retry(session: aiohttp.ClientSession, url: str, retries: int = 3) -> Dict:
    """Fetch data from URL with retry logic"""
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed to fetch {url}: {str(e)}")
                return {}
            await asyncio.sleep(1)

async def fetch_probable_pitchers(session: aiohttp.ClientSession, date: str) -> Dict:
    """Fetch probable pitchers asynchronously"""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher"
    return await fetch_with_retry(session, url)

async def fetch_lineups(session: aiohttp.ClientSession, game_pk: str) -> Dict:
    """Fetch lineups asynchronously"""
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    return await fetch_with_retry(session, url)


async def process_games(games: List[Dict]) -> List[List]:
    """Process games concurrently with connection pooling"""
    timeout = ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        lineups_data = []
        
        # Process games in smaller chunks
        for i in range(0, len(games), CHUNK_SIZE):
            chunk = games[i:i + CHUNK_SIZE]
            tasks = []
            
            # Create tasks for fetching lineups
            for game in chunk:
                game_pk = game.get('gamePk', 'N/A')
                tasks.append(fetch_lineups(session, game_pk))
            
            # Process chunk of games with progress bar
            print(f"Fetching lineup data for games {i+1}-{min(i+CHUNK_SIZE, len(games))}...")
            lineups_results = []
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing games"):
                lineups_results.append(await f)
            
            # Process results for this chunk
            for game, lineup in zip(chunk, lineups_results):
                game_date = game.get('officialDate', 'N/A')
                teams = game.get('teams', {})
                home_team = teams.get('home', {}).get('team', {}).get('name', 'N/A')
                away_team = teams.get('away', {}).get('team', {}).get('name', 'N/A')
                home_pitcher = teams.get('home', {}).get('probablePitcher', {}).get('fullName', 'N/A')
                away_pitcher = teams.get('away', {}).get('probablePitcher', {}).get('fullName', 'N/A')

                home_batters = lineup.get('teams', {}).get('home', {}).get('batters', [])
                away_batters = lineup.get('teams', {}).get('away', {}).get('batters', [])
                home_confirmed = bool(home_batters)
                away_confirmed = bool(away_batters)

                lineups_data.append([
                    game_date, home_team, home_pitcher, home_confirmed, home_batters,
                    away_team, away_pitcher, away_confirmed, away_batters
                ])
            
            # Add small delay between chunks
            await asyncio.sleep(0.5)
        
        return lineups_data

def get_last_processed_date(csv_path: str) -> datetime:
    """Get the most recent date from existing CSV file"""
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty or 'Date' not in df.columns:
            return None
        # Subtract one day from last date to ensure we reprocess the last day
        last_date = pd.to_datetime(df['Date']).max() - timedelta(days=1)
        return last_date.to_pydatetime()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

async def save_lineups_data(lineups_data: List[List]):
    """Save lineups data with temp file handling"""
    try:
        df = pd.DataFrame(lineups_data, columns=[
            "Date", "Home Team", "Home Pitcher", "Home Lineup Confirmed", "Home Lineup",
            "Away Team", "Away Pitcher", "Away Lineup Confirmed", "Away Lineup"
        ])
        
        # Append to existing file if it exists
        mode = 'a' if os.path.exists(LINEUPS_FILE) else 'w'
        header = not os.path.exists(LINEUPS_FILE)
        
        df.to_csv(LINEUPS_FILE, mode=mode, header=header, index=False)
        logger.info(f"Successfully saved {len(lineups_data)} records")
        
    except Exception as e:
        logger.error(f"Error saving lineups data: {e}")
        raise

def is_valid_mlb_date(date_obj: datetime) -> bool:
    """
    Check if a given date falls within an MLB season.
    """
    year = date_obj.year
    if year not in MLB_SEASON_DATES:
        logger.debug(f"No MLB season data available for year {year}")
        return False
        
    season = MLB_SEASON_DATES[year]
    regular_start, regular_end = [datetime.strptime(d, '%Y-%m-%d').date() for d in season['regular_season']]
    post_start, post_end = [datetime.strptime(d, '%Y-%m-%d').date() for d in season['postseason']]
    
    date_to_check = date_obj.date()
    is_valid = (regular_start <= date_to_check <= regular_end or 
                post_start <= date_to_check <= post_end)
    
    if not is_valid:
        logger.debug(f"Date {date_to_check} is outside MLB season for year {year}")
    
    return is_valid

async def main_async():
    """Process MLB lineups data asynchronously using MLB season dates"""
    today = datetime.now()
    three_days_ahead = today + timedelta(days=3)
    
    # Get the last processed date from existing CSV
    last_processed = get_last_processed_date(LINEUPS_FILE)
    
    if last_processed:
        start_year = last_processed.year
        # Always reprocess the last 3 days to catch updates to lineups
        reprocess_date = today - timedelta(days=3)
        if last_processed > reprocess_date:
            last_processed = reprocess_date
            logger.info(f"Forcing reprocess of the last 3 days to catch lineup updates")
    else:
        start_year = START_SEASON
        
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process one year at a time to manage memory
        for year in range(start_year, today.year + 1):
            logger.info(f"Processing year {year}")
            
            if year not in MLB_SEASON_DATES:
                logger.warning(f"No MLB season data for year {year}, skipping")
                continue
                
            season = MLB_SEASON_DATES[year]
            regular_start = datetime.strptime(season['regular_season'][0], '%Y-%m-%d')
            post_end = datetime.strptime(season['postseason'][1], '%Y-%m-%d')
            
            # Adjust start/end dates
            if last_processed and year == last_processed.year:
                range_start = last_processed
            else:
                range_start = regular_start
                
            range_end = min(three_days_ahead, post_end) if year == today.year else post_end
            
            # Skip if range_end is before range_start (can happen if trying to process next year)
            if range_end < range_start:
                logger.info(f"Skipping year {year}: end date before start date")
                continue
                
            logger.info(f"Date range for {year}: {range_start.strftime('%Y-%m-%d')} to {range_end.strftime('%Y-%m-%d')}")
            
            # Process in smaller chunks (e.g., monthly)
            current = range_start
            while current <= range_end:
                chunk_end = min(current + timedelta(days=30), range_end)
                dates_chunk = []
                
                while current <= chunk_end:
                    if is_valid_mlb_date(current):
                        dates_chunk.append(current.strftime('%Y-%m-%d'))
                    current += timedelta(days=1)
                
                if not dates_chunk:
                    continue
                    
                logger.info(f"Processing dates from {dates_chunk[0]} to {dates_chunk[-1]}")
                
                # Process chunk and periodically clear memory
                chunk_data = []
                for date in tqdm(dates_chunk, desc=f"Processing {year} games"):
                    data = await fetch_probable_pitchers(session, date)
                    games = data.get('dates', [])
                    if not games:
                        continue
                        
                    games = games[0].get('games', [])
                    if games:
                        lineups_data = await process_games(games)
                        chunk_data.extend(lineups_data)
                    
                    # Save more frequently and clear memory
                    if len(chunk_data) >= 50:
                        await save_lineups_data(chunk_data)
                        chunk_data = []
                        
                    await asyncio.sleep(1)  # Rate limiting
                
                # Save any remaining data
                if chunk_data:
                    await save_lineups_data(chunk_data)
                
                # Force garbage collection after each chunk
                gc.collect()

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
