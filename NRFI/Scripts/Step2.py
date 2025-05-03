import re
import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta, date
import os
import sys
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directory to Python path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, START_DATE, CURRENT_DATE, setup_logging,
                   MLB_SEASON_DATES, TEMP_DIR, NRFI_DIR)

# Setup logging using centralized configuration
logger = setup_logging('Step2')

def is_valid_mlb_date(date_obj: datetime) -> bool:
    """
    Check if a given date falls within an MLB season (regular season or postseason).
    
    Args:
        date_obj: datetime object to check
    
    Returns:
        bool: True if date is within MLB season, False otherwise
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

# Update file paths to use NRFI_DIR instead of DATA_DIR
HISTORICAL_FILE = os.path.join(NRFI_DIR, "F1_historical_schedule.csv")
DAILY_STARTERS_FILE = os.path.join(NRFI_DIR, "F1_daily_starters.csv")

if sys.platform == 'win32':
    # Set the event loop policy to use the SelectorEventLoop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def fetch_schedule(session: aiohttp.ClientSession, date: str = None) -> Dict:
    """
    Fetch MLB schedule for a given date with probable pitchers, weather, and first inning scores
    """
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher,team,weather,linescore"

    retries = 3
    while retries > 0:
        try:
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            retries -= 1
            if retries == 0:
                raise e
            await asyncio.sleep(1)
    return None



def parse_wind(wind: str) -> tuple[float, str]:
    """
    Parse wind string into speed and direction
    """
    if not wind:
        return None, None

    wind_speed = None
    wind_direction = None

    speed_match = re.search(r'(\d+\.?\d*) mph', wind)
    if speed_match:
        wind_speed = float(speed_match.group(1))

    direction_match = re.search(r'mph,\s(.+)', wind)
    if direction_match:
        wind_direction = direction_match.group(1)

    return wind_speed, wind_direction


def process_schedule_data(schedule_data: Dict, historical_data: pd.DataFrame = None) -> List[Dict]:
    """Process the schedule data into a flat structure for DataFrame with weather & first inning scores"""
    processed_games = []

    for date in schedule_data.get('dates', []):
        for game in date.get('games', []):
            # Skip games with status "Postponed"
            if game.get('status', {}).get('detailedState') == "Postponed":
                continue

            linescore = game.get('linescore', {}).get('innings', [])

            # Extract first inning stats if available
            home_inning_1_runs = None
            home_inning_1_hits = None
            home_inning_1_errors = None
            home_inning_1_leftOnBase = None
            
            away_inning_1_runs = None
            away_inning_1_hits = None
            away_inning_1_errors = None
            away_inning_1_leftOnBase = None

            if len(linescore) >= 1:  # Check if at least first inning data exists
                home_inning_1_data = linescore[0].get('home', {})
                away_inning_1_data = linescore[0].get('away', {})

                home_inning_1_runs = home_inning_1_data.get('runs')
                home_inning_1_hits = home_inning_1_data.get('hits')
                home_inning_1_errors = home_inning_1_data.get('errors')
                home_inning_1_leftOnBase = home_inning_1_data.get('leftOnBase')

                away_inning_1_runs = away_inning_1_data.get('runs')
                away_inning_1_hits = away_inning_1_data.get('hits')
                away_inning_1_errors = away_inning_1_data.get('errors')
                away_inning_1_leftOnBase = away_inning_1_data.get('leftOnBase')

            # Extract weather data if available
            weather = game.get('weather', {})
            temperature = weather.get('temp')
            condition = weather.get('condition')
            wind_raw = weather.get('wind')
            wind_speed, wind_direction = parse_wind(wind_raw)

            # Fill missing weather data using historical data
            if historical_data is not None and (pd.isna(temperature) or pd.isna(condition)):
                venue_name = game.get('venue', {}).get('name')
                recent_weather = historical_data[historical_data['venue_name'] == venue_name].sort_values('date', ascending=False).head(1)
                if not recent_weather.empty:
                    temperature = temperature or recent_weather['temperature'].values[0]
                    condition = condition or recent_weather['condition'].values[0]
                    wind_speed = wind_speed or recent_weather['wind_speed'].values[0]
                    wind_direction = wind_direction or recent_weather['wind_direction'].values[0]

            game_info = {
                'date': date.get('date'),
                'game_pk': game.get('gamePk'),
                'status': game.get('status', {}).get('detailedState'),
                'game_time': game.get('gameDate'),

                'home_team': game.get('teams', {}).get('home', {}).get('team', {}).get('name'),
                'home_team_id': game.get('teams', {}).get('home', {}).get('team', {}).get('id'),
                'home_pitcher_name': game.get('teams', {}).get('home', {}).get('probablePitcher', {}).get('fullName'),
                'home_pitcher_id': game.get('teams', {}).get('home', {}).get('probablePitcher', {}).get('id'),

                'away_team': game.get('teams', {}).get('away', {}).get('team', {}).get('name'),
                'away_team_id': game.get('teams', {}).get('away', {}).get('team', {}).get('id'),
                'away_pitcher_name': game.get('teams', {}).get('away', {}).get('probablePitcher', {}).get('fullName'),
                'away_pitcher_id': game.get('teams', {}).get('away', {}).get('probablePitcher', {}).get('id'),

                'venue_name': game.get('venue', {}).get('name'),

                'home_inning_1_runs': home_inning_1_runs,
                'home_inning_1_hits': home_inning_1_hits,
                'home_inning_1_errors': home_inning_1_errors,
                'home_inning_1_leftOnBase': home_inning_1_leftOnBase,

                'away_inning_1_runs': away_inning_1_runs,
                'away_inning_1_hits': away_inning_1_hits,
                'away_inning_1_errors': away_inning_1_errors,
                'away_inning_1_leftOnBase': away_inning_1_leftOnBase,

                'temperature': temperature,
                'condition': condition,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
            }

            processed_games.append(game_info)

    return processed_games


async def fetch_historical_schedule(start_date: str = None) -> pd.DataFrame:
    """
    Fetch MLB schedule from start_date up to and including today with concurrent requests,
    only for dates within MLB season.
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    end_date = datetime.now().strftime('%Y-%m-%d')
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    # Filter for valid MLB dates
    dates = []
    current = current_date
    while current <= end_date_obj:
        if is_valid_mlb_date(current):
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    if not dates:
        logger.info("No valid MLB season dates found in the specified range")
        return pd.DataFrame()

    all_games = []
    batch_size = 10
    
    with tqdm(total=len(dates), desc="Fetching schedule") as pbar:
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(dates), batch_size):
                batch = dates[i:i + batch_size]
                tasks = [fetch_schedule(session, date) for date in batch]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                for response in responses:
                    if isinstance(response, Exception):
                        logger.error(f"Error fetching schedule: {response}")
                        continue

                    if response:
                        processed_games = process_schedule_data(response)
                        all_games.extend(processed_games)

                pbar.update(len(batch))

    return pd.DataFrame(all_games)


async def sync_historical_data():
    """Sync historical schedule data from configured start date with smart updates"""
    os.makedirs(NRFI_DIR, exist_ok=True) # Use NRFI_DIR
    historical_file = HISTORICAL_FILE
    
    # Get current date for boundary checking
    end_date = CURRENT_DATE - timedelta(days=1)
    today = datetime.now().date()
    
    existing_data = None
    last_date = None
    games_to_update = pd.DataFrame()
    
    # Define expected columns to ensure consistency
    expected_columns = [
        'date', 'game_pk', 'status', 'game_time', 
        'home_team', 'home_team_id', 'home_pitcher_name', 'home_pitcher_id',
        'away_team', 'away_team_id', 'away_pitcher_name', 'away_pitcher_id',
        'venue_name', 'home_inning_1_runs', 'home_inning_1_hits', 
        'home_inning_1_errors', 'home_inning_1_leftOnBase',
        'away_inning_1_runs', 'away_inning_1_hits', 'away_inning_1_errors',
        'away_inning_1_leftOnBase', 'temperature', 'condition',
        'wind_speed', 'wind_direction'
    ]
    
    if os.path.exists(historical_file):
        try:
            # Load existing data
            existing_data = pd.read_csv(historical_file)
            
            # Parse dates handling both date-only and datetime formats
            existing_data['date'] = pd.to_datetime(existing_data['date']).dt.tz_localize(None)  # Make timezone naive
            existing_data['game_time'] = pd.to_datetime(existing_data['game_time']).dt.tz_localize(None)
            
            # Add any missing columns with NaN values
            for col in expected_columns:
                if col not in existing_data.columns:
                    existing_data[col] = pd.NA
            
            # Find games that need updating (recent games that might have been completed)
            recent_cutoff = pd.Timestamp(today - timedelta(days=3)).tz_localize(None)
            games_to_update = existing_data[
                (existing_data['date'] >= recent_cutoff) & 
                (existing_data['status'] != 'Final')
            ].copy()
            
            # Remove these games from existing data as we'll fetch fresh data for them
            if not games_to_update.empty:
                existing_data = existing_data[~existing_data['game_pk'].isin(games_to_update['game_pk'])]
            
            # Find the last date in our dataset for new game fetching
            last_date = existing_data['date'].max()
            
            print(f"Found {len(games_to_update)} recent games to update")
            start_date = last_date + timedelta(days=1)
            
        except Exception as e:
            print(f"Error reading historical file: {e}")
            start_date = START_DATE
            existing_data = pd.DataFrame(columns=expected_columns)
    else:
        start_date = START_DATE
        existing_data = pd.DataFrame(columns=expected_columns)
    
    start_date = start_date.strftime('%Y-%m-%d') if isinstance(start_date, (date, datetime)) else START_DATE.strftime('%Y-%m-%d')
    print(f"Fetching new games from {start_date} to {end_date.strftime('%Y-%m-%d')}")
    
    # Fetch new data
    new_data = await fetch_historical_schedule(start_date)
    
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data)
        
    # Ensure new_data has all expected columns
    for col in expected_columns:
        if col not in new_data.columns:
            new_data[col] = pd.NA
    
    if not new_data.empty:
        # Process dates consistently - convert all to timezone naive
        new_data['date'] = pd.to_datetime(new_data['date']).dt.tz_localize(None)
        new_data['game_time'] = pd.to_datetime(new_data['game_time']).dt.tz_localize(None)
        
        # If we have games to update, fetch their fresh data
        if not games_to_update.empty:
            update_dates = games_to_update['date'].dt.strftime('%Y-%m-%d').unique()
            update_data = []
            
            async with aiohttp.ClientSession() as session:
                for date_str in update_dates:
                    schedule = await fetch_schedule(session, date_str)
                    if schedule:
                        processed_games = process_schedule_data(schedule)
                        update_data.extend(processed_games)
            
            if update_data:
                update_df = pd.DataFrame(update_data)
                # Ensure update_df has all expected columns
                for col in expected_columns:
                    if col not in update_df.columns:
                        update_df[col] = pd.NA
                
                # Convert update_df dates consistently
                update_df['date'] = pd.to_datetime(update_df['date']).dt.tz_localize(None)
                update_df['game_time'] = pd.to_datetime(update_df['game_time']).dt.tz_localize(None)
                
                # Add updated games to new_data
                new_data = pd.concat([new_data, update_df], ignore_index=True)
        
        # Combine with existing data
        if not existing_data.empty:
            # Ensure both DataFrames have the same columns and timezone handling before concatenation
            existing_data = existing_data[expected_columns].copy()
            new_data = new_data[expected_columns].copy()
            
            # Ensure dates are timezone naive in both DataFrames
            existing_data['date'] = pd.to_datetime(existing_data['date']).dt.tz_localize(None)
            existing_data['game_time'] = pd.to_datetime(existing_data['game_time']).dt.tz_localize(None)
            
            # Handle empty/NA columns before concatenation
            for col in expected_columns:
                # Convert all-NA columns to appropriate dtype based on column name
                if existing_data[col].isna().all() or new_data[col].isna().all():
                    if col in ['date', 'game_time']:
                        dtype = 'datetime64[ns]'
                    elif col in ['game_pk', 'home_team_id', 'away_team_id', 
                               'home_pitcher_id', 'away_pitcher_id']:
                        dtype = 'Int64'  # Nullable integer type
                    elif col in ['temperature', 'wind_speed']:
                        dtype = 'float64'
                    else:
                        dtype = 'string'
                    
                    existing_data[col] = existing_data[col].astype(dtype)
                    new_data[col] = new_data[col].astype(dtype)
            
            # Perform concatenation after type alignment
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Remove duplicates, keeping the newer version of any duplicate games
            combined_data = combined_data.sort_values('date').drop_duplicates(
                subset=['date', 'game_pk'], 
                keep='last'
            )
        else:
            combined_data = new_data[expected_columns]
        
        # Save the final dataset
        combined_data.sort_values('date', inplace=True)
        
        # Convert dates to string format for storage
        combined_data['date'] = combined_data['date'].dt.strftime('%Y-%m-%d')
        combined_data['game_time'] = combined_data['game_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        combined_data.to_csv(historical_file, index=False)
        print(f"Updated historical data: {len(new_data)} new/updated games added/modified")
        print(f"Total games in dataset: {len(combined_data)}")
    else:
        print("No new historical data to save")


async def save_daily_starters():
    """
    Save game schedule with probable pitchers for today only if it's during MLB season
    """
    os.makedirs(NRFI_DIR, exist_ok=True)
    daily_starters_file = DAILY_STARTERS_FILE

    today = datetime.now()
    if not is_valid_mlb_date(today):
        logger.info(f"Today ({today.date()}) is not within MLB season. Skipping daily starters update.")
        return

    all_games = []

    # Load historical data for weather predictions
    historical_data = None
    if os.path.exists(HISTORICAL_FILE):
        historical_data = pd.read_csv(HISTORICAL_FILE)

    async with aiohttp.ClientSession() as session:
        date_str = today.strftime('%Y-%m-%d')
        logger.info(f"Fetching schedule for {date_str}")
        schedule = await fetch_schedule(session, date_str)
        
        if schedule:
            games = process_schedule_data(schedule, historical_data)
            all_games.extend(games)
            logger.info(f"Found {len(games)} games for today")

    if all_games:
        df = pd.DataFrame(all_games)
        df.to_csv(daily_starters_file, index=False)
        print(f"Saved {len(df)} games to daily starters file for today")
    else:
        print("No games scheduled for today")


async def main():
    """Main function to run both historical and daily updates"""
    logger.info("Starting MLB schedule data update process")
    
    try:
        # Run historical update
        await sync_historical_data()
        # Save daily starters
        await save_daily_starters()
        logger.info("MLB schedule data update completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())