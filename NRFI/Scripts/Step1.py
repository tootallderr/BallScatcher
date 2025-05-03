'''
Step1.py - Baseball Data Scraper for NRFI Analysis

This script scrapes first-inning baseball statistics from Baseball Savant for both pitchers and batters.
It includes functionality for both historical data collection and daily updates.

Key Features:
- Concurrent scraping of pitcher and batter data using ThreadPoolExecutor
- Incremental data updates by checking existing CSV files
- Progress bars for download monitoring
- Automatic handling of monthly date ranges
- Robust error handling and logging
- Support for both full historical and daily update modes

Usage:
    Regular historical update: python Step1.py
    Daily update only: python Step1.py --daily

Output Files:
    - data/1stInningPitchersHistorical.csv: Historical pitcher statistics
    - data/1stInningBattersHistorical.csv: Historical batter statistics
    - scraper.log: Detailed logging of the scraping process

Dependencies:
    - pandas
    - requests
    - tqdm
    - logging
    - concurrent.futures

Configuration:
    Required config.py variables:
    - START_SEASON: Initial season to scrape from
    - START_DATE: Initial date to scrape from
    - CURRENT_DATE: Current date cutoff
    - USER_AGENT: User agent string for requests
    - DATA_DIR: Directory for storing CSV files
'''

import calendar
import os
import sys
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, START_DATE, USER_AGENT, setup_logging, 
                   TEMP_DIR, PITCHERS_CSV, BATTERS_CSV, MLB_SEASON_DATES) # Added MLB_SEASON_DATES

# Setup logging using centralized configuration
logger = setup_logging('Step1')

# Create a global session to reuse connections
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

def is_valid_mlb_date(date):
    """
    Check if a given date falls within an MLB season (regular season or postseason).
    
    Args:
        date: datetime object to check
    
    Returns:
        bool: True if date is within an MLB season, False otherwise
    """
    year = date.year
    date_str = date.strftime('%Y-%m-%d')
    
    if year not in MLB_SEASON_DATES:
        return False
        
    season = MLB_SEASON_DATES[year]
    regular_start, regular_end = [datetime.strptime(d, '%Y-%m-%d') for d in season['regular_season']]
    post_start, post_end = [datetime.strptime(d, '%Y-%m-%d') for d in season['postseason']]
    
    return (regular_start.date() <= date.date() <= regular_end.date() or 
            post_start.date() <= date.date() <= post_end.date())

def get_date_ranges(file_path, is_daily=False):
    # Get yesterday's date instead of current date
    end_date = pd.Timestamp(datetime.now().date() - timedelta(days=1))
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if 'game_date' in df.columns and not df.empty:
                df['game_date'] = pd.to_datetime(df['game_date'])
                last_date = df['game_date'].max()
                start_date = pd.to_datetime(last_date + timedelta(days=1))
                if start_date > end_date:
                    logger.info("Data is already up to date through yesterday")
                    return []
                logger.info(f"Found existing data up to {last_date.strftime('%Y-%m-%d')}. Starting from {start_date.strftime('%Y-%m-%d')}")
            else:
                logger.warning(f"No game_date column or empty file found in {file_path}")
                start_date = pd.to_datetime(START_DATE)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"Could not parse {file_path}, error: {e}. Starting from START_DATE")
            start_date = pd.to_datetime(START_DATE)
    else:
        logger.info(f"No existing file found at {file_path}. Starting from START_DATE: {START_DATE.strftime('%Y-%m-%d')}")
        start_date = pd.to_datetime(START_DATE)
    
    if is_daily:
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        if is_valid_mlb_date(datetime(yesterday.year, yesterday.month, yesterday.day)):
            return [(yesterday, yesterday)]
        else:
            logger.info(f"Yesterday ({yesterday}) is not within MLB season. Skipping.")
            return []
    
    date_ranges = []
    current_date = start_date
    while current_date <= end_date:
        last_day = calendar.monthrange(current_date.year, current_date.month)[1]
        month_end = datetime(current_date.year, current_date.month, last_day)
        if month_end > end_date:
            month_end = end_date
            
        # Find the actual range of valid MLB dates within this month
        current = current_date
        valid_start = None
        valid_end = None
        
        while current <= month_end:
            if is_valid_mlb_date(current):
                if valid_start is None:
                    valid_start = current
                valid_end = current
            elif valid_start is not None:
                # We found a gap in MLB dates, add the range we've collected
                date_ranges.append((valid_start, valid_end))
                valid_start = None
            current += timedelta(days=1)
            
        # Add the final range if we ended in a valid period
        if valid_start is not None:
            date_ranges.append((valid_start, valid_end))
            
        if month_end.month == 12:
            current_date = datetime(month_end.year + 1, 1, 1)
        else:
            current_date = datetime(month_end.year, month_end.month + 1, 1)
    
    if date_ranges:
        logger.info(f"Generated {len(date_ranges)} MLB season date ranges for download")
    else:
        logger.info("No valid MLB season dates found in the specified range")
        
    return date_ranges

def download_with_progress(url, headers, csv_path, desc, append=False):
    """Helper function to download a file with a progress bar using the global session."""
    # Create temp filename using original filename + timestamp
    temp_filename = f"{os.path.basename(csv_path)}_{int(time.time())}.temp"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        head_resp = session.head(url, headers=headers)
        total_size = int(head_resp.headers.get('content-length', 0))
    except requests.RequestException as e:
        logger.warning(f"Failed to get content length: {e}")
        total_size = 0
    
    try:
        with session.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()
            # Use a larger chunk size (16KB)
            chunk_size = 16384
            with open(temp_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                mininterval=0.5,
                colour='green'
            ) as pbar:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        size = f.write(chunk)
                        downloaded += size
                        pbar.update(size)
                if total_size != 0 and downloaded != total_size:
                    logger.warning(f"Downloaded size ({downloaded}) does not match expected size ({total_size}).")
        
        try:
            temp_df = pd.read_csv(temp_path)
            if temp_df.empty:
                logger.error("Downloaded file is empty. Skipping.")
                os.remove(temp_path)
                return False
                
            logger.info(f"Downloaded {len(temp_df)} rows successfully")
            
            if append and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                logger.info(f"Appending data to existing file {csv_path}")
                existing_df = pd.read_csv(csv_path)
                combined_df = pd.concat([existing_df, temp_df])
                combined_df.drop_duplicates(inplace=True)
                combined_df.to_csv(csv_path, index=False)
                logger.info(f"Combined data has {len(combined_df)} rows after deduplication")
            else:
                os.replace(temp_path, csv_path)
                logger.info(f"Created new file at {csv_path} with {len(temp_df)} rows")
                
            # Clean up temp file after successful processing
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return True
        
        except pd.errors.EmptyDataError:
            logger.error("Downloaded file is empty or invalid CSV. Skipping.")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    
    except Exception as e:
        logger.error(f"Error during download: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def scrape_1st_inning_pitchers(is_daily=False):
    """
    Scrapes FIRST INNING ONLY Pitcher data from Baseball Savant in incremental chunks
    and updates data/1stInningPitchersHistorical.csv.
    
    Args:
        is_daily: If True, only download today's data.
    """
    os.makedirs(os.path.dirname(PITCHERS_CSV), exist_ok=True) # Ensure directory exists
    csv_path = PITCHERS_CSV # Use PITCHERS_CSV from config
    date_ranges = get_date_ranges(csv_path, is_daily)
    
    if not date_ranges:
        logger.info("First inning pitchers data is already up to date.")
        return
    
    success_count = 0
    for i, (start_date, end_date) in enumerate(date_ranges):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        logger.info(f"Downloading first inning pitchers data for range: {start_str} to {end_str}")
        
        url = (
            "https://baseballsavant.mlb.com/statcast_search/csv?"
            "hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&"
            f"hfSea={start_date.year}%7C&hfSit=&player_type=pitcher&hfOuts=&hfOpponent=&pitcher_throws=&"
            f"batter_stands=&hfSA=&game_date_gt={start_str}&game_date_lt={end_str}&hfMo=&"
            "hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=1%7C&hfBBT=&"  # Note: hfInn=1%7C ensures first inning only
            "hfFlag=&metric_1=&group_by=name-date&min_pitches=0&min_results=0&min_pas=0&"
            "sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc"
        ) + "&" + "&".join([
            "chk_stats_pa=on", "chk_stats_abs=on", "chk_stats_bip=on", "chk_stats_hits=on",
            "chk_stats_singles=on", "chk_stats_dbls=on", "chk_stats_triples=on", "chk_stats_hrs=on",
            "chk_stats_so=on", "chk_stats_k_percent=on", "chk_stats_bb=on", "chk_stats_bb_percent=on",
            "chk_stats_whiffs=on", "chk_stats_swings=on", "chk_stats_api_break_z_with_gravity=on", 
            "chk_stats_api_break_x_arm=on", "chk_stats_api_break_x_batter_in=on", 
            "chk_stats_delev_pitcher_run_exp=on", "chk_stats_delev_pitcher_run_value_per_100=on", 
            "chk_stats_unadj_pitcher_run_exp=on", "chk_stats_velocity=on", 
            "chk_stats_effective_speed=on", "chk_stats_spin_rate=on", "chk_stats_release_pos_z=on", 
            "chk_stats_release_extension=on", "chk_stats_arm_angle=on"
        ])
        
        headers = {}  # Global session already has headers set
        append = i > 0 and success_count > 0
        
        if download_with_progress(
            url=url,
            headers=headers,
            csv_path=csv_path,
            desc=f"Downloading first inning pitchers data ({start_str} to {end_str})",
            append=append
        ):
            success_count += 1
        else:
            logger.warning(f"Failed to download first inning data for {start_str} to {end_str}")
    
    if success_count > 0:
        logger.info(f"First inning pitchers CSV updated: {csv_path} with {success_count} successful downloads")
    else:
        logger.error("Failed to download any first inning pitchers data")

def scrape_1st_inning_batters(is_daily=False):
    """
    Scrapes 1st-inning Batter data from Baseball Savant in incremental chunks
    and updates data/1stInningBattersHistorical.csv.
    
    Args:
        is_daily: If True, only download today's data.
    """
    os.makedirs(os.path.dirname(BATTERS_CSV), exist_ok=True) # Ensure directory exists
    csv_path = BATTERS_CSV # Use BATTERS_CSV from config
    date_ranges = get_date_ranges(csv_path, is_daily)
    
    if not date_ranges:
        logger.info("Batters data is already up to date.")
        return
    
    success_count = 0
    for i, (start_date, end_date) in enumerate(date_ranges):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        logger.info(f"Downloading batters data for range: {start_str} to {end_str}")
        
        url = (
            "https://baseballsavant.mlb.com/statcast_search/csv?"
            "hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&"
            f"hfSea={start_date.year}%7C&hfSit=&player_type=batter&"
            "hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&"
            f"game_date_gt={start_str}&game_date_lt={end_str}&"
            "hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&"
            "hfInn=1%7C&hfBBT=&hfFlag=is%5C.%5C.remove%5C.%5C.bunts%7C&"
            "metric_1=&group_by=name-date&min_pitches=0&min_results=0&min_pas=0&"
            "sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc"
        ) + "&" + "&".join([
            "chk_stats_pa=on", "chk_stats_abs=on", "chk_stats_bip=on", "chk_stats_hits=on",
            "chk_stats_singles=on", "chk_stats_dbls=on", "chk_stats_triples=on", "chk_stats_hrs=on",
            "chk_stats_so=on", "chk_stats_k_percent=on", "chk_stats_bb=on", "chk_stats_bb_percent=on",
            "chk_stats_whiffs=on", "chk_stats_swings=on", "chk_stats_api_break_z_with_gravity=on", 
            "chk_stats_api_break_x_batter_in=on", "chk_stats_delev_run_exp=on", 
            "chk_stats_delev_batter_run_value_per_100=on", "chk_stats_unadj_run_exp=on",
            "chk_stats_launch_speed=on", "chk_stats_hyper_speed=on", "chk_stats_launch_angle=on",
            "chk_stats_bbdist=on", "chk_stats_hardhit_percent=on", "chk_stats_barrels_per_bbe_percent=on",
            "chk_stats_barrels_per_pa_percent=on"
        ])
        
        headers = {}
        append = i > 0 and success_count > 0
        
        if download_with_progress(
            url=url,
            headers=headers,
            csv_path=csv_path,
            desc=f"Downloading batters data ({start_str} to {end_str})",
            append=append
        ):
            success_count += 1
        else:
            logger.warning(f"Failed to download data for {start_str} to {end_str}")
    
    if success_count > 0:
        logger.info(f"Batters CSV updated: {csv_path} with {success_count} successful downloads")
    else:
        logger.error("Failed to download any batters data")

def main():
    is_daily = False
    if len(sys.argv) > 1 and sys.argv[1] == "--daily":
        is_daily = True
        logger.info("Running daily update mode")
    
    # Run both scraping functions concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(scrape_1st_inning_pitchers, is_daily),
            executor.submit(scrape_1st_inning_batters, is_daily)
        ]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
