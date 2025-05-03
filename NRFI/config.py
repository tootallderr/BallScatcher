from datetime import datetime
import os
import logging

# Base directory (location of the config.py file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root directory (one level up from BASE_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Base configuration
START_SEASON = 2025
START_DATE = datetime(START_SEASON, 3, 29)
CURRENT_SEASON = datetime.now().year
CURRENT_DATE = datetime.now()

# Define explicit paths - now relative to project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Ensure DATA_DIR is an absolute path
if not os.path.isabs(DATA_DIR):
    DATA_DIR = os.path.abspath(DATA_DIR)

# Define subdirectories properly nested under data
NRFI_DIR = os.path.join(DATA_DIR, "First Inning NRFI")
VISUALS_DIR = os.path.join(NRFI_DIR, "visuals")  # visuals moved under NRFI
TEMP_DIR = os.path.join(NRFI_DIR, "temp_stuff")  # temp_stuff moved under NRFI
LOGS_DIR = os.path.join(NRFI_DIR, "logs")  # logs moved under NRFI

# Define data file paths
BATTERS_CSV = os.path.join(NRFI_DIR, "1stInningBattersHistorical.csv")
PITCHERS_CSV = os.path.join(NRFI_DIR, "1stInningPitchersHistorical.csv")

# Remove old directories if they exist in root
old_temp = os.path.join(PROJECT_ROOT, "temp_stuff")
old_visuals = os.path.join(PROJECT_ROOT, "visuals")
if os.path.exists(old_temp):
    import shutil
    shutil.rmtree(old_temp)
if os.path.exists(old_visuals):
    import shutil
    shutil.rmtree(old_visuals)

# Ensure all directories exist in correct location under NRFI
for directory in [DATA_DIR, NRFI_DIR, VISUALS_DIR, TEMP_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# MLB Season Dates
MLB_SEASON_DATES = {
    2000: {
        'regular_season': ('2000-04-03', '2000-10-01'),
        'postseason': ('2000-10-03', '2000-10-26')
    },
    2001: {
        'regular_season': ('2001-04-01', '2001-10-07'),
        'postseason': ('2001-10-09', '2001-11-04')
    },
    2002: {
        'regular_season': ('2002-03-31', '2002-09-29'),
        'postseason': ('2002-10-01', '2002-10-27')
    },
    2003: {
        'regular_season': ('2003-03-30', '2003-09-28'),
        'postseason': ('2003-09-30', '2003-10-25')
    },
    2004: {
        'regular_season': ('2004-04-04', '2004-10-03'),
        'postseason': ('2004-10-05', '2004-10-27')
    },
    2005: {
        'regular_season': ('2005-04-03', '2005-10-02'),
        'postseason': ('2005-10-04', '2005-10-26')
    },
    2006: {
        'regular_season': ('2006-04-02', '2006-10-01'),
        'postseason': ('2006-10-03', '2006-10-27')
    },
    2007: {
        'regular_season': ('2007-04-01', '2007-09-30'),
        'postseason': ('2007-10-02', '2007-10-28')
    },
    2008: {
        'regular_season': ('2008-03-25', '2008-09-30'),
        'postseason': ('2008-10-01', '2008-10-29')
    },
    2009: {
        'regular_season': ('2009-04-05', '2009-10-06'),
        'postseason': ('2009-10-07', '2009-11-04')
    },
    2010: {
        'regular_season': ('2010-04-04', '2010-10-03'),
        'postseason': ('2010-10-06', '2010-11-01')
    },
    2011: {
        'regular_season': ('2011-03-31', '2011-09-28'),
        'postseason': ('2011-09-30', '2011-10-28')
    },
    2012: {
        'regular_season': ('2012-03-28', '2012-10-03'),
        'postseason': ('2012-10-05', '2012-10-28')
    },
    2013: {
        'regular_season': ('2013-03-31', '2013-09-30'),
        'postseason': ('2013-10-01', '2013-10-30')
    },
    2014: {
        'regular_season': ('2014-03-22', '2014-09-28'),
        'postseason': ('2014-09-30', '2014-10-29')
    },
    2015: {
        'regular_season': ('2015-04-05', '2015-10-04'),
        'postseason': ('2015-10-06', '2015-11-01')
    },
    2016: {
        'regular_season': ('2016-04-03', '2016-10-02'),
        'postseason': ('2016-10-04', '2016-11-02')
    },
    2017: {
        'regular_season': ('2017-04-02', '2017-10-01'),
        'postseason': ('2017-10-03', '2017-11-01')
    },
    2018: {
        'regular_season': ('2018-03-29', '2018-10-01'),
        'postseason': ('2018-10-02', '2018-10-28')
    },
    2019: {
        'regular_season': ('2019-03-20', '2019-09-29'),
        'postseason': ('2019-10-01', '2019-10-30')
    },
    2020: {
        'regular_season': ('2020-07-23', '2020-09-27'),
        'postseason': ('2020-09-29', '2020-10-27')
    },
    2021: {
        'regular_season': ('2021-04-01', '2021-10-03'),
        'postseason': ('2021-10-05', '2021-11-02')
    },
    2022: {
        'regular_season': ('2022-04-07', '2022-10-05'),
        'postseason': ('2022-10-07', '2022-11-05')
    },
    2023: {
        'regular_season': ('2023-03-30', '2023-10-01'),
        'postseason': ('2023-10-03', '2023-11-04')
    },
    2024: {
        'regular_season': ('2024-03-28', '2024-09-29'),
        'postseason': ('2024-10-01', '2024-11-02')
    },
    2025: {
        'regular_season': ('2025-03-27', '2025-09-28'),
        'postseason': ('2025-09-30', '2025-11-01')
    }
}

# Data paths updated to use NRFI_DIR instead of DATA_DIR
PITCHERS_CSV = os.path.join(NRFI_DIR, "1stInningPitchersHistorical.csv")
BATTERS_CSV = os.path.join(NRFI_DIR, "1stInningBattersHistorical.csv")

# Baseball Savant configuration
BASEBALL_SAVANT_BASE_URL = "https://baseballsavant.mlb.com/statcast_search/csv?"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def setup_logging(script_name):
    """Setup logging configuration for scripts"""
    # Update logs directory to be under DATA_DIR
    logs_dir = os.path.join(NRFI_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log filename with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(logs_dir, f"{script_name}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(script_name)



