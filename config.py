"""
Global configuration system for the baseball analysis dashboard.
Defines the structure for different innings analysis modes.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Define specific directories from the project structure
NRFI_DIR = os.path.join(DATA_DIR, "First Inning NRFI")
ALL_INNINGS_DIR = os.path.join(DATA_DIR, "Full Game F9")
AOI_SPORTS_DIR = os.path.join(DATA_DIR, "AOI Sports")
PLAYER_PROPS_DIR = os.path.join(DATA_DIR, "Player_props")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(NRFI_DIR, exist_ok=True)
os.makedirs(ALL_INNINGS_DIR, exist_ok=True)
os.makedirs(AOI_SPORTS_DIR, exist_ok=True)
os.makedirs(PLAYER_PROPS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Create subdirectories for visualizations
os.makedirs(os.path.join(NRFI_DIR, "visuals", "teams"), exist_ok=True)
os.makedirs(os.path.join(NRFI_DIR, "visuals", "upcoming"), exist_ok=True)
os.makedirs(os.path.join(NRFI_DIR, "temp_stuff"), exist_ok=True)
os.makedirs(os.path.join(ALL_INNINGS_DIR, "charts", "full_game"), exist_ok=True)
os.makedirs(os.path.join(ALL_INNINGS_DIR, "charts", "first_5_innings"), exist_ok=True)
os.makedirs(os.path.join(ALL_INNINGS_DIR, "charts", "first_3_innings"), exist_ok=True)
os.makedirs(os.path.join(ALL_INNINGS_DIR, "charts", "combined_analysis"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "predictions"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "temp_stuff"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "visuals"), exist_ok=True)

# NRFI specific file paths
NRFI_BATTERS_FILE = os.path.join(NRFI_DIR, "1stInningBattersHistorical.csv")
NRFI_PITCHERS_FILE = os.path.join(NRFI_DIR, "1stInningPitchersHistorical.csv")
NRFI_PREDICTIONS_FILE = os.path.join(NRFI_DIR, "nrfi_predictions.csv")
NRFI_HISTORICAL_SCHEDULE = os.path.join(NRFI_DIR, "F1_historical_schedule.csv")
NRFI_CONFIDENCE_COMPARISON = os.path.join(NRFI_DIR, "F1_confidence_comparison.png")
NRFI_BACKTEST_PERFORMANCE = os.path.join(NRFI_DIR, "F1_backtest_performance.png")

# All Innings specific file paths
F3_PREDICTIONS_FILE = os.path.join(ALL_INNINGS_DIR, "F3_predictions.csv")
F5_PREDICTIONS_FILE = os.path.join(ALL_INNINGS_DIR, "F5_predictions.csv")
F9_PREDICTIONS_FILE = os.path.join(ALL_INNINGS_DIR, "F9_predictions.csv")
F9_HISTORICAL_SCHEDULE = os.path.join(ALL_INNINGS_DIR, "F9_historical_schedule.csv")
F9_DAILY_STARTERS = os.path.join(ALL_INNINGS_DIR, "F9_daily_starters.csv")

# Model files
MLB_MODEL_FILE = os.path.join(DATA_DIR, "models", "MLB_Money_trained_model.joblib")
NBA_MODEL_FILE = os.path.join(DATA_DIR, "models", "NBA_Money_trained_model.joblib")
EPL_MODEL_FILE = os.path.join(DATA_DIR, "models", "EPL_Money_trained_model.joblib")

# Predictions files
MLB_PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions", "MLB_upcoming_predictions.csv")
NBA_PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions", "NBA_upcoming_predictions.csv")
EPL_PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions", "EPL_upcoming_predictions.csv")

# Player props files
PROP_BETTING_OPPORTUNITIES = os.path.join(DATA_DIR, "prop_betting_opportunities.csv")
PROPS_FILE = os.path.join(DATA_DIR, "props.csv")
ANALYZED_PROPS_FILE = os.path.join(DATA_DIR, "analyzed_props.csv")

# Mode configurations
MODE_CONFIGS = {
    "NRFI": {
        "display_name": "No Runs First Inning Analysis",
        "subtitle": "Predict likelihood of scoreless first innings",
        "scripts_dir": os.path.join(BASE_DIR, "NRFI", "Scripts"),
        "data_dir": NRFI_DIR,
        "predictions_file": os.path.join(NRFI_DIR, "F1_predictions.csv"),
        "pipeline_steps": [
            {"name": "Step 1: Get MLB Schedule", "script": "Step1.py"},
            {"name": "Step 2: Download Game Data", "script": "Step2.py"},
            {"name": "Step 3: Process 1st Inning Stats", "script": "Step3.py"},
            {"name": "Step 4: Pre-Process Data", "script": "Pre-Process.py"},
            {"name": "Step 5: Generate NRFI Analysis", "script": "NRFI.py"},
            {"name": "Step 6: Generate Visualizations", "script": "graphs.py"}
        ]
    },
    "PROPS": {
        "display_name": "Player Props Analysis",
        "subtitle": "Analyze player performance metrics",
        "scripts_dir": os.path.join(BASE_DIR, "Player_props", "Scripts"),
        "data_dir": PLAYER_PROPS_DIR,
        "pipeline_steps": [
            {"name": "Step 1: Process Game Logs", "script": "game_logs.py"},
            {"name": "Step 2: Process Batters Data", "script": "Batters.py"},
            {"name": "Step 3: Process Pitchers Data", "script": "Pitchers.py"},
            {"name": "Step 4: Analyze Props", "script": "propsmodel.py"}
        ]
    },
    "ALL_INNINGS": {
        "display_name": "All Innings Analysis",
        "subtitle": "Analyze all innings (F3/F5/F9)",
        "scripts_dir": os.path.join(BASE_DIR, "All Innings"),
        "data_dir": ALL_INNINGS_DIR,
        "pipeline_steps": [
            {"name": "Step 1: Get Game Data", "script": "Step1.py"},
            {"name": "Step 2: Train & Predict (F3/F5/F9)", "script": "All_Innings_model.py"}
        ]
    },
    "AOI_SPORTS": {
        "display_name": "AOI Sports Analytics",
        "subtitle": "Comprehensive sports analytics and betting predictions",
        "scripts_dir": os.path.join(BASE_DIR, "AOI Sports", "TeamScrapers"),
        "data_dir": AOI_SPORTS_DIR,
        "pipeline_steps": [
            {"name": "Step 1: Get Upcoming Scoreboard", "script": "upcoming_scoreboard.py"},
            {"name": "Step 2: Process Historic Scoreboard", "script": "historic_scoreboard.py"},
            {"name": "Step 3: Generate Matchup Analysis", "script": "matchup.py"},
            {"name": "Step 4: Clean & Prepare Data", "script": "Cleaner.py"},
            {"name": "Step 5: Update Injuries Data", "script": "injuries.py"},
            {"name": "Step 6: Update Athletes Data", "script": "athletes.py"},
            {"name": "Step 7: Run Prediction Models", "script": os.path.join("..", "Model", "Model", "AOI_Moneyline.py")}
        ]
    }
}

def get_mode_config(mode_code):
    """
    Get configuration for the specified analysis mode.
    
    Args:
        mode_code (str): The mode code (NRFI, F3, F5, PROPS)
    
    Returns:
        dict: Configuration dictionary for the mode
    """
    return MODE_CONFIGS.get(mode_code, MODE_CONFIGS["NRFI"])

def get_file_path(mode_code, file_key):
    """
    Get the full path to a file for the specified mode.
    
    Args:
        mode_code (str): The mode code (NRFI, F3, F5, PROPS)
        file_key (str): The file key in the mode config
    
    Returns:
        str: Full path to the file
    """
    config = get_mode_config(mode_code)
    file_name = config.get(file_key, "")
    return os.path.join(config["data_dir"], file_name)

def get_mode_paths(mode_code):
    """
    Get all important paths for the specified mode.
    
    Args:
        mode_code (str): The mode code (NRFI, F3, F5, PROPS)
    
    Returns:
        dict: Dictionary of paths for the mode
    """
    config = get_mode_config(mode_code)
    return {
        "data_dir": config["data_dir"],
        "scripts_dir": config["scripts_dir"],
        "teams_dir": os.path.join(config["data_dir"], "visuals", "teams"),
        "upcoming_dir": os.path.join(config["data_dir"], "visuals", "upcoming")
    }

def get_fallback_paths(mode_code):
    """
    Get fallback paths for the specified mode files in the root data directory.
    
    Args:
        mode_code (str): The mode code (NRFI, F3, F5, PROPS)
    
    Returns:
        dict: Dictionary of fallback paths for the mode
    """
    config = get_mode_config(mode_code)
    prefix = config.get("prefix", "").lower()
    return {
        "predictions_file": os.path.join(DATA_DIR, f"{prefix}predictions.csv"),
        "historical_file": os.path.join(DATA_DIR, f"{prefix}historical_data.csv"),
        "daily_starters_file": os.path.join(DATA_DIR, f"{prefix}daily_starters.csv"),
    }

def get_visuals_dirs(mode_code):
    """
    Get directories for visualizations for the specified mode.
    
    Args:
        mode_code (str): The mode code (NRFI, F3, F5, PROPS)
    
    Returns:
        dict: Dictionary of visual directories for the mode, returns empty dict if not configured.
    """
    config = get_mode_config(mode_code)
    base_dir = os.path.join(config["data_dir"], "visuals")
    return {
        "base_dir": base_dir,
        "teams_dir": os.path.join(base_dir, "teams"),
        "upcoming_dir": os.path.join(base_dir, "upcoming")
    }

# Save files to the data directory
def save_progress(data, filename_prefix="mlb_game_logs"):
    file_path = os.path.join(DATA_DIR, f"{filename_prefix}_partial.csv")
    data.to_csv(file_path, index=False)
