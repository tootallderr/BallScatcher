# NRFI/YRFI Trend Analysis Script with Advanced Visualizations
import os
import sys
import calendar
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

# Suppress matplotlib category and cmdstanpy debug warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['CMDSTAN_QUIET'] = '1'  # Suppress cmdstanpy debug messages

# Add parent directory to Python path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (CURRENT_SEASON, DATA_DIR, NRFI_DIR, MLB_SEASON_DATES, TEMP_DIR, VISUALS_DIR, setup_logging) # Added NRFI_DIR

# Setup logging using centralized configuration
logger = setup_logging('graphs')

# Ensure visuals directory exists
os.makedirs(VISUALS_DIR, exist_ok=True)

# Update file paths to use NRFI_DIR instead of DATA_DIR
HISTORICAL_FILE = os.path.join(NRFI_DIR, "F1_historical_schedule.csv") # Use NRFI_DIR
DAILY_STARTERS_FILE = os.path.join(NRFI_DIR, "F1_daily_starters.csv") # Use NRFI_DIR
RANKINGS_FILE = os.path.join(NRFI_DIR, "F1_NRFI_Rankings.csv") # Use NRFI_DIR
LEAGUE_STATS_FILE = os.path.join(NRFI_DIR, "F1_League_Baseline_Stats.csv") # Use NRFI_DIR

# Add new style constants after imports
GRAPH_COLORS = {
    'bg_dark': "#000000",           # Main background
    'bg_content': "#1e1e1e",        # Content areas
    'accent_primary': "#2962ff",    # Primary accent for main lines
    'accent_secondary': "#0097a7",  # Secondary accent for additional lines
    'accent_success': "#2ecc71",    # Success indicators (e.g., NRFI)
    'accent_warning': "#f39c12",    # Warning indicators
    'accent_error': "#e74c3c",      # Error indicators (e.g., YRFI)
    'text_primary': "#ffffff",      # Primary text
    'text_secondary': "#b3b3b3",    # Secondary text
    'grid': "#333333",             # Grid lines
    'trend_positive': "#4CAF50",   # Positive trends
    'trend_negative': "#f44336",   # Negative trends
}

GRAPH_STYLE = {
    'figure.facecolor': GRAPH_COLORS['bg_dark'],
    'axes.facecolor': GRAPH_COLORS['bg_content'],
    'axes.edgecolor': GRAPH_COLORS['grid'],
    'axes.labelcolor': GRAPH_COLORS['text_primary'],
    'axes.grid': True,
    'grid.color': GRAPH_COLORS['grid'],
    'grid.alpha': 0.3,
    'xtick.color': GRAPH_COLORS['text_primary'],
    'ytick.color': GRAPH_COLORS['text_primary'],
    'text.color': GRAPH_COLORS['text_primary'],
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'font.family': 'Arial',
}

# Set default style
plt.style.use('dark_background')
for key, value in GRAPH_STYLE.items():
    plt.rcParams[key] = value

# Custom color maps for various visualizations
NRFI_CMAP = LinearSegmentedColormap.from_list('nrfi_cmap', 
    [GRAPH_COLORS['accent_error'], GRAPH_COLORS['accent_success']], N=2)
    
TREND_CMAP = LinearSegmentedColormap.from_list('trend_cmap',
    [GRAPH_COLORS['trend_negative'], '#fee090', GRAPH_COLORS['trend_positive']])

def create_directories():
    """Create necessary directories"""
    # Use configured directory paths
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Create subdirectories under VISUALS_DIR
    teams_dir = os.path.join(VISUALS_DIR, 'teams')
    upcoming_dir = os.path.join(VISUALS_DIR, 'upcoming')
    os.makedirs(teams_dir, exist_ok=True)
    os.makedirs(upcoming_dir, exist_ok=True)
    
    return DATA_DIR, teams_dir, upcoming_dir

def load_and_prepare_data(include_upcoming=False):
    """Load and prepare the data for analysis"""
    try:
        # Load historical data using DATA_DIR
        if not os.path.exists(HISTORICAL_FILE):
            logger.error(f"Historical data file not found: {HISTORICAL_FILE}")
            raise FileNotFoundError(f"Historical data file not found: {HISTORICAL_FILE}")

        df = pd.read_csv(HISTORICAL_FILE)
        logger.info(f"Loaded {len(df)} rows from historical data")

        # Normalize date column and ensure numeric types
        df['date'] = pd.to_datetime(df['date'])

        # Only filter out future games
        current_date = datetime.now()
        df = df[df['date'] <= current_date].copy()

        # Filter completed games and calculate NRFI
        df = df[df['status'] == 'Final'].copy()
        logger.info(f"Found {len(df)} completed games")

        if 'home_inning_1_runs' not in df.columns or 'away_inning_1_runs' not in df.columns:
            logger.error("Required columns for NRFI calculation missing")
            raise ValueError("Missing required columns: home_inning_1_runs and/or away_inning_1_runs")

        df['NRFI'] = ((df['home_inning_1_runs'] == 0) & (df['away_inning_1_runs'] == 0)).astype(int)

        # Log unique teams for validation
        home_teams = set(df['home_team'].unique())
        away_teams = set(df['away_team'].unique())
        all_teams = home_teams.union(away_teams)
        logger.info(f"Found {len(all_teams)} unique teams:")
        logger.info("\n".join(sorted(all_teams)))

        if include_upcoming:
            if not os.path.exists(DAILY_STARTERS_FILE):
                logger.warning(f"Upcoming games file not found: {DAILY_STARTERS_FILE}")
                return df, pd.DataFrame()

            upcoming_df = pd.read_csv(DAILY_STARTERS_FILE)
            upcoming_df['date'] = pd.to_datetime(upcoming_df['date'])
            return df, upcoming_df

        return df

    except Exception as e:
        logger.error(f"Error in load_and_prepare_data: {str(e)}")
        raise

def calculate_league_stats(df):
    """Calculate league-wide NRFI statistics"""
    return {
        'nrfi_percentage': round(df['NRFI'].mean() * 100, 2),
        'total_games': len(df),
        'home_nrfi_percentage': round((df['NRFI'].groupby(df['home_team'], observed=True).mean() * 100).mean(), 2),
        'away_nrfi_percentage': round((df['NRFI'].groupby(df['away_team'], observed=True).mean() * 100).mean(), 2),
        'last_month_percentage': round(df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]['NRFI'].mean() * 100, 2)
    }

# -------------------------------
# Original Visualization Functions
# -------------------------------
def create_team_nrfi_trend(df, team_name, output_dir, league_stats):
    """Create basic NRFI trend analysis for a single team"""
    # Get team's games (both home and away)
    team_games = pd.concat([df[df['home_team'] == team_name][['date', 'NRFI']],
                            df[df['away_team'] == team_name][['date', 'NRFI']]]).sort_values('date')

    # Calculate team's overall NRFI percentage
    team_nrfi_pct = (team_games['NRFI'].mean() * 100).round(2)

    # Get the last 20 games
    last_20_games = team_games.tail(20).reset_index(drop=True)

    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.plot(last_20_games['date'], last_20_games['NRFI'].map({1: 100, 0: 0}),
             marker='o', label='NRFI Outcome (Yes=100%, No=0%)', color='purple', linewidth=2)
    plt.axhline(y=team_nrfi_pct, color='orange', linestyle='--',
                label=f'Team Season Average ({team_nrfi_pct:.1f}%)')
    plt.axhline(y=league_stats['nrfi_percentage'], color='red', linestyle='--',
                label=f'League Average ({league_stats["nrfi_percentage"]:.1f}%)')
    plt.title(f'{team_name} NRFI Trend Analysis\n(Last 20 Games by Date)', fontsize=12, pad=15)
    plt.xlabel('Date')
    plt.ylabel('NRFI Outcome')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.gcf().autofmt_xdate()
    recent_nrfi = last_20_games['NRFI'].mean() * 100
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    text_box = (f'Recent 20 Games: {recent_nrfi:.1f}%\n'
                f'Season Average: {team_nrfi_pct:.1f}%\n'
                f'League Average: {league_stats["nrfi_percentage"]:.1f}%\n'
                f'Total Games: {len(team_games)}')
    plt.text(1.25, 0.2, text_box, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'F1_{team_name.replace(" ", "_")}_NRFI_Analysis.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    return {'team': team_name, 'recent_20_nrfi': recent_nrfi, 'season_nrfi': team_nrfi_pct, 'total_games': len(team_games)}

# -------------------------------
# Enhanced Visualizations & Advanced Analytics
# -------------------------------
def create_team_nrfi_trend_improved(df, team_name, output_dir, league_stats):
    """
    Enhanced NRFI trend analysis:
      - Overlays multiple moving averages (3, 5, and 10-game)
      - Adds a degree-2 polynomial trendline
    """
    team_games = pd.concat([df[df['home_team'] == team_name][['date', 'NRFI']],
                            df[df['away_team'] == team_name][['date', 'NRFI']]]).sort_values('date').reset_index(drop=True)

    team_nrfi_pct = (team_games['NRFI'].mean() * 100).round(2)
    last_20 = team_games.tail(20).reset_index(drop=True)

    # Compute moving averages (scaled to percentage)
    last_20['MA_3'] = last_20['NRFI'].rolling(window=3, min_periods=1).mean() * 100
    last_20['MA_5'] = last_20['NRFI'].rolling(window=5, min_periods=1).mean() * 100
    last_20['MA_10'] = last_20['NRFI'].rolling(window=10, min_periods=1).mean() * 100

    # Fit a polynomial trendline (degree 2)
    x_vals = last_20['date'].map(datetime.toordinal).values
    y_vals = last_20['NRFI'].map({1: 100, 0: 0}).values
    poly_coef = np.polyfit(x_vals, y_vals, 2)
    poly_func = np.poly1d(poly_coef)
    trendline = poly_func(x_vals)

    plt.figure(figsize=(12, 6))
    plt.plot(last_20['date'], last_20['NRFI'].map({1: 100, 0: 0}),
             marker='o', label='NRFI Outcome', color='purple', linewidth=2)
    plt.plot(last_20['date'], last_20['MA_3'], label='3-Game MA', linestyle='--', color='green', linewidth=1.5)
    plt.plot(last_20['date'], last_20['MA_5'], label='5-Game MA', linestyle='--', color='blue', linewidth=1.5)
    plt.plot(last_20['date'], last_20['MA_10'], label='10-Game MA', linestyle='--', color='orange', linewidth=1.5)
    plt.plot(last_20['date'], trendline, label='Polynomial Trendline', linestyle='-', color='red', linewidth=2)
    plt.axhline(y=team_nrfi_pct, color='gray', linestyle='--', label=f'Team Season Avg ({team_nrfi_pct:.1f}%)')
    plt.axhline(y=league_stats['nrfi_percentage'], color='black', linestyle='--', label=f'League Avg ({league_stats["nrfi_percentage"]:.1f}%)')
    plt.title(f'{team_name} Enhanced NRFI Trend Analysis (Last 20 Games)', fontsize=12, pad=15)
    plt.xlabel('Date')
    plt.ylabel('NRFI Outcome (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'F1_{team_name.replace(" ", "_")}_NRFI_Trend_Enhanced.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def forecast_nrfi_trend(df, team_name, output_dir, periods=10):
    """Forecast future NRFI probabilities with 10-game pre-forecast window"""
    # Get team's games (both home and away)
    team_games = pd.concat([df[df['home_team'] == team_name][['date', 'NRFI']],
                          df[df['away_team'] == team_name][['date', 'NRFI']]]).sort_values('date')

    # Get current date and last 10 completed games before today
    current_date = datetime.now()
    pre_forecast_games = team_games[team_games['date'] < current_date].tail(10).copy()
    
    # Prepare data for Prophet
    ts = pre_forecast_games[['date', 'NRFI']].rename(columns={'date': 'ds', 'NRFI': 'y'})
    ts['y'] = ts['y'] * 100  # Scale binary outcomes to percentages

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(ts)
    
    # Create future dataframe for next 10 games
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)

    # Create visualization
    fig = plt.figure(figsize=(12, 6))
    
    # Plot pre-forecast window (last 10 games)
    plt.plot(pre_forecast_games['date'], pre_forecast_games['NRFI'] * 100, 
             'ko-', alpha=0.6, label='Pre-forecast Window', markersize=6)
    
    # Plot forecasted values with confidence intervals
    forecast_dates = forecast['ds'].tail(periods)
    forecast_values = forecast['yhat'].tail(periods)
    forecast_lower = forecast['yhat_lower'].tail(periods)
    forecast_upper = forecast['yhat_upper'].tail(periods)
    
    plt.plot(forecast_dates, forecast_values, 
             'b-', label='Forecasted NRFI %', linewidth=2)
    plt.fill_between(forecast_dates,
                     forecast_lower,
                     forecast_upper,
                     color='blue', alpha=0.1, label='95% Confidence Interval')

    plt.title(f'{team_name} NRFI Forecast\nPre-forecast Window (10 games) + Next {periods} Games', 
             fontsize=14, pad=20)
    plt.xlabel('Date')
    plt.ylabel('NRFI Probability (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.gcf().autofmt_xdate()
    plt.ylim(0, 100)

    # Add summary statistics
    pre_forecast_mean = pre_forecast_games['NRFI'].mean() * 100
    forecasted_avg = forecast_values.mean()
    text_box = (f'Pre-forecast Window Avg: {pre_forecast_mean:.1f}%\n'
                f'Forecasted Avg: {forecasted_avg:.1f}%\n'
                f'Forecast Range: {forecast_lower.mean():.1f}% - '
                f'{forecast_upper.mean():.1f}%')
    plt.text(1.02, 0.2, text_box, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'F1_{team_name.replace(" ", "_")}_NRFI_Forecast.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return forecast

def create_nrfi_radar_chart(team_name, metrics, values, output_dir):
    """
    Create a radar (spider) chart for a team's NRFI profile using current season data only.
    `metrics` is a list of metric names and `values` is a list of percentages.
    """
    # Ensure we're working with current season data only
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    # Complete the loop for radar chart
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.5)
    ax.plot(angles, values, color='skyblue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    ax.set_title(f'{team_name} NRFI Radar Profile (2025 Season)', size=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'F1_{team_name.replace(" ", "_")}_Radar_Chart.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def create_violin_box_plots(df, team_name, output_dir):
    """
    Create violin and box plots to visualize NRFI outcome distributions by day of week
    for current season only.
    """
    # Filter for current season only
    current_date = datetime.now()
    current_year = current_date.year
    season_start = pd.to_datetime(MLB_SEASON_DATES[current_year]['regular_season'][0])
    
    # Filter data for current season
    current_season_df = df[df['date'] >= season_start].copy()
    
    team_games = pd.concat([
        current_season_df[current_season_df['home_team'] == team_name][['date', 'NRFI']],
        current_season_df[current_season_df['away_team'] == team_name][['date', 'NRFI']]
    ]).sort_values('date')

    # Ensure 'date' is in datetime format
    team_games['date'] = pd.to_datetime(team_games['date'], errors='coerce')
    team_games = team_games.dropna(subset=['date'])  # Drop rows with invalid dates

    team_games['day_of_week'] = team_games['date'].dt.day_name()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='day_of_week', y='NRFI', data=team_games, inner="box",
                   order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title(f'{team_name} NRFI Distribution by Day of Week (2025 Season)')
    plt.ylabel('NRFI (0 = Runs, 1 = No Runs)')
    plt.xlabel('Day of Week')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'F1_{team_name.replace(" ", "_")}_Violin_Box.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def create_sankey_diagram(output_dir):
    """
    Create a sample Sankey diagram using Plotly to visualize relationships among pitchers, teams, and outcomes.
    Replace the sample data with your actual dataset as needed.
    """
    labels = ["Pitcher A", "Pitcher B", "Team X", "Team Y", "NRFI", "YRFI"]
    source = [0, 1, 2, 3]
    target = [2, 3, 4, 5]
    values = [10, 5, 12, 3]

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels),
        link=dict(source=source, target=target, value=values)
    )])
    fig.update_layout(title_text="Pitcher-Team NRFI Sankey Diagram", font_size=10)
    sankey_file = os.path.join(output_dir, 'F1_NRFI_Sankey_Diagram.html')
    fig.write_html(sankey_file)

def create_animated_nrfi_trend(df, output_dir):
    """
    Create an animated bar chart showing NRFI trends over time for each team using Plotly Express.
    This example aggregates NRFI percentages per team per date.
    """
    df_long = df.copy()
    # For simplicity, assign home_team as the team; you can adjust to merge home and away data.
    df_long['team'] = df_long['home_team']
    df_grouped = df_long.groupby(['date', 'team'])['NRFI'].mean().reset_index()
    df_grouped['NRFI_pct'] = df_grouped['NRFI'] * 100

    fig = px.bar(df_grouped, x="team", y="NRFI_pct", animation_frame=df_grouped['date'].dt.strftime('%Y-%m-%d'),
                 range_y=[0, 100], title="NRFI Trends Over Time")
    animated_file = os.path.join(output_dir, 'F1_NRFI_Animated_Trend.html')
    fig.write_html(animated_file)

def create_hexbin_density_plot(df, output_dir):
    """
    Create a hexbin/density plot to analyze the relationship between temperature and NRFI probability.
    This function assumes a 'temperature' column exists in the dataframe.
    """
    if 'temperature' not in df.columns:
        print("No temperature data available for hexbin plot.")
        return

    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(df['temperature'], df['NRFI'] * 100, gridsize=20, cmap='coolwarm', reduce_C_function=np.mean)
    plt.colorbar(hb, label='NRFI Probability (%)')
    plt.xlabel('Temperature (°F)')
    plt.ylabel('NRFI Probability (%)')
    plt.title('NRFI Probability vs Temperature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'F1_NRFI_Hexbin_Density.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def run_interactive_dashboard():
    """
    Placeholder for an interactive dashboard implementation.
    You can build an interactive dashboard using Streamlit or Plotly Dash.
    For example, with Streamlit:
    
        import streamlit as st
        st.title("NRFI Interactive Dashboard")
        st.plotly_chart(fig)
    
    This function is a stub.
    """
    print("Interactive dashboard functionality is not implemented here but can be built using Streamlit or Dash.")

# -------------------------------
# Other Existing Visualization Functions
# -------------------------------
def create_nrfi_calendar_heatmap(df, team_name, output_dir):
    """Create a calendar heatmap showing NRFI outcomes for current and previous month only"""
    # Get team's games
    team_games = pd.concat([
        df[df['home_team'] == team_name][['date', 'NRFI', 'away_team']].assign(opponent=lambda x: x['away_team'], home_away='Home'),
        df[df['away_team'] == team_name][['date', 'NRFI', 'home_team']].assign(opponent=lambda x: x['home_team'], home_away='Away')
    ]).sort_values('date')
    
    team_games = team_games.drop(['away_team', 'home_team'], axis=1, errors='ignore')
    team_games['date'] = pd.to_datetime(team_games['date'], errors='coerce')
    
    # Filter for current and previous month only
    current_date = datetime.now()
    last_month_start = (current_date.replace(day=1) - timedelta(days=1)).replace(day=1)
    
    team_games = team_games[
        (team_games['date'] >= last_month_start) & 
        (team_games['date'] <= current_date)
    ]
    
    team_games['month'] = team_games['date'].dt.month
    team_games['day'] = team_games['date'].dt.day
    team_games['year'] = team_games['date'].dt.year

    fig, ax = plt.subplots(figsize=(14, 8))
    months = sorted(team_games['month'].unique())
    years = sorted(team_games['year'].unique())
    cmap = LinearSegmentedColormap.from_list('nrfi_cmap', ['#FF6B6B', '#4CAF50'], N=2)
    x_offset = 0

    for year in years:
        for month in months:
            month_data = team_games[(team_games['month'] == month) & (team_games['year'] == year)]
            if len(month_data) == 0:
                continue
                
            _, last_day = calendar.monthrange(year, month)
            month_grid = np.full((6, 7), np.nan)
            first_day = datetime(year, month, 1).weekday()
            first_day = (first_day + 1) % 7  # Adjust to make Sunday = 0
            
            for _, game in month_data.iterrows():
                day = game['day']
                row = (first_day + day - 1) // 7
                col = (first_day + day - 1) % 7
                month_grid[row, col] = game['NRFI']
                
            month_ax = fig.add_subplot(1, len(months) * len(years), x_offset + 1)
            sns.heatmap(month_grid, cmap=cmap, cbar=False, linewidths=1, linecolor='white',
                       mask=np.isnan(month_grid), vmin=0, vmax=1, ax=month_ax)
            
            month_name = calendar.month_name[month]
            month_ax.set_title(f"{month_name} {year}", fontsize=12)
            month_ax.set_xticks(np.arange(7) + 0.5)
            month_ax.set_xticklabels(['S', 'M', 'T', 'W', 'T', 'F', 'S'])
            month_ax.set_yticks(np.arange(6) + 0.5)
            month_ax.set_yticklabels(['W1', 'W2', 'W3', 'W4', 'W5', 'W6'])
            month_ax.set_ylabel('')
            x_offset += 1

    plt.suptitle(f'{team_name} NRFI Calendar - Last Two Months\nGreen = NRFI (No Runs), Red = YRFI (Runs Scored)', 
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'F1_{team_name.replace(" ", "_")}_NRFI_Calendar.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return team_games

def create_nrfi_moving_averages(df, team_name, output_dir, league_stats):
    """Create enhanced moving averages chart for NRFI outcomes with improved visualization"""
    # Convert date column to datetime if not already
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate NRFI column - True when both teams score 0 runs in first inning
    df['nrfi'] = ((df['home_inning_1_runs'] == 0) & (df['away_inning_1_runs'] == 0)).astype(int)
    
    # Get team's games - both home and away
    team_games = pd.concat([
        df[df['home_team'] == team_name][['date', 'nrfi']],
        df[df['away_team'] == team_name][['date', 'nrfi']]
    ]).sort_values('date')
    
    if len(team_games) < 5:  # Minimum games needed for smallest MA
        return None

    # Get the most recent 30 games that have been completed
    current_date = datetime.now()
    completed_games = team_games[team_games['date'] <= current_date]
    
    # Get last 30 games regardless of season boundaries
    team_games = completed_games.tail(30).copy()
    team_games = team_games.sort_values('date')

    # Calculate sequential moving averages correctly
    windows = [5, 10, 20]
    for window in windows:
        col_name = f'MA_{window}'
        # Use min_periods=1 to start calculating as soon as we have at least 1 game
        team_games[col_name] = team_games['nrfi'].rolling(window=window, min_periods=1).mean() * 100
        
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Set the style using seaborn
    sns.set_style("darkgrid")

    # Plot individual game results as small markers first (so they're in the background)
    plt.scatter(team_games['date'], team_games['nrfi'] * 100, 
               color='gray', alpha=0.3, s=30, label='Game Results')

    # Plot moving averages with enhanced styling and correct order
    plt.plot(team_games['date'], team_games['MA_5'], 
            label='5-Game MA', color='#FF9999', linewidth=2, alpha=0.9)
    plt.plot(team_games['date'], team_games['MA_10'], 
            label='10-Game MA', color='#66B2FF', linewidth=2, alpha=0.9)
    plt.plot(team_games['date'], team_games['MA_20'], 
            label='20-Game MA', color='#99FF99', linewidth=2.5, alpha=0.9)

    # Add league average reference line
    plt.axhline(y=league_stats['nrfi_percentage'], 
                color='red', linestyle='--', alpha=0.5,
                label=f"League Avg ({league_stats['nrfi_percentage']:.1f}%)")

    # Customize axes
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Show date every 5 days for better readability
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.xticks(rotation=45)
    plt.ylim(0, 100)

    # Add labels and title
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('NRFI Success Rate (%)', fontsize=10)
    plt.title(f'{team_name} NRFI Trends - Rolling Averages\n'
             f'Last {len(team_games)} Games', fontsize=12, pad=15)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Calculate and add current averages to legend (using only the available data points)
    current_5ma = team_games['MA_5'].iloc[-1] if not team_games['MA_5'].empty else 0
    current_10ma = team_games['MA_10'].iloc[-1] if not team_games['MA_10'].empty else 0
    current_20ma = team_games['MA_20'].iloc[-1] if not team_games['MA_20'].empty else 0

    # Add a text box with the actual number of games used in each MA
    recent_games = len(team_games)
    games_5 = min(5, recent_games)
    games_10 = min(10, recent_games)
    games_20 = min(20, recent_games)
    
    text_box = (f'Current Values (actual games used):\n'
                f'5-Game MA: {current_5ma:.1f}% ({games_5} games)\n'
                f'10-Game MA: {current_10ma:.1f}% ({games_10} games)\n'
                f'20-Game MA: {current_20ma:.1f}% ({games_20} games)\n'
                f'League Avg: {league_stats["nrfi_percentage"]:.1f}%')
    
    plt.text(1.02, 0.5, text_box, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Adjust layout to prevent text cutoff
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, f'F1_{team_name.replace(" ", "_")}_NRFI_Moving_Averages.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return team_games

def get_league_stats(df):
    """Calculate league-wide NRFI statistics"""
    return {
        'nrfi_percentage': df['nrfi'].mean() * 100,
        'total_games': len(df)
    }

def create_weekly_performance_chart(df, team_name, output_dir):
    """Create a chart showing NRFI performance by day of week for current season only"""
    # Filter for current season only
    current_date = datetime.now()
    current_year = current_date.year
    season_start = pd.to_datetime(MLB_SEASON_DATES[current_year]['regular_season'][0])
    
    # Filter data for current season
    current_season_df = df[df['date'] >= season_start].copy()
    
    team_games = pd.concat([
        current_season_df[current_season_df['home_team'] == team_name][['date', 'NRFI']],
        current_season_df[current_season_df['away_team'] == team_name][['date', 'NRFI']]
    ]).sort_values('date')

    # Ensure 'date' is in datetime format
    team_games['date'] = pd.to_datetime(team_games['date'], errors='coerce')
    team_games = team_games.dropna(subset=['date'])  # Drop rows with invalid dates

    team_games['day_of_week'] = team_games['date'].dt.day_name()
    day_performance = team_games.groupby('day_of_week')['NRFI'].agg(['mean', 'count']).reset_index()
    day_performance['mean'] = day_performance['mean'] * 100
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_performance = day_performance[day_performance['count'] >= 1]  # Changed minimum games requirement to 1 for new season

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='day_of_week', y='mean', hue='day_of_week', data=day_performance,
                     order=[d for d in days_order if d in day_performance['day_of_week'].values],
                     legend=False)
    
    for i, p in enumerate(ax.patches):
        day = day_performance.iloc[i]
        count = int(day['count'])
        percentage = day['mean']
        ax.annotate(f'{percentage:.1f}%\n({count} games)', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    
    plt.title(f'{team_name} NRFI Performance by Day of Week (2025 Season)', fontsize=14, pad=20)
    plt.xlabel('Day of Week')
    plt.ylabel('NRFI Percentage')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    overall_nrfi = team_games['NRFI'].mean() * 100
    plt.axhline(y=overall_nrfi, color='red', linestyle='--', 
                label=f'Season Average: {overall_nrfi:.1f}%')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'F1_{team_name.replace(" ", "_")}_NRFI_Weekly.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return day_performance

def create_home_away_comparison(df, team_name, output_dir, league_stats):
    """Create a comparison of NRFI performance at home vs away for current season only"""
    # Filter for current season only
    current_date = datetime.now()
    current_year = current_date.year
    season_start = pd.to_datetime(MLB_SEASON_DATES[current_year]['regular_season'][0])
    
    # Filter data for current season
    current_season_df = df[df['date'] >= season_start].copy()
    
    home_games = current_season_df[current_season_df['home_team'] == team_name][['date', 'NRFI']]
    away_games = current_season_df[current_season_df['away_team'] == team_name][['date', 'NRFI']]
    
    # Calculate season stats
    home_nrfi_pct = (home_games['NRFI'].mean() * 100).round(2) if len(home_games) > 0 else 0
    away_nrfi_pct = (away_games['NRFI'].mean() * 100).round(2) if len(away_games) > 0 else 0
    overall_nrfi_pct = ((home_games['NRFI'].sum() + away_games['NRFI'].sum()) / 
                        (len(home_games) + len(away_games)) * 100).round(2)

    # Get last 10 games stats for both home and away
    recent_home = home_games.tail(10)['NRFI'].mean() * 100 if len(home_games) > 0 else float('nan')
    recent_away = away_games.tail(10)['NRFI'].mean() * 100 if len(away_games) > 0 else float('nan')

    data = pd.DataFrame({
        'Location': ['Home', 'Away', 'Overall'],
        'NRFI Percentage': [home_nrfi_pct, away_nrfi_pct, overall_nrfi_pct],
        'Games': [len(home_games), len(away_games), len(home_games) + len(away_games)]
    })

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Location', y='NRFI Percentage', hue='Location', data=data, legend=False)
    
    for i, p in enumerate(ax.patches):
        location_data = data.iloc[i]
        count = int(location_data['Games'])
        percentage = location_data['NRFI Percentage']
        ax.annotate(f'{percentage:.1f}%\n({count} games)', 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                    ha='center', va='bottom', fontsize=11)

    plt.axhline(y=league_stats['nrfi_percentage'], color='red', linestyle='--',
                label=f"League Average ({league_stats['nrfi_percentage']:.1f}%)")
    
    plt.title(f'{team_name} NRFI Performance: Home vs Away (2025 Season)', fontsize=14, pad=20)
    plt.xlabel('')
    plt.ylabel('NRFI Percentage')
    plt.ylim(0, max(100, data['NRFI Percentage'].max() + 10))
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()

    text_box = (f'Last 10 Home Games: {recent_home:.1f}%\n' if not np.isnan(recent_home) else 'Not enough home data\n'
                f'Last 10 Away Games: {recent_away:.1f}%\n' if not np.isnan(recent_away) else 'Not enough away data\n'
                f'League Average: {league_stats["nrfi_percentage"]:.1f}%')
    
    plt.text(0.02, 0.05, text_box, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'F1_{team_name.replace(" ", "_")}_Home_Away_Comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return data

def create_league_comparison_dashboard(team_stats_df, output_dir, league_stats):
    """Create a league-wide NRFI comparison dashboard"""
    sorted_teams = team_stats_df.sort_values('recent_20_nrfi', ascending=False)
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(x='recent_20_nrfi', y='team', hue='team', data=sorted_teams, legend=False)
    plt.axvline(x=league_stats['nrfi_percentage'], color='red', linestyle='--',
                label=f"League Average ({league_stats['nrfi_percentage']:.1f}%)")
    plt.title('MLB Teams NRFI Performance (Last 20 Games)', fontsize=16, pad=20)
    plt.xlabel('NRFI Percentage')
    plt.ylabel('')
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.3, axis='x')
    for i, p in enumerate(ax.patches):
        percentage = sorted_teams.iloc[i]['recent_20_nrfi']
        ax.annotate(f'{percentage:.1f}%', 
                    (p.get_width() + 1, p.get_y() + p.get_height() / 2),
                    va='center')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'F1_MLB_Teams_NRFI_Comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    return sorted_teams

def create_matchup_comparison(historical_df, upcoming_game, output_dir):
    """Create comparison visualizations for an upcoming matchup"""
    home_team = upcoming_game['home_team']
    away_team = upcoming_game['away_team']
    game_date = upcoming_game['date']
    matchup_dir = os.path.join(output_dir, f"F1_{game_date.strftime('%Y-%m-%d')}_{away_team}_at_{home_team}")
    os.makedirs(matchup_dir, exist_ok=True)
    home_games = pd.concat([historical_df[historical_df['home_team'] == home_team],
                            historical_df[historical_df['away_team'] == home_team]]).sort_values('date')
    away_games = pd.concat([historical_df[historical_df['home_team'] == away_team],
                            historical_df[historical_df['away_team'] == away_team]]).sort_values('date')
    home_stats = {
        'team': home_team,
        'nrfi_percentage': round(home_games['NRFI'].mean() * 100, 2),
        'total_games': len(home_games),
        'recent_20_nrfi': round(home_games.tail(20)['NRFI'].mean() * 100, 2)
    }
    away_stats = {
        'team': away_team,
        'nrfi_percentage': round(away_games['NRFI'].mean() * 100, 2),
        'total_games': len(away_games),
        'recent_20_nrfi': round(away_games.tail(20)['NRFI'].mean() * 100, 2)
    }
    create_head_to_head_comparison(home_stats, away_stats, matchup_dir, game_date)
    create_matchup_trends(home_games, away_games, home_team, away_team, matchup_dir)
    venue_name = upcoming_game['venue_name']
    create_venue_analysis(historical_df, venue_name, matchup_dir)
    return matchup_dir

def create_head_to_head_comparison(home_stats, away_stats, output_dir, game_date):
    """Create head-to-head comparison visualization"""
    plt.figure(figsize=(12, 8))
    teams = [home_stats['team'], away_stats['team']]
    recent_nrfi = [home_stats['recent_20_nrfi'], away_stats['recent_20_nrfi']]
    season_nrfi = [home_stats['nrfi_percentage'], away_stats['nrfi_percentage']]
    x = np.arange(len(teams))
    width = 0.35
    plt.bar(x - width/2, recent_nrfi, width, label='Last 20 Games NRFI%', color='#66b3ff')
    plt.bar(x + width/2, season_nrfi, width, label='Season NRFI%', color='#99ff99')
    plt.ylabel('NRFI Percentage')
    plt.title(f'NRFI Comparison - {game_date.strftime("%Y-%m-%d")}\n{away_stats["team"]} at {home_stats["team"]}')
    plt.xticks(x, teams)
    plt.legend()
    for i, v in enumerate(recent_nrfi):
        plt.text(i - width/2, v, f'{v:.1f}%', ha='center', va='bottom')
    for i, v in enumerate(season_nrfi):
        plt.text(i + width/2, v, f'{v:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'F1_head_to_head_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()

def create_matchup_trends(home_games, away_games, home_team, away_team, output_dir):
    """Create recent trends comparison for both teams"""
    plt.figure(figsize=(14, 7))
    home_recent = home_games.tail(20)
    away_recent = away_games.tail(20)
    home_rolling = home_recent['NRFI'].rolling(window=5).mean() * 100
    away_rolling = away_recent['NRFI'].rolling(window=5).mean() * 100
    plt.plot(range(len(home_rolling)), home_rolling, label=f'{home_team} (5-game avg)', linewidth=2)
    plt.plot(range(len(away_rolling)), away_rolling, label=f'{away_team} (5-game avg)', linewidth=2)
    plt.title('Recent NRFI Trends (Last 20 Games)')
    plt.xlabel('Games Ago')
    plt.ylabel('NRFI Percentage (5-game rolling average)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'F1_recent_trends.png'), bbox_inches='tight', dpi=300)
    plt.close()

def create_venue_analysis(historical_df, venue_name, output_dir):
    """Create venue-specific NRFI analysis"""
    venue_games = historical_df[historical_df['venue_name'] == venue_name].copy()
    if len(venue_games) == 0:
        return

    # Ensure date is in datetime format
    venue_games['date'] = pd.to_datetime(venue_games['date'])
    venue_games['month'] = venue_games['date'].dt.month

    monthly_stats = venue_games.groupby('month')['NRFI'].agg(['mean', 'count']).reset_index()
    if len(monthly_stats) == 0:
        return
    monthly_stats['mean'] = monthly_stats['mean'] * 100
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='month', y='mean', data=monthly_stats, color='#66b3ff')
    for i, p in enumerate(ax.patches):
        count = monthly_stats.iloc[i]['count']
        percentage = monthly_stats.iloc[i]['mean']
        ax.annotate(f'{percentage:.1f}%\n({count} games)', 
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom')
    plt.title(f'NRFI Performance at {venue_name}\nBy Month')
    plt.xlabel('Month')
    plt.ylabel('NRFI Percentage')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'F1_venue_analysis.png'), bbox_inches='tight', dpi=300)
    plt.close()

def analyze_upcoming_matchups():
    """Analyze all upcoming matchups"""
    historical_df, upcoming_df = load_and_prepare_data(include_upcoming=True)
    main_dir, teams_dir, upcoming_dir = create_directories()
    today = pd.Timestamp.now().normalize()

    # Clean up old directories for future dates
    for item in os.listdir(upcoming_dir):
        item_path = os.path.join(upcoming_dir, item)
        if os.path.isdir(item_path):
            try:
                item_date = pd.to_datetime(item.split('_')[0]).normalize()
                if item_date > today:
                    import shutil
                    shutil.rmtree(item_path)
            except:
                continue

    # Filter for MLB games only (exclude minor league and exhibition games)
    mlb_teams = set(pd.concat([historical_df['home_team'], historical_df['away_team']]).unique())
    upcoming_df = upcoming_df[(upcoming_df['home_team'].isin(mlb_teams)) & (upcoming_df['away_team'].isin(mlb_teams))]

    for _, game in tqdm(upcoming_df.iterrows(), desc="Processing upcoming matchups"):
        game_date = pd.to_datetime(game['date']).normalize()

        # Include games from today onwards
        if game_date < today:
            continue

        # Create matchup directory
        matchup_dir = os.path.join(upcoming_dir, f"{game_date.strftime('%Y-%m-%d')}_{game['away_team']}_at_{game['home_team']}")

        # Skip if analysis already exists for today
        if os.path.exists(matchup_dir):
            dir_mtime = pd.Timestamp.fromtimestamp(os.path.getmtime(matchup_dir)).normalize()
            if dir_mtime == today:
                print(f"Skipping {game['away_team']} at {game['home_team']} - analysis already exists")
                continue

        # Process the matchup even if pitcher data is missing
        matchup_dir = create_matchup_comparison(historical_df, game, upcoming_dir)
        create_matchup_summary(game, historical_df, matchup_dir)

def create_matchup_summary(game, historical_df, output_dir):
    """Create a text summary of the matchup"""
    summary = {
        'date': game['date'],
        'matchup': f"{game['away_team']} at {game['home_team']}",
        'venue': game['venue_name'],
        'pitchers': f"{game['away_pitcher_name']} vs {game['home_pitcher_name']}"
    }
    with open(os.path.join(output_dir, 'F1_matchup_summary.txt'), 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

def create_nrfi_temperature_map(df, output_dir):
    """Create a temperature map showing NRFI trends for all teams over last 10 games"""
    teams = pd.unique(pd.concat([df['home_team'], df['away_team']]))

    temp_data = []
    for team in teams:
        team_games = pd.concat([df[df['home_team'] == team][['date', 'NRFI']],
                                df[df['away_team'] == team][['date', 'NRFI']]]).sort_values('date')
        last_10 = team_games.tail(10)
        if len(last_10) < 5:
            continue
        rolling = last_10['NRFI'].rolling(window=5, min_periods=1).mean() * 100
        start_val = rolling.iloc[0]
        end_val = rolling.iloc[-1]
        trend = end_val - start_val
        temp = end_val
        temp_data.append({'team': team, 'temperature': temp, 'trend': trend, 'games': len(last_10)})

    temp_df = pd.DataFrame(temp_data)
    plt.figure(figsize=(14, 10))
    temp_df = temp_df.sort_values('temperature', ascending=False)
    sns.heatmap(temp_df[['temperature']].T, annot=True, fmt='.1f',
                cmap=plt.cm.coolwarm, linewidths=1, cbar=False,
                xticklabels=temp_df['team'])
    for i, (_, row) in enumerate(temp_df.iterrows()):
        if row['trend'] > 5:
            plt.text(i + 0.5, 0.5, '↑', ha='center', va='center', color='darkred', fontsize=16)
        elif row['trend'] < -5:
            plt.text(i + 0.5, 0.5, '↓', ha='center', va='center', color='darkblue', fontsize=16)
    plt.title('MLB Teams NRFI Temperature Map (Last 10 Games)', fontsize=16, pad=20)
    plt.ylabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(output_dir, 'F1_NRFI_Temperature_Map.png'), 
                    bbox_inches='tight', dpi=300)
        print(f"Temperature map saved successfully to {output_dir}")
    except Exception as e:
        print(f"Error saving temperature map: {e}")
    plt.close()
    return temp_df

def create_nrfi_trend_direction_heatmap(df, output_dir):
    """Create a heatmap showing NRFI trend direction for each team"""
    teams = pd.unique(pd.concat([df['home_team'], df['away_team']]))

    trend_data = []
    for team in teams:
        team_games = pd.concat([df[df['home_team'] == team][['date', 'NRFI']],
                                df[df['away_team'] == team][['date', 'NRFI']]]).sort_values('date')
        recent_games = team_games.tail(15)
        if len(recent_games) < 5:
            continue
        last_5 = recent_games.tail(5)['NRFI'].mean() * 100
        last_10 = recent_games.tail(10)['NRFI'].mean() * 100
        last_15 = recent_games['NRFI'].mean() * 100
        trend_5_vs_10 = last_5 - last_10
        trend_10_vs_15 = last_10 - last_15
        trend_data.append({
            'team': team,
            'last_5': last_5,
            'last_10': last_10,
            'last_15': last_15,
            'trend_5_vs_10': trend_5_vs_10,
            'trend_10_vs_15': trend_10_vs_15
        })
    trend_df = pd.DataFrame(trend_data)
    plt.figure(figsize=(14, 10))
    trend_df = trend_df.sort_values('last_5', ascending=False)
    heatmap_data = trend_df[['trend_5_vs_10', 'trend_10_vs_15']].copy()
    heatmap_data.index = trend_df['team']
    cmap = LinearSegmentedColormap.from_list('custom_rdylgn', ['#d73027', '#fee090', '#4575b4'])
    sns.heatmap(heatmap_data.T, annot=True, fmt='.1f',
                cmap=cmap, linewidths=1, center=0,
                xticklabels=heatmap_data.index,
                cbar_kws={'label': 'Trend Direction (%)'})
    plt.title('MLB Teams NRFI Trend Direction\nPositive = Improving, Negative = Declining', fontsize=16, pad=20)
    plt.ylabel('Trend Window')
    plt.yticks([0.5, 1.5], ['Last 5 vs Last 10', 'Last 10 vs Last 15'], rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(output_dir, 'F1_NRFI_Trend_Direction.png'), 
                    bbox_inches='tight', dpi=300)
        print(f"Trend direction heatmap saved successfully to {output_dir}")
    except Exception as e:
        print(f"Error saving trend direction heatmap: {e}")
    plt.close()
    return trend_df

def create_league_heat_matrix(df, output_dir):
    """Create a league-wide NRFI heat matrix for all teams"""
    # Define season date ranges
    season_ranges = {
        2018: (pd.to_datetime('2018-03-29'), pd.to_datetime('2018-10-28')),
        2019: (pd.to_datetime('2019-03-20'), pd.to_datetime('2019-10-30')),
        2020: (pd.to_datetime('2020-07-23'), pd.to_datetime('2020-10-27')),
        2021: (pd.to_datetime('2021-04-01'), pd.to_datetime('2021-11-02')),
        2022: (pd.to_datetime('2022-04-07'), pd.to_datetime('2022-11-05')),
        2023: (pd.to_datetime('2023-03-30'), pd.to_datetime('2023-11-04')),
        2024: (pd.to_datetime('2024-03-28'), pd.to_datetime('2024-11-02'))
    }

    # Filter data to only include games within season dates
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    valid_dates = pd.Series(False, index=df.index)
    for start, end in season_ranges.values():
        year_mask = (df['date'] >= start) & (df['date'] <= end)
        valid_dates = valid_dates | year_mask
    df = df[valid_dates]

    # Continue with existing heat matrix logic
    teams = pd.unique(pd.concat([df['home_team'], df['away_team']]))

    heat_data = []
    for team in teams:
        team_games = pd.concat([df[df['home_team'] == team][['date', 'NRFI']],
                                df[df['away_team'] == team][['date', 'NRFI']]]).sort_values('date')
        recent_games = team_games.tail(20)
        if len(recent_games) < 10:
            continue
        last_3 = recent_games.tail(3)['NRFI'].mean() * 100
        last_5 = recent_games.tail(5)['NRFI'].mean() * 100
        last_7 = recent_games.tail(7)['NRFI'].mean() * 100
        last_10 = recent_games.tail(10)['NRFI'].mean() * 100
        last_15 = recent_games.tail(15)['NRFI'].mean() * 100
        last_20 = recent_games['NRFI'].mean() * 100
        heat_data.append({
            'team': team,
            'last_3': last_3,
            'last_5': last_5,
            'last_7': last_7,
            'last_10': last_10,
            'last_15': last_15,
            'last_20': last_20
        })
    plt.figure(figsize=(16, 12))
    heat_df = pd.DataFrame(heat_data)
    heat_df = heat_df.sort_values('last_5', ascending=False)
    heatmap_data = heat_df[['last_3', 'last_5', 'last_7', 'last_10', 'last_15', 'last_20']].copy()
    heatmap_data.index = heat_df['team']
    sns.heatmap(heatmap_data, annot=True, fmt='.1f',
                cmap='coolwarm', linewidths=0.5, 
                xticklabels=['Last 3', 'Last 5', 'Last 7', 'Last 10', 'Last 15', 'Last 20'],
                vmin=0, vmax=100)
    plt.title('MLB Teams NRFI Heat Matrix', fontsize=16, pad=20)
    plt.ylabel('Team')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'F1_NRFI_Heat_Matrix.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    return heat_df

# -------------------------------
# Main Function: Run Analysis and Generate Visuals
# -------------------------------
def main():
    """Main function to run analysis and generate visuals"""
    # Create directories
    main_dir, teams_dir, upcoming_dir = create_directories()

    # Load historical data
    df = load_and_prepare_data()

    # Ensure date column is in datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # Drop rows with invalid dates

    # Calculate league statistics
    league_stats = calculate_league_stats(df)

    # Get unique teams for current season
    all_teams = pd.unique(pd.concat([df['home_team'], df['away_team']]))

    team_statistics = []

    for team in tqdm(all_teams, desc="Processing teams"):
        # Generate visualizations using current season data only
        stats = create_team_nrfi_trend(df, team, teams_dir, league_stats)
        team_statistics.append(stats)

        # Additional visualizations (all using current season data)
        create_team_nrfi_trend_improved(df, team, teams_dir, league_stats)
        create_nrfi_calendar_heatmap(df, team, teams_dir)
        create_nrfi_moving_averages(df, team, teams_dir, league_stats)
        create_weekly_performance_chart(df, team, teams_dir)
        create_home_away_comparison(df, team, teams_dir, league_stats)
        create_violin_box_plots(df, team, teams_dir)

        # Generate radar chart with current season metrics
        radar_metrics = ['Home', 'Away', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        # Calculate actual values based on current season data instead of random
        home_games = df[df['home_team'] == team]['NRFI'].mean() * 100
        away_games = df[df['away_team'] == team]['NRFI'].mean() * 100
        days_data = []

        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            day_games = df[
                (df['home_team'] == team) | (df['away_team'] == team) & 
                (pd.to_datetime(df['date']).dt.day_name() == day)
            ]['NRFI'].mean() * 100
            days_data.append(day_games if not pd.isna(day_games) else 50)  # Default to 50 if no games on that day

        radar_values = [home_games, away_games] + days_data
        create_nrfi_radar_chart(team, radar_metrics, radar_values, teams_dir)

        # Only forecast if we have enough current season data
        if len(df[df['home_team'] == team]) + len(df[df['away_team'] == team]) > 30:
            forecast_nrfi_trend(df, team, teams_dir, periods=10)

    # Save team rankings and league stats to DATA_DIR
    rankings_df = pd.DataFrame(team_statistics).sort_values('recent_20_nrfi', ascending=False)
    rankings_df.to_csv(os.path.join(DATA_DIR, 'F1_NRFI_Rankings.csv'), index=False)
    league_df = pd.DataFrame([league_stats])
    league_df.to_csv(os.path.join(DATA_DIR, 'F1_League_Baseline_Stats.csv'), index=False)

    # Create league comparison dashboard using VISUALS_DIR
    create_league_comparison_dashboard(rankings_df, VISUALS_DIR, league_stats)

    # Generate advanced league-wide visuals using current season data
    create_sankey_diagram(VISUALS_DIR)
    create_animated_nrfi_trend(df, VISUALS_DIR)
    create_hexbin_density_plot(df, VISUALS_DIR)
    run_interactive_dashboard()

    # Analyze upcoming matchups with current season context
    analyze_upcoming_matchups()

    # Additional advanced visualizations using VISUALS_DIR
    create_nrfi_temperature_map(df, VISUALS_DIR)
    create_nrfi_trend_direction_heatmap(df, VISUALS_DIR)
    create_league_heat_matrix(df, VISUALS_DIR)

    print('NRFI trend analyses and advanced visualizations generated successfully!')

if __name__ == "__main__":
    main()
