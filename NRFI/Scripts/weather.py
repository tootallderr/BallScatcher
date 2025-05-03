#!/usr/bin/env python3
# weather.py - Script to fetch weather data for MLB stadiums

import requests
import pandas as pd
from datetime import datetime
import os
import time
import json
from typing import Dict, Tuple
import logging
import NRFI.config as config
import sys

# Add parent directory to Python path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, NRFI_DIR, setup_logging # Import NRFI_DIR

# Setup logging using centralized configuration
logger = setup_logging('Weather')

# Define file paths using config variables
SCHEDULE_FILE = os.path.join(NRFI_DIR, "F1_historical_schedule.csv") # Load from NRFI_DIR
WEATHER_OUTPUT_FILE = os.path.join(NRFI_DIR, "detailed_weather_data.csv") # Save to NRFI_DIR

# Ensure NRFI directory exists
os.makedirs(NRFI_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("weather_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MLB Stadium information with ZIP codes
MLB_STADIUMS = {
    "Angels Stadium": {"team": "LAA", "zip": "92806", "city": "Anaheim", "state": "CA"},
    "Minute Maid Park": {"team": "HOU", "zip": "77002", "city": "Houston", "state": "TX"},
    "RingCentral Coliseum": {"team": "OAK", "zip": "94621", "city": "Oakland", "state": "CA"},
    "T-Mobile Park": {"team": "SEA", "zip": "98134", "city": "Seattle", "state": "WA"},
    "Globe Life Field": {"team": "TEX", "zip": "76011", "city": "Arlington", "state": "TX"},
    "Rogers Centre": {"team": "TOR", "zip": "M5V 1J1", "city": "Toronto", "state": "ON", "country": "CA"},
    "Truist Park": {"team": "ATL", "zip": "30339", "city": "Atlanta", "state": "GA"},
    "American Family Field": {"team": "MIL", "zip": "53214", "city": "Milwaukee", "state": "WI"},
    "Busch Stadium": {"team": "STL", "zip": "63102", "city": "St. Louis", "state": "MO"},
    "Great American Ball Park": {"team": "CIN", "zip": "45202", "city": "Cincinnati", "state": "OH"},
    "PNC Park": {"team": "PIT", "zip": "15212", "city": "Pittsburgh", "state": "PA"},
    "Wrigley Field": {"team": "CHC", "zip": "60613", "city": "Chicago", "state": "IL"},
    "Chase Field": {"team": "ARI", "zip": "85004", "city": "Phoenix", "state": "AZ"},
    "Coors Field": {"team": "COL", "zip": "80205", "city": "Denver", "state": "CO"},
    "Dodger Stadium": {"team": "LAD", "zip": "90012", "city": "Los Angeles", "state": "CA"},
    "Oracle Park": {"team": "SF", "zip": "94107", "city": "San Francisco", "state": "CA"},
    "Petco Park": {"team": "SD", "zip": "92101", "city": "San Diego", "state": "CA"},
    "Guaranteed Rate Field": {"team": "CWS", "zip": "60616", "city": "Chicago", "state": "IL"},
    "Comerica Park": {"team": "DET", "zip": "48201", "city": "Detroit", "state": "MI"},
    "Kauffman Stadium": {"team": "KC", "zip": "64129", "city": "Kansas City", "state": "MO"},
    "Progressive Field": {"team": "CLE", "zip": "44115", "city": "Cleveland", "state": "OH"},
    "Target Field": {"team": "MIN", "zip": "55403", "city": "Minneapolis", "state": "MN"},
    "Camden Yards": {"team": "BAL", "zip": "21201", "city": "Baltimore", "state": "MD"},
    "Fenway Park": {"team": "BOS", "zip": "02215", "city": "Boston", "state": "MA"},
    "Citi Field": {"team": "NYM", "zip": "11368", "city": "Queens", "state": "NY"},
    "Citizens Bank Park": {"team": "PHI", "zip": "19148", "city": "Philadelphia", "state": "PA"},
    "Nationals Park": {"team": "WSH", "zip": "20003", "city": "Washington", "state": "DC"},
    "Tropicana Field": {"team": "TB", "zip": "33705", "city": "St. Petersburg", "state": "FL"},
    "LoanDepot Park": {"team": "MIA", "zip": "33125", "city": "Miami", "state": "FL"},
    "Yankee Stadium": {"team": "NYY", "zip": "10451", "city": "Bronx", "state": "NY"},
}

class WeatherScraper:
    def __init__(self, api_key=None, api_service="weatherapi"):
        """
        Initialize the weather scraper with API keys and service selection
        
        Args:
            api_key: API key for the weather service
            api_service: The weather service to use ('weatherapi', 'openweathermap', or 'nws')
        """
        # Use provided API key, environment variable, or default WeatherAPI.com key
        self.api_key = api_key or os.environ.get("WEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Please provide it via argument or environment variable.")
        self.api_service = api_service
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Create a directory for historical weather data
        self.historical_dir = os.path.join(self.data_dir, "historical_weather")
        if not os.path.exists(self.historical_dir):
            os.makedirs(self.historical_dir)
        
        # Different services have different endpoints and response formats
        if api_service == "weatherapi":
            if not self.api_key:
                raise ValueError("WeatherAPI requires an API key")
            self.base_url = "http://api.weatherapi.com/v1"
        elif api_service == "openweathermap":
            if not self.api_key:
                raise ValueError("OpenWeatherMap requires an API key")
            self.base_url = "https://api.openweathermap.org/data/2.5"
        elif api_service == "nws":
            # National Weather Service doesn't require an API key
            self.base_url = "https://api.weather.gov"
        else:
            raise ValueError(f"Unsupported weather API service: {api_service}")

        # Load the MLB schedule
        self.load_mlb_schedule()

    def load_mlb_schedule(self):
        """Load and process the MLB schedule from historical_schedule.csv"""
        schedule_path = os.path.join(NRFI_DIR, "F1_historical_schedule.csv")
        if os.path.exists(schedule_path):
            self.schedule_df = pd.read_csv(schedule_path)
            # Convert date column to datetime
            self.schedule_df['date'] = pd.to_datetime(self.schedule_df['date'])
            # Sort by date
            self.schedule_df = self.schedule_df.sort_values('date')
        else:
            logger.error(f"MLB schedule file not found at {schedule_path}")
            self.schedule_df = pd.DataFrame()

    def get_games_for_date(self, date: datetime) -> pd.DataFrame:
        """Get all games scheduled for a specific date"""
        return self.schedule_df[self.schedule_df['date'].dt.date == date.date()]

    def get_weather_filename(self, date: datetime) -> str:
        """Generate filename for weather data based on date"""
        return os.path.join(self.historical_dir, f"weather_{date.strftime('%Y%m%d')}.csv")

    def is_weather_data_exists(self, date: datetime) -> bool:
        """Check if weather data for a specific date already exists"""
        filename = self.get_weather_filename(date)
        return os.path.exists(filename)

    def get_season_range(self, season: int) -> Tuple[datetime, datetime]:
        """Get start and end dates for a given MLB season from config"""
        if season not in config.MLB_SEASON_DATES:
            raise ValueError(f"Season {season} not found in config.MLB_SEASON_DATES")
            
        season_dates = config.MLB_SEASON_DATES[season]
        regular_start = datetime.strptime(season_dates['regular_season'][0], '%Y-%m-%d')
        postseason_end = datetime.strptime(season_dates['postseason'][1], '%Y-%m-%d')
        
        return regular_start, postseason_end

    def update_weather_data(self):
        """
        Update weather data intelligently:
        1. For today's games: Always update as conditions change
        2. For historical dates: Only fetch if data doesn't exist
        3. For future dates in schedule: Update daily forecasts
        """
        if self.schedule_df.empty:
            logger.error("No schedule data loaded")
            return

        current_date = datetime.now().date()
        
        # Get all unique dates from schedule that are up to today or in future
        all_dates = pd.to_datetime(self.schedule_df['date'].unique())
        dates_to_process = [d for d in all_dates if d.date() <= current_date]
        dates_to_process.sort()

        # Process each date
        for date in dates_to_process:
            date_to_process = date.to_pydatetime()
            should_update = False

            # Determine if we should update this date's data
            if date_to_process.date() == current_date:
                # Always update today's data as conditions change
                should_update = True
                logger.info(f"Updating today's weather data")
            elif date_to_process.date() > current_date:
                # Update future dates (forecast data)
                should_update = True
                logger.info(f"Updating forecast for {date_to_process.date()}")
            elif not self.is_weather_data_exists(date_to_process):
                # Update if historical data is missing
                should_update = True
                logger.info(f"Fetching historical weather for {date_to_process.date()}")

            if should_update:
                date_games = self.get_games_for_date(date_to_process)
                if not date_games.empty:
                    # Get weather for all games on this date
                    weather_data = {}
                    for _, game in date_games.iterrows():
                        venue_name = game['venue_name']
                        game_time = pd.to_datetime(game['game_time'])
                        
                        # Find the stadium in our MLB_STADIUMS dictionary
                        stadium_info = None
                        for stadium, info in MLB_STADIUMS.items():
                            if venue_name in stadium:  # Partial match to handle naming variations
                                stadium_info = info
                                break
                        
                        if stadium_info:
                            try:
                                weather = self.get_stadium_weather(
                                    zip_code=stadium_info['zip'],
                                    date=game_time
                                )
                                if weather:
                                    weather_data[venue_name] = {
                                        'game_pk': game['game_pk'],
                                        'game_time': game_time,
                                        'home_team': game['home_team'],
                                        'away_team': game['away_team'],
                                        'weather': weather
                                    }
                            except Exception as e:
                                logger.error(f"Error fetching weather for {venue_name}: {str(e)}")
                    
                    # Save weather data for this date
                    if weather_data:
                        self.save_weather_to_csv(weather_data, self.get_weather_filename(date_to_process))
                    
                    # Be nice to the API
                    time.sleep(1)

    def get_historical_weather(self, game_pk: int) -> Dict:
        """
        Get historical weather data for a specific game
        
        Args:
            game_pk: Game ID to get weather for
            
        Returns:
            Dictionary with weather data if found, None otherwise
        """
        # Find the game in schedule
        game_info = self.schedule_df[self.schedule_df['game_pk'] == game_pk]
        if game_info.empty:
            return None
            
        game_date = game_info['date'].iloc[0]
        filename = self.get_weather_filename(game_date)
        
        if os.path.exists(filename):
            weather_df = pd.read_csv(filename)
            game_weather = weather_df[weather_df['game_pk'] == game_pk]
            if not game_weather.empty:
                return json.loads(game_weather['weather'].iloc[0])
        
        return None

    def get_stadium_weather(self, stadium_name=None, team=None, zip_code=None, date=None):
        """
        Get current weather or forecast for a stadium
        
        Args:
            stadium_name: Name of the stadium
            team: Team code (alternative to stadium_name)
            zip_code: ZIP code (alternative to stadium_name and team)
            date: Optional date for forecast (default: current weather)
            
        Returns:
            Dictionary with weather data
        """
        # Determine the ZIP code based on inputs
        if zip_code is None:
            if stadium_name is not None:
                if stadium_name in MLB_STADIUMS:
                    zip_code = MLB_STADIUMS[stadium_name]["zip"]
                else:
                    raise ValueError(f"Stadium not found: {stadium_name}")
            elif team is not None:
                for stadium, info in MLB_STADIUMS.items():
                    if info["team"] == team:
                        zip_code = info["zip"]
                        stadium_name = stadium
                        break
                else:
                    raise ValueError(f"Team not found: {team}")
            else:
                raise ValueError("Must provide stadium_name, team, or zip_code")
        
        # Get weather data
        if self.api_service == "weatherapi":
            return self._get_weatherapi_data(zip_code, date)
        elif self.api_service == "openweathermap":
            return self._get_openweathermap_data(zip_code, date)
        elif self.api_service == "nws":
            return self._get_nws_data(zip_code, date)
        
    def _get_weatherapi_data(self, zip_code, date=None):
        """Get data from WeatherAPI.com"""
        endpoint = "/forecast.json" if date else "/current.json"
        
        params = {
            "key": self.api_key,
            "q": zip_code,
            "aqi": "yes"  # Include air quality data
        }
        
        if date:
            # Format date as YYYY-MM-DD
            if isinstance(date, datetime):
                date_str = date.strftime("%Y-%m-%d")
            else:
                date_str = date
            params["dt"] = date_str
            params["days"] = 1
            
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant weather information
            if date:
                forecast = data["forecast"]["forecastday"][0]
                return {
                    "date": forecast["date"],
                    "max_temp_f": forecast["day"]["maxtemp_f"],
                    "min_temp_f": forecast["day"]["mintemp_f"],
                    "avg_temp_f": forecast["day"]["avgtemp_f"],
                    "condition": forecast["day"]["condition"]["text"],
                    "max_wind_mph": forecast["day"]["maxwind_mph"],
                    "total_precip_in": forecast["day"]["totalprecip_in"],
                    "humidity": forecast["day"]["avghumidity"],
                    "hourly": [{
                        "time": hour["time"],
                        "temp_f": hour["temp_f"],
                        "wind_mph": hour["wind_mph"],
                        "wind_direction": hour["wind_dir"],
                        "wind_degree": hour["wind_degree"],
                        "pressure": hour["pressure_mb"],
                        "humidity": hour["humidity"],
                        "cloud": hour["cloud"],
                        "precipitation_in": hour["precip_in"],
                        "condition": hour["condition"]["text"]
                    } for hour in forecast["hour"]]
                }
            else:
                # Handle current weather data
                current = data.get("current", {})
                
                # Check if current data exists and has the expected format
                if not current:
                    logger.error(f"No current weather data found in response for {zip_code}")
                    return None

                # Build a response with safe access to potentially missing keys
                return {
                    "last_updated": current.get("last_updated", datetime.now().isoformat()),
                    "temp_f": current.get("temp_f", None),
                    "wind_mph": current.get("wind_mph", None),
                    "wind_direction": current.get("wind_dir", ""),
                    "wind_degree": current.get("wind_degree", None),
                    "pressure": current.get("pressure_mb", None),
                    "humidity": current.get("humidity", None),
                    "cloud": current.get("cloud", None),
                    "precipitation_in": current.get("precip_in", 0),
                    "condition": current.get("condition", {}).get("text", "Unknown")
                }
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching WeatherAPI data for {zip_code}: {str(e)}")
            return None
        except KeyError as e:
            logger.error(f"Missing expected key in WeatherAPI response for {zip_code}: {str(e)}")
            # Print the actual response structure to help debug
            logger.debug(f"Received response structure: {data}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing weather data for {zip_code}: {str(e)}")
            return None

    def _get_openweathermap_data(self, zip_code, date=None):
        """Get data from OpenWeatherMap"""
        if date:
            # For forecast
            endpoint = "/forecast"
        else:
            # For current weather
            endpoint = "/weather"
            
        params = {
            "zip": f"{zip_code},us",
            "appid": self.api_key,
            "units": "imperial"  # Use imperial units for temperature in Fahrenheit
        }
            
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            if not date:
                # Current weather
                return {
                    "timestamp": datetime.fromtimestamp(data["dt"]).isoformat(),
                    "temp_f": data["main"]["temp"],
                    "feels_like_f": data["main"]["feels_like"],
                    "temp_min_f": data["main"]["temp_min"],
                    "temp_max_f": data["main"]["temp_max"],
                    "pressure": data["main"]["pressure"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed_mph": data["wind"]["speed"],
                    "wind_degree": data["wind"]["deg"],
                    "wind_direction": self._degree_to_direction(data["wind"]["deg"]),
                    "clouds_percent": data["clouds"]["all"],
                    "condition": data["weather"][0]["main"],
                    "description": data["weather"][0]["description"]
                }
            else:
                # Process forecast data
                forecast_data = []
                for forecast in data["list"]:
                    forecast_data.append({
                        "timestamp": datetime.fromtimestamp(forecast["dt"]).isoformat(),
                        "temp_f": forecast["main"]["temp"],
                        "feels_like_f": forecast["main"]["feels_like"],
                        "pressure": forecast["main"]["pressure"],
                        "humidity": forecast["main"]["humidity"],
                        "wind_speed_mph": forecast["wind"]["speed"],
                        "wind_degree": forecast["wind"]["deg"],
                        "wind_direction": self._degree_to_direction(forecast["wind"]["deg"]),
                        "clouds_percent": forecast["clouds"]["all"],
                        "condition": forecast["weather"][0]["main"],
                        "description": forecast["weather"][0]["description"],
                        "rain_3h": forecast.get("rain", {}).get("3h", 0),
                    })
                return forecast_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching OpenWeatherMap data for {zip_code}: {str(e)}")
            return None

    def _get_nws_data(self, zip_code, date=None):
        """Get data from National Weather Service (NWS)"""
        # NWS API requires coordinates, not ZIP codes
        # First, we need to convert ZIP to lat/lon using a geocoding service
        coords = self._zip_to_coords(zip_code)
        if not coords:
            logger.error(f"Could not convert ZIP code {zip_code} to coordinates")
            return None
            
        lat, lon = coords
        
        # Get points information for the coordinates
        try:
            points_url = f"{self.base_url}/points/{lat},{lon}"
            response = requests.get(points_url, headers={'User-Agent': 'MLBWeatherTracker'})
            response.raise_for_status()
            points_data = response.json()
            
            # Get the forecast URL from the points response
            if date:
                forecast_url = points_data["properties"]["forecast"]
            else:
                forecast_url = points_data["properties"]["forecastHourly"]
                
            # Get the forecast
            forecast_response = requests.get(forecast_url, headers={'User-Agent': 'MLBWeatherTracker'})
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
            periods = forecast_data["properties"]["periods"]
            
            # Process the periods into a standardized format
            processed_data = []
            for period in periods:
                processed_data.append({
                    "timestamp": period["startTime"],
                    "temp_f": period["temperature"],
                    "wind_speed": period["windSpeed"],
                    "wind_direction": period["windDirection"],
                    "short_forecast": period["shortForecast"],
                    "detailed_forecast": period.get("detailedForecast", ""),
                    "precipitation_probability": period.get("probabilityOfPrecipitation", {}).get("value", 0)
                })
            
            # If we're not looking for a specific date, return just the first period
            if not date:
                return processed_data[0]
            return processed_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NWS data for {zip_code}: {str(e)}")
            return None
    
    def _zip_to_coords(self, zip_code):
        """Convert ZIP code to latitude/longitude using Census.gov API"""
        try:
            # Handle Canadian postal codes for Toronto
            if " " in zip_code:
                # Use a default coordinate for Rogers Centre in Toronto
                return (43.6414, -79.3894)
                
            response = requests.get(
                f"https://geocoding.geo.census.gov/geocoder/locations/address",
                params={
                    "street": "",
                    "city": "",
                    "state": "",
                    "benchmark": "2020",
                    "format": "json",
                    "zip": zip_code
                }
            )
            response.raise_for_status()
            data = response.json()
            
            if data["result"]["addressMatches"]:
                coords = data["result"]["addressMatches"][0]["coordinates"]
                return (coords["y"], coords["x"])  # lat, lon
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error geocoding ZIP code {zip_code}: {str(e)}")
            return None
    
    def _degree_to_direction(self, degree):
        """Convert a wind degree to a cardinal direction"""
        directions = [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
        ]
        index = round(degree / 22.5) % 16
        return directions[index]
    
    def get_all_stadiums_weather(self, date=None):
        """Get weather for all MLB stadiums"""
        results = {}
        for stadium_name, info in MLB_STADIUMS.items():
            try:
                logger.info(f"Fetching weather for {stadium_name}")
                weather = self.get_stadium_weather(stadium_name=stadium_name, date=date)
                if weather:
                    results[stadium_name] = {
                        "team": info["team"],
                        "location": f"{info['city']}, {info['state']}",
                        "weather": weather
                    }
                    
                # Be nice to the API and add a delay
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error fetching weather for {stadium_name}: {str(e)}")
                
        return results
    
    def save_weather_to_csv(self, weather_data, filename=None):
        """Save weather data to CSV"""
        if filename is None:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = os.path.join(NRFI_DIR, f"detailed_weather_data.csv")
            
        # Convert nested dictionary to flatten structure for DataFrame
        flat_data = []
        for stadium, data in weather_data.items():
            weather = data["weather"]
            if isinstance(weather, list):
                # For forecast data
                for w in weather:
                    flat_data.append({
                        "stadium": stadium,
                        "team": data["team"],
                        "location": data["location"],
                        **w
                    })
            else:
                # For current weather
                flat_data.append({
                    "stadium": stadium,
                    "team": data["team"],
                    "location": data["location"],
                    **weather
                })
                
        df = pd.DataFrame(flat_data)
        df.to_csv(filename, index=False)
        logger.info(f"Weather data saved to {filename}")
        return filename

def main():
    """Main function to run the weather scraper"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLB Stadium Weather Scraper")
    parser.add_argument("--api", choices=["weatherapi", "openweathermap", "nws"], 
                        default="weatherapi", help="Weather API service to use")
    parser.add_argument("--key", help="API key for the weather service")
    parser.add_argument("--stadium", help="Stadium name to get weather for (optional)")
    parser.add_argument("--team", help="Team code to get weather for (optional)")
    parser.add_argument("--date", help="Date for forecast in YYYY-MM-DD format (optional)")
    parser.add_argument("--all", action="store_true", help="Get weather for all stadiums")
    parser.add_argument("--output", help="Output CSV file path (optional)")
    parser.add_argument("--update-historical", action="store_true", 
                        help="Update historical weather data")
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.key or os.environ.get(f"{args.api.upper()}_API_KEY")
    
    scraper = WeatherScraper(api_key=api_key, api_service=args.api)
    
    if args.update_historical:
        # If a specific date is provided, use that, otherwise update all dates
        specific_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None
        scraper.update_weather_data(specific_date)
    elif args.all:
        # Get weather for all stadiums
        date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None
        weather_data = scraper.get_all_stadiums_weather(date=date)
        filename = scraper.save_weather_to_csv(weather_data, filename=args.output)
        print(f"Weather data saved to {filename}")
    else:
        # Get weather for a specific stadium or team
        if not (args.stadium or args.team):
            parser.error("Either --stadium, --team, or --all must be specified")
            
        date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None
        weather = scraper.get_stadium_weather(
            stadium_name=args.stadium,
            team=args.team,
            date=date
        )
        
        print(json.dumps(weather, indent=2))

if __name__ == "__main__":
    main()