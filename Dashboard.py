import os
import sys
import subprocess

def ensure_venv():
    """Ensure we're running in the correct virtual environment"""
    # Check if we're already in the right venv
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True
        
    # If not, relaunch using launch.py
    print("Not running in virtual environment. Launching with proper environment...")
    launch_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch.py")
    subprocess.run([sys.executable, launch_script])
    sys.exit()

# Check virtual environment before any other imports
ensure_venv()

# Package dependency checker - runs before other imports
import os
import sys
import subprocess
import importlib.util
import pkg_resources

# Base directory constants - defined early for requirements check
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")

def check_and_install_dependencies():
    """Check for missing dependencies and install them if needed"""
    print("Checking dependencies...")
    
    # Function to parse requirements.txt
    def parse_requirements():
        if not os.path.exists(REQUIREMENTS_FILE):
            print(f"Warning: requirements.txt not found at {REQUIREMENTS_FILE}")
            return []
            
        requirements = []
        with open(REQUIREMENTS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and filepath comments
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                # Remove version specifiers
                if '=' in line or '>' in line or '<' in line:
                    line = line.split('=')[0].split('>')[0].split('<')[0]
                line = line.strip()
                if line:  # Ensure there's actually a package name
                    requirements.append(line)
        return requirements
    
    # Function to check if a package is installed
    def is_package_installed(package_name):
        try:
            if package_name.lower() == 'tkinter':
                # Special handling for tkinter which is part of standard library
                spec = importlib.util.find_spec('tkinter')
                return spec is not None
            else:
                pkg_resources.get_distribution(package_name)
                return True
        except pkg_resources.DistributionNotFound:
            return False
    
    # Get list of required packages
    required_packages = parse_requirements()
    missing_packages = [pkg for pkg in required_packages if not is_package_installed(pkg)]
    
    if not missing_packages:
        print("All required packages are installed.")
        return True
    
    print(f"Missing packages: {', '.join(missing_packages)}")
    
    # Try to install missing packages
    try:
        # Use pip to install missing packages
        cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        print(f"Installing missing packages: {' '.join(missing_packages)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error installing packages: {result.stderr}")
            return False
        
        print("Successfully installed missing packages.")
        return True
    except Exception as e:
        print(f"Failed to install missing packages: {e}")
        return False

# Run dependency check before importing other modules
check_and_install_dependencies()

# Now proceed with normal imports
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from datetime import datetime
import subprocess
from PIL import Image, ImageTk
from tkinter import ttk, PhotoImage, Scrollbar
import glob
import pandas as pd
import numpy as np
import re
from datetime import datetime
from NRFI.config import START_SEASON
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from concurrent.futures import ThreadPoolExecutor
import queue
from config import MODE_CONFIGS, get_mode_config, get_mode_paths, get_visuals_dirs
from config import PLOTS_DIR  # Import PLOTS_DIR from config

# Base directory constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Debug mode for additional logging and interface elements
DEBUG_MODE = False  # Set to True to enable debug information in the UI

# Add these constants near the top after imports
# Minimum dimensions and layout constants
MIN_WINDOW_WIDTH = 1024
MIN_WINDOW_HEIGHT = 768
HEADER_MIN_HEIGHT = 150
CONTENT_MIN_HEIGHT = 400
LOG_MIN_HEIGHT = 150

# Enhanced color scheme
COLORS = {
    'bg_dark': "#000000",           # Main background
    'bg_content': "#1e1e1e",        # Content areas
    'accent_primary': "#2962ff",    # Primary accent (buttons, selections)
    'accent_secondary': "#0097a7",  # Secondary accent
    'accent_success': "#2ecc71",    # Success indicators
    'accent_warning': "#f39c12",    # Warning/running indicators
    'accent_error': "#e74c3c",      # Error indicators
    'text_primary': "#ffffff",      # Primary text
    'text_secondary': "#b3b3b3",    # Secondary text
    'border': "#333333",            # Borders
    'hover': "#3d5afe",             # Hover state
    'button_gradient': ["#2962ff", "#0039cb"]  # Button gradient
}

# Style configurations
STYLES = {
    'button': {
        'font': ('Arial', 10),
        'padding': (10, 5),
        'radius': 4,
        'border_width': 0
    },
    'heading': {
        'font': ('Arial', 14, 'bold'),
        'padding': (10, 5)
    },
    'text': {
        'font': ('Arial', 10),
        'padding': (5, 2)
    }
}

# Add new constant for streak windows
STREAK_WINDOWS = [
    (3, "Last 3"),
    (5, "Last 5"),
    (10, "Last 10")
]

# Analysis mode constants
ANALYSIS_MODES = [
    ("NRFI", "No Runs First Inning"),
    ("PROPS", "Player Props"),
    ("ALL_INNINGS", "All Innings Analysis"),
    ("AOI_SPORTS", "AOI Sports Analytics")
]

# Graph type definitionss
TEAM_GRAPH_TYPES = [
    ("_NRFI_Analysis.png", "Overall Analysis"),
    ("_Home_Away_Comparison.png", "Home/Away"),
    ("_NRFI_Calendar.png", "Calendar"), 
    ("_NRFI_Forecast.png", "Forecast"),
    ("_NRFI_Moving_Averages.png", "Moving Avg"),
    ("_NRFI_Trend_Enhanced.png", "Trends"),
    ("_NRFI_Weekly.png", "Weekly"),
    ("_Radar_Chart.png", "Radar"),
    ("_Violin_Box.png", "Distribution")
]

MATCHUP_GRAPH_TYPES = [
    ("F1_head_to_head_comparison.png", "H2H Compare"),
    ("F1_recent_trends.png", "Recent Trends"), 
    ("F1_venue_analysis.png", "Venue Analysis")
]

MODEL_EVAL_GRAPHS = [
    ("feature_importance.png", "Feature Importance"),
    ("confusion_matrix.png", "Confusion Matrix"),
    ("F1_backtest_performance.png", "Backtest Performance")
]

# --- Tooltip Implementation ---
class ToolTip:
    """Enhanced tooltip with delay for tkinter widgets."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.id = None  # For after_cancel
        self.hover_start = None  # Track when hover starts
        self.delay = 2000  # Delay in milliseconds (2 seconds)
        widget.bind("<Enter>", self.schedule_tip)
        widget.bind("<Leave>", self.hide_tip)
        widget.bind("<Motion>", self.on_motion)  # Track mouse movement

    def schedule_tip(self, event=None):
        """Schedule the tooltip to appear after delay"""
        self.hover_start = time.time()
        self.unschedule()  # Cancel any existing scheduled tooltip
        self.id = self.widget.after(self.delay, self.show_tip)

    def unschedule(self):
        """Cancel scheduled tooltip"""
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def on_motion(self, event=None):
        """Reset timer on mouse movement"""
        if self.tip_window:
            self.hide_tip()
        self.schedule_tip()

    def show_tip(self, event=None):
        """Display the tooltip"""
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, 
            text=self.text, 
            justify=tk.LEFT,
            background="#ffffe0", 
            relief=tk.SOLID, 
            borderwidth=1,
            font=("tahoma", "8", "normal")
        )
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        """Hide the tooltip"""
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None
        self.unschedule()

# --- Redirect stdout/stderr ---
class RedirectText:
    """Redirect stdout to a tkinter Text widget"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        # Schedule the update shortly to avoid freezing
        self.text_widget.after(10, self.update_widget)
    
    def update_widget(self):
        if self.buffer:
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, self.buffer)
            self.text_widget.configure(state='disabled')
            self.text_widget.see(tk.END)
            self.buffer = ""
    
    def flush(self):
        pass

# --- Main Dashboard Class ---
class NRFIDashboard:
    def __init__(self, root):
        self.root = root
        root.title("Baseball Analysis Dashboard")
        
        # Apply custom styles for ttk widgets
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview",
            background=COLORS['bg_content'],
            foreground=COLORS['text_primary'],
            fieldbackground=COLORS['bg_content'],
            font=STYLES['text']['font']
        )
        style.configure("Treeview.Heading",
            background=COLORS['bg_dark'],
            foreground=COLORS['text_primary'],
            font=STYLES['heading']['font']
        )
        
        # Configure grid for proportional resizing
        root.grid_columnconfigure(0, weight=1)  # Main column expands
        for i in range(3):  # Three main rows
            weight = 3 if i == 1 else 1  # Content row expands more
            root.grid_rowconfigure(i, weight=weight)
        
        # Set minimum window size
        root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        
        # Configure main grid weights
        root.grid_rowconfigure(0, weight=0, minsize=HEADER_MIN_HEIGHT)  # Header fixed
        root.grid_rowconfigure(1, weight=3, minsize=CONTENT_MIN_HEIGHT)  # Content expands most
        root.grid_rowconfigure(2, weight=1, minsize=LOG_MIN_HEIGHT)     # Log expands less
        root.grid_columnconfigure(0, weight=1)

        # --- Dark Theme Colors ---
        self.bg_color = "#000000"          # Overall background (black)
        self.frame_color = "#1e1e1e"       # Dark gray frames
        self.button_color = "#007acc"      # Blue buttons
        self.button_text_color = "#ffffff" # White text on buttons
        self.running_color = "#f39c12"     # Vibrant orange for running
        self.completed_color = "#2ecc71"   # Green for completed
        self.error_color = "#e74c3c"       # Red for error

        # Current analysis mode
        self.analysis_mode = tk.StringVar(value="NRFI")  # Default to NRFI analysis

        # Load and set background image (if exists)
        bg_image_path = os.path.join("data", "Dashboard", "mlb", "mlb_fallback_2.jpg")
        if os.path.exists(bg_image_path):
            bg_image = Image.open(bg_image_path)
            bg_image = bg_image.resize((900, 700), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            self.bg_label = tk.Label(root, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            self.bg_label.lower()

        # Use dark background for the root window
        self.root.configure(bg=self.bg_color)
        
        # Define pipeline steps based on the analysis mode
        self.update_steps_for_mode()
        
        # Create main frames using grid
        self.create_header_frame()  # Row 0
        self.create_content_frame() # Row 1
        self.create_log_frame()     # Row 2

        # Track running process
        self.currently_running = None
        self.process = None

        # Bind keyboard shortcuts
        self.root.bind("<Control-r>", lambda event: self.run_all_steps())
        self.root.bind("<Control-l>", lambda event: self.clear_logs())

        # Add these new attributes
        self.running_steps = {}  # Track multiple running steps
        self.step_processes = {}  # Track processes for each step
        self.max_concurrent = 3  # Maximum number of concurrent steps
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        self.step_queue = queue.Queue()

        # Initialize streak frames
        self.home_streak_frame = None
        self.away_streak_frame = None
        self.team_streak_frame = None  
        self.game_streak_frame = None

        # Add responsive layout management
        self.layout_manager = ResponsiveLayout(root)
    def view_all_innings_predictions(self):
        """Display predictions for all innings in a similar format to NRFI predictions"""
        # Use config for all innings predictions file path - adjust as needed based on your structure
        predictions_path = os.path.join("data", "All_Innings", "all_innings_predictions.csv")
        
        if not os.path.exists(predictions_path):
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, "\n[ERROR] No all innings predictions file found. Run the All Innings analysis first.\n")
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)
            return
            
        pred_window = tk.Toplevel(self.root)
        pred_window.title("All Innings Predictions")
        pred_window.geometry("1200x600")
        pred_window.configure(bg=self.bg_color)
        
        header_frame = tk.Frame(pred_window, bg=self.bg_color, pady=10)
        header_frame.pack(fill=tk.X)
        
        tk.Label(
            header_frame,
            text="All Innings Predictions",
            font=STYLES['heading']['font'],
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=10)
        
        info_frame = tk.Frame(pred_window, bg=self.bg_color)
        info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Label(
            info_frame,
            text="• Green: High confidence No Runs prediction for inning",
            font=STYLES['text']['font'],
            bg=self.bg_color,
            fg=COLORS['text_secondary'],
            anchor="w"
        ).pack(fill=tk.X, padx=10)
        
        tk.Label(
            info_frame,
            text="• Red: High confidence Yes Runs prediction for inning",
            font=STYLES['text']['font'],
            bg=self.bg_color,
            fg=COLORS['text_secondary'],
            anchor="w"
        ).pack(fill=tk.X, padx=10)
        
        # Create a frame for the treeview
        tree_frame = tk.Frame(pred_window, bg=self.bg_color)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ("game", "inning", "no_runs_prob", "confidence", "prediction")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)
        scrollbar.config(command=tree.yview)
        
        # Define column headings
        tree.heading("game", text="Game")
        tree.heading("inning", text="Inning")
        tree.heading("no_runs_prob", text="No Runs Probability")
        tree.heading("confidence", text="Confidence")
        tree.heading("prediction", text="Prediction")
        
        # Define column widths
        tree.column("game", width=300)
        tree.column("inning", width=60, anchor="center")
        tree.column("no_runs_prob", width=120, anchor="center")
        tree.column("confidence", width=100, anchor="center")
        tree.column("prediction", width=100, anchor="center")
        
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Function to load and display predictions
        def load_predictions(event=None):
         # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            
            try:
                # Load predictions from CSV
                df = pd.read_csv(predictions_path)
                
                # Process and display each prediction
                for index, row in df.iterrows():
                    # Extract relevant data - adjust column names as needed
                    game_str = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"
                    inning = row.get('inning', 'N/A')
                    no_runs_prob = row.get('no_runs_probability', 0)
                    confidence = row.get('confidence', 0)
                    prediction = "No Runs" if no_runs_prob > 0.5 else "Yes Runs"
                    
                    # Format values
                    no_runs_prob_str = f"{no_runs_prob:.1%}"
                    confidence_str = f"{confidence:.1%}"
                    
                    # Determine row color based on prediction and confidence
                    if prediction == "No Runs" and confidence > 0.6:
                        tag = "high_no_runs"
                    elif prediction == "No Runs" and confidence > 0.3:
                        tag = "medium_no_runs"
                    elif prediction == "Yes Runs" and confidence > 0.6:
                        tag = "high_yes_runs"
                    elif prediction == "Yes Runs" and confidence > 0.3:
                        tag = "medium_yes_runs"
                    else:
                        tag = "neutral"
                    
                    # Insert row with appropriate tag
                    tree.insert("", tk.END, values=(game_str, inning, no_runs_prob_str, confidence_str, prediction), tags=(tag,))
                
                # Configure row colors
                tree.tag_configure("high_no_runs", background="#144a14")
                tree.tag_configure("medium_no_runs", background="#1f6e1f")
                tree.tag_configure("high_yes_runs", background="#661414")
                tree.tag_configure("medium_yes_runs", background="#8b1e1e")
                tree.tag_configure("neutral", background=COLORS['bg_dark'])
                
            except Exception as e:
                error_label = tk.Label(
                    tree_frame,
                    text=f"Error loading predictions: {e}",
                    bg=self.bg_color,
                    fg=COLORS['accent_error'],
                    font=STYLES['text']['font']
                )
                error_label.pack(pady=20)
        
        # Load predictions initially
        load_predictions()
        
        # Add refresh button
        tk.Button(
            pred_window,
            text="Refresh",
            command=load_predictions,
            bg=self.button_color,
            fg=self.button_text_color,
            cursor="hand2"
        ).pack(pady=10)
        
    def create_header_frame(self):
        """Create header frame with title, mode selection, and streak display"""
        header_frame = tk.LabelFrame(
            self.root,
            text="Configuration",
            font=STYLES['heading']['font'],
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary'],
            padx=10,
            pady=10
        )
        header_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
        
        # Configure header frame grid
        for i in range(3):
            header_frame.grid_columnconfigure(i, weight=1)
            
        # Create sections with proper weights
        left_section = self.create_mode_section(header_frame)
        left_section.grid(row=0, column=0, sticky="nsew", padx=5)
        
        center_section = self.create_control_section(header_frame)
        center_section.grid(row=0, column=1, sticky="nsew", padx=5)
        
        right_section = self.create_streak_section(header_frame)
        right_section.grid(row=0, column=2, sticky="nsew", padx=5)

    def create_mode_section(self, parent):
        """Create mode selection section"""
        section = tk.Frame(parent, bg=COLORS['bg_content'])
        
        # Title and subtitle
        self.title_label = tk.Label(
            section,
            text="No Runs First Inning Analysis",
            font=STYLES['heading']['font'],
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary']
        )
        self.title_label.pack(anchor="w", fill=tk.X)

        self.subtitle_label = tk.Label(
            section,
            text="Predict likelihood of scoreless first innings",
            font=STYLES['text']['font'],
            bg=COLORS['bg_content'],
            fg=COLORS['text_secondary']
        )
        self.subtitle_label.pack(anchor="w", fill=tk.X, pady=(0, 10))        # Mode selection with dropdown instead of radio buttons
        mode_frame = tk.Frame(section, bg=COLORS['bg_content'])
        mode_frame.pack(anchor="w", fill=tk.X, pady=5)
        
        tk.Label(
            mode_frame,
            text="Analysis Mode:",
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary'],
            font=STYLES['text']['font']
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        # Create a list of mode names for the dropdown
        mode_names = [name for _, name in ANALYSIS_MODES]
        mode_values = [code for code, _ in ANALYSIS_MODES]
        
        # Create the combobox
        self.mode_dropdown = ttk.Combobox(
            mode_frame,
            textvariable=self.analysis_mode,
            values=mode_values,
            state="readonly",
            width=20
        )
        self.mode_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Set the display text for the combobox
        self.mode_dropdown['values'] = mode_names
        
        # Bind the dropdown selection event to change_analysis_mode
        self.mode_dropdown.bind("<<ComboboxSelected>>", lambda e: self.change_analysis_mode())

        # Season input
        season_frame = tk.Frame(section, bg=COLORS['bg_content'], pady=5)
        season_frame.pack(anchor="w", fill=tk.X)

        tk.Label(
            season_frame,
            text="Start Season:",
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT)

        self.season_var = tk.StringVar(value=str(datetime.now().year))
        season_entry = tk.Entry(
            season_frame,
            textvariable=self.season_var,
            width=6
        )
        season_entry.pack(side=tk.LEFT, padx=5)

        return section

    def create_control_section(self, parent):
        """Create control section with prediction buttons"""
        section = tk.Frame(parent, bg=COLORS['bg_content'])
        
        # Add prediction buttons section
        prediction_frame = tk.Frame(section, bg=COLORS['bg_content'], pady=5)
        prediction_frame.pack(anchor="w", fill=tk.X)
        
        # NRFI Predictions button
        nrfi_pred_btn = tk.Button(
            prediction_frame,
            text="View NRFI Predictions",
            command=self.view_predictions,
            bg="#2a4494",  # Distinctive blue
            fg=self.button_text_color,
            cursor="hand2"
        )
        nrfi_pred_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(nrfi_pred_btn, "View today's NRFI predictions and analysis")
          # Game Predictions button (NEW)
        game_pred_btn = tk.Button(
            prediction_frame,
            text="View Game Predictions",
            command=self.view_game_predictions,
            bg="#2a6b94",  # Distinctive light blue
            fg=self.button_text_color,
            cursor="hand2"
        )
        game_pred_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(game_pred_btn, "View sports game predictions from moneyline model")
        
        
        # All Innings Predictions button
        all_innings_btn = tk.Button(
            prediction_frame,
            text="View All Innings Predictions",
            command=self.view_all_innings_predictions,
            bg="#2a7a94",  # Distinctive blue-green
            fg=self.button_text_color,
            cursor="hand2"
        )
        all_innings_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(all_innings_btn, "View inning-by-inning analysis predictions")
        
        # Props Predictions button
        props_pred_btn = tk.Button(
            prediction_frame,
            text="View Props Predictions",
            command=self.view_props,
            bg="#4a2a94",  # Distinctive purple
            fg=self.button_text_color,
            cursor="hand2"
        )
        props_pred_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(props_pred_btn, "View player props recommendations")
        
        # Backtesting Results button
        backtest_btn = tk.Button(
            prediction_frame,
            text="Backtesting Results",
            command=self.view_backtesting_results,
            bg="#2a944a",  # Distinctive green
            fg=self.button_text_color,
            cursor="hand2"
        )
        backtest_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(backtest_btn, "View model performance and historical results")

        return section

    def create_streak_section(self, parent):
        """Create streak display section"""
        section = tk.Frame(parent, bg=COLORS['bg_content'])
        
        # Add streak window controls
        streak_controls = tk.Frame(section, bg=COLORS['bg_content'])
        streak_controls.pack(fill=tk.X, pady=5)
        
        tk.Label(
            streak_controls,
            text="Streak Window:",
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)
        
        self.streak_window = tk.IntVar(value=10)
        for window_size, display_text in STREAK_WINDOWS:
            rb = tk.Radiobutton(
                streak_controls,
                text=display_text,
                variable=self.streak_window,
                value=window_size,
                command=self.update_streak_display,
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary'],
                selectcolor="black"
            )
            rb.pack(side=tk.LEFT, padx=5)
            
            # Add tooltip
            ToolTip(rb, f"Show {display_text} games")

        # Add display frames for streaks
        self.team_streak_display = tk.Frame(section, bg=COLORS['bg_content'])
        self.team_streak_display.pack(fill=tk.X, pady=2)
        
        self.game_streak_display = tk.Frame(section, bg=COLORS['bg_content'])
        self.game_streak_display.pack(fill=tk.X, pady=2)

        return section

    def load_playlist(self):
        """Load M3U playlist and populate stream selection"""
        playlist_path = os.path.join("NRFI", "Player", "playlist.m3u")
        try:
            with open(playlist_path, 'r', encoding='utf-8') as f:
                content = f.readlines()

            streams = []
            current_title = None
            
            for line in content:
                line = line.strip()
                if line.startswith('#EXTINF:'):
                    # Extract title from EXTINF line
                    if 'tvg-name="' in line:
                        # Parse tvg-name attribute
                        match = re.search(r'tvg-name="([^"]*)"', line)
                        if match:
                            current_title = match.group(1)
                    elif ',' in line:
                        # Fallback to text after comma
                        current_title = line.split(',', 1)[1]
                elif line.startswith('http'):
                    if current_title:
                        streams.append((current_title, line))
                        current_title = None

            # Update combobox values
            self.stream_select['values'] = [s[0] for s in streams]
            self.streams = dict(streams)
            
            # Select first stream
            if streams:
                self.stream_select.set(streams[0][0])
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, f"[INFO] Loaded {len(streams)} streams\n")
                self.log_text.configure(state='disabled')
                self.log_text.see(tk.END)
                
        except Exception as e:
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, f"[ERROR] Failed to load playlist: {e}\n")
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)

    def on_stream_selected(self, event=None):
        """Handle stream selection from dropdown"""
        if not self.secure_connection_established:
            return
            
        selected = self.stream_var.get()
        if selected in self.streams:
            try:
                # If using full M3UPlayer
                if hasattr(self.player, 'channels'):
                    # Find the channel index
                    for i, channel in enumerate(self.player.channels):
                        if channel.get('name') == selected:
                            # Update player's current channel
                            self.player.current_channel_index = i
                            self.player.channel_list.selection_clear(0, tk.END)
                            self.player.channel_list.selection_set(i)
                            self.player.channel_list.see(i)
                            # Play the selected channel
                            self.player.play_selected()
                            break

                # Update interface
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, f"[INFO] Playing stream: {selected}\n")
                self.log_text.configure(state='disabled')
                self.log_text.see(tk.END)
                
                # Update play button text
                self.play_button.config(text="Pause")
                self.is_playing = True
                
            except Exception as e:
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, f"[ERROR] Failed to select stream: {e}\n")
                self.log_text.configure(state='disabled')
                self.log_text.see(tk.END)


    def delete_source_files(self):
        """Delete source CSV files to start fresh"""
        csv_paths = [
            "data/historical_data.csv",
            "data/upcoming_games.csv",
            os.path.join("data", "First Inning NRFI", "F1_predictions.csv"),
            "data/nrfi_predictions.csv"  # Keep both paths for backward compatibility
        ]
        
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete all source CSV files? This action cannot be undone."):

            deleted = False
            for path in csv_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        deleted = True
                        print(f"[INFO] Deleted {path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to delete {path}: {e}")
            
            if deleted:
                messagebox.showinfo("Success", "Source files deleted successfully. You can now run the pipeline for fresh data.")
            else:
                            messagebox.showinfo("Info", "No source files found to delete.")
                            
    def create_content_frame(self):
        """Create frame for steps and graph display"""
        self.content_frame = tk.Frame(self.root, bg=COLORS['bg_content'])
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Configure content frame grid for steps and graphs
        self.content_frame.grid_columnconfigure(0, weight=1)  # Steps column
        self.content_frame.grid_columnconfigure(1, weight=3)  # Graph column
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Create steps section with minimum width
        steps_frame = self.create_steps_section(self.content_frame)
        steps_frame.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        
        # Create graph section that expands
        graph_frame = self.create_graph_section(self.content_frame)
        graph_frame.grid(row=0, column=1, sticky="nsew")
        self.content_frame = tk.Frame(self.root, bg=COLORS['bg_content'])
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Configure content frame grid for steps and graphs
        self.content_frame.grid_columnconfigure(0, weight=1)  # Steps column
        self.content_frame.grid_columnconfigure(1, weight=3)  # Graph column
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Create steps section with minimum width
        steps_frame = self.create_steps_section(self.content_frame)
        steps_frame.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        
        # Create graph section that expands
        graph_frame = self.create_graph_section(self.content_frame)
        graph_frame.grid(row=0, column=1, sticky="nsew")

    def create_steps_section(self, parent):
        """Create steps section"""
        steps_frame = tk.LabelFrame(
            parent,
            text="Analysis Pipeline",
            font=STYLES['heading']['font'],
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary'],
            padx=10,
            pady=5
        )
        steps_frame.grid_rowconfigure(0, weight=1)
        
        # Update steps based on current mode
        self.update_steps_for_mode()
        
        # Create step buttons
        for i, step in enumerate(self.steps):
            step_frame = tk.Frame(steps_frame, bg=COLORS['bg_content'])
            step_frame.grid(row=i, column=0, sticky="ew", pady=2)
            
            # Status label
            step["status_label"] = tk.Label(
                step_frame,
                text="🟡 Ready",
                bg=COLORS['bg_content'],
                fg=COLORS['text_secondary']
            )
            step["status_label"].grid(row=0, column=0, padx=5, sticky="w")
            
            # Step button
            step["button"] = tk.Button(
                step_frame,
                text="Run",
                command=lambda s=step: self.run_step(s),
                bg=self.button_color,
                fg=self.button_text_color,
                cursor="hand2"
            )
            step["button"].grid(row=0, column=2, padx=5, sticky="e")
            
            # Progress bar
            step["progress_bar"] = ttk.Progressbar(
                step_frame,
                mode='indeterminate',
                length=200
            )
            step["progress_bar"].grid(row=0, column=1, padx=5, sticky="ew")
            
            # Step name
            # Put step name in the same row to reduce vertical spacing
            tk.Label(
                step_frame,
                text=step["name"],
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary']
            ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 2))
            
            # Configure grid weights
            step_frame.grid_columnconfigure(1, weight=1)
            
            # Make sure all rows in step_frame have consistent weights
            step_frame.grid_rowconfigure(0, weight=0)
            step_frame.grid_rowconfigure(1, weight=0)
        
        # Add Run All button at bottom of steps frame
        run_all_frame = tk.Frame(steps_frame, bg=COLORS['bg_content'])
        run_all_frame.grid(row=len(self.steps), column=0, sticky="ew", pady=5)
        
        tk.Button(
            run_all_frame,
            text="Run All Steps",
            command=self.run_all_steps,
            bg=self.button_color,
            fg=self.button_text_color,
            cursor="hand2"
        ).grid(row=0, column=0, sticky="e", padx=5)

        return steps_frame

    def create_graph_section(self, parent):
        """Create graph section"""
        graph_frame = tk.LabelFrame(
            parent,
            text="Graph Viewer",
            font=STYLES['heading']['font'],
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary'],
            padx=10,
            pady=10
        )
        graph_frame.grid_rowconfigure(1, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)
        
        # Controls frame for view mode and selection
        controls_frame = tk.Frame(graph_frame, bg=COLORS['bg_content'])
        controls_frame.grid(row=0, column=0, sticky="ew", pady=5)
        controls_frame.grid_columnconfigure(0, weight=1)
        
        # Mode selection
        mode_frame = tk.Frame(controls_frame, bg=COLORS['bg_content'])
        mode_frame.grid(row=0, column=0, sticky="ew", pady=5)
        
        self.view_mode = tk.StringVar(value="team")
        col = 0
        for mode, text in [("team", "Team View"), ("upcoming", "Upcoming Games"), ("model", "Model Evaluation")]:
            tk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.view_mode,
                value=mode,
                command=self.update_dropdown,
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary'],
                selectcolor="black"
            ).grid(row=0, column=col, padx=10)
            col += 1

        # Selection dropdown
        dropdown_frame = tk.Frame(controls_frame, bg=COLORS['bg_content'])
        dropdown_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        tk.Label(
            dropdown_frame,
            text="Select:",
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary']
        ).grid(row=0, column=0, padx=5)

        self.selection_dropdown = ttk.Combobox(dropdown_frame, state="readonly", width=30)
        self.selection_dropdown.grid(row=0, column=1, padx=5)
        self.selection_dropdown.bind("<<ComboboxSelected>>", self.update_graphs)

        # Add graph selection buttons
        self.create_graph_buttons(controls_frame)

        # Create scrollable canvas for graph display
        canvas_frame = tk.Frame(graph_frame, bg=COLORS['bg_content'])
        canvas_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)

        # Create canvas with scrollbar
        self.graph_canvas = tk.Canvas(
            canvas_frame,
            bg=COLORS['bg_content'],
            highlightthickness=0
        )
        self.graph_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.graph_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure canvas
        self.graph_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create frame for graph content
        self.graph_display = tk.Frame(self.graph_canvas, bg=COLORS['bg_content'])
        self.graph_display.bind("<Configure>", self._on_frame_configure)
        
        # Create window in canvas
        self.canvas_window = self.graph_canvas.create_window(
            (0, 0),
            window=self.graph_display,
            anchor="nw"
        )
        
        # Bind canvas resize
        self.graph_canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Initialize dropdown
        self.update_dropdown()

        return graph_frame

    def _on_frame_configure(self, event):
        """Update scroll region when the frame changes size"""
        # Update the scroll region to encompass the inner frame
        self.graph_canvas.configure(scrollregion=self.graph_canvas.bbox("all"))
        
    def _on_canvas_configure(self, event):
        """Handle canvas resize events"""
        # Update the width of the window and frame when canvas is resized
        width = event.width - 4  # Subtract a small amount for borders
        # Update the canvas window width
        self.graph_canvas.itemconfig(self.canvas_window, width=width)
        # Force the display frame to match the canvas width
        self.graph_display.configure(width=width)

    def resize_image(self, img, event_width):
        """Resize image while maintaining aspect ratio based on available width"""
        # Get the original image dimensions
        orig_width = img.size[0]
        orig_height = img.size[1]
        
        # Calculate new dimensions based on available width
        # Leave some padding (40px) for scrollbar and borders
        available_width = event_width - 40
        
        # Only resize if the image is larger than available space
        if orig_width > available_width:
            # Calculate new height maintaining aspect ratio
            ratio = available_width / orig_width
            new_height = int(orig_height * ratio)
            return img.resize((int(available_width), new_height), Image.Resampling.LANCZOS)
        return img

    def update_dropdown(self):
        """Update dropdown options and graph buttons based on selected mode"""
        mode = self.analysis_mode.get()
        
        if self.view_mode.get() == "model":
            # Set dropdown for model evaluation
            self.selection_dropdown['values'] = ["Model Performance"]
            self.selection_dropdown.set("Model Performance")
            self.update_graphs()
            return
            
        elif self.view_mode.get() == "team":
            # Load team list from the teams folder
            mode_paths = get_mode_paths(mode)
            if "teams_dir" not in mode_paths:
                messagebox.showerror("Error", f"No teams directory configured for mode '{mode}'")
                return
            teams_path = mode_paths["teams_dir"]
            teams = []
            
            # Use correct prefix based on analysis mode
            prefix = f"{mode}_" if mode != "NRFI" else ""
            
            if os.path.exists(teams_path):
                for file in os.listdir(teams_path):
                    # Filter files by the current mode prefix
                    if file.startswith(prefix) and file.endswith("_NRFI_Analysis.png"):
                        team_name = file.replace(f"{prefix}", "").replace("_NRFI_Analysis.png", "").replace("_", " ")
                        if team_name not in teams:
                            teams.append(team_name)
                teams.sort()
                self.selection_dropdown['values'] = teams
                if teams:
                    self.selection_dropdown.set(teams[0])
            else:
                # Fall back to direct teams folder if data/teams doesn't exist
                teams_path = "teams"
                if os.path.exists(teams_path):
                    for file in os.listdir(teams_path):
                        # Filter files by the current mode prefix
                        if file.startswith(prefix) and file.endswith("_NRFI_Analysis.png"):
                            team_name = file.replace(f"{prefix}", "").replace("_NRFI_Analysis.png", "").replace("_", " ")
                            if team_name not in teams:
                                teams.append(team_name)
                    teams.sort()
                    self.selection_dropdown['values'] = teams
                    if teams:
                        self.selection_dropdown.set(teams[0])
                else:
                    # Silently handle missing teams directory - no error dialog
                    print(f"[INFO] Teams directory not found at {teams_path}")
                    self.selection_dropdown['values'] = []
        else:
            # Handle upcoming games selection
            upcoming_path = get_mode_paths(mode)["upcoming_dir"]
            if not os.path.exists(upcoming_path):
                upcoming_path = os.path.join("data", mode, "visuals", "upcoming")  # Try alternative path
                
            if os.path.exists(upcoming_path):
                games = []
                for folder in os.listdir(upcoming_path):
                    if os.path.isdir(os.path.join(upcoming_path, folder)):
                        # Updated to handle F1_ prefix
                        parts = folder.split("_")
                        if len(parts) >= 4:
                            # Skip the F1_ prefix if it exists
                            if parts[0] == "F1":
                                parts = parts[1:]  # Remove F1_ prefix for display
                            
                            game_date = parts[0]
                            away_team = " ".join(parts[1:-2])  # Handle team names with spaces
                            home_team = " ".join(parts[-2:]).replace("_at_", "")
                            
                            try:
                                formatted_date = datetime.strptime(game_date, "%Y-%m-%d").strftime("%B %d, 2023")
                                display_name = f"{formatted_date}: {away_team} @ {home_team}"
                                games.append((folder, display_name))
                            except ValueError:
                                # Skip folders with invalid date format
                                continue
                
                games.sort(key=lambda x: x[1])  # Sort by display name
                self.matchup_folders = {display: folder for folder, display in games}
                display_names = [display for _, display in games]
                self.selection_dropdown['values'] = display_names
                if display_names:
                    self.selection_dropdown.set(display_names[0])
                    self.update_graphs()  # Automatically show graphs for first selection            else:
                # Silently handle missing upcoming games directory - no error dialog
                print(f"[INFO] Upcoming games directory not found at {upcoming_path}")
                self.selection_dropdown['values'] = []

        # Update visible graph buttons
        for btn in self.graph_buttons.values():
            btn.pack_forget()
        
        if self.view_mode.get() == "team":
            col = 0
            for graph_type, _ in TEAM_GRAPH_TYPES:
                if graph_type in self.graph_buttons:
                    self.graph_buttons[graph_type].grid(row=0, column=col, padx=2)
                    col += 1
        elif self.view_mode.get() == "upcoming":
            col = 0
            for graph_type, _ in MATCHUP_GRAPH_TYPES:
                if graph_type in self.graph_buttons:
                    self.graph_buttons[graph_type].grid(row=0, column=col, padx=2)
                    col += 1
        else:  # Model evaluation mode
            col = 0
            for graph_type, _ in MODEL_EVAL_GRAPHS:
                if graph_type in self.graph_buttons:
                                    self.graph_buttons[graph_type].grid(row=0, column=col, padx=2)
                                    
    def update_graphs(self, event=None):
        """Update displayed graphs based on current selection"""
        selection = self.selection_dropdown.get()
        if selection:
            # Clear previous graphs
            for widget in self.graph_display.winfo_children():
                widget.destroy()
            
            # Get the current canvas width for proper image sizing
            canvas_width = self.graph_canvas.winfo_width()
            mode = self.analysis_mode.get()
            mode_config = get_mode_config(mode)
            
            # Track loaded graphs for debugging
            loaded_graphs = []
            failed_graphs = []
            
            # Get the correct directories for the selected mode
            mode_dirs = self.get_mode_directories(mode)
            
            if self.view_mode.get() == "model":
                # Get model evaluation graph configurations from the config
                model_graphs = mode_config.get("graph_types", {}).get("model", [])
                
                for graph_config in model_graphs:
                    graph_file = graph_config["file"]
                    title = graph_config["display"]
                    
                    # Find the correct path for the selected mode and graph
                    graph_path = self.find_graph_path(mode, graph_file, "model", mode_dirs)
                    
                    self._add_graph_to_display(graph_path, title, canvas_width, loaded_graphs, failed_graphs)
                    
            elif self.view_mode.get() == "team":
                formatted_team = selection.replace(" ", "_")
                
                # Get team graph configurations from the config
                team_graphs = mode_config.get("graph_types", {}).get("team", [])
                
                for graph_config in team_graphs:
                    graph_type = graph_config["file"]
                    title = graph_config["display"]
                    
                    # Find the correct path for the selected mode, team, and graph
                    graph_path = self.find_graph_path(mode, graph_type, "team", mode_dirs, team=formatted_team)
                    
                    self._add_graph_to_display(graph_path, title, canvas_width, loaded_graphs, failed_graphs)
            
            else:  # Upcoming games view
                folder_name = self.matchup_folders.get(selection)
                if folder_name:
                    # Find the correct matchup directory
                    matchup_dir = self.find_matchup_directory(mode, folder_name, mode_dirs)
                    
                    if matchup_dir and os.path.exists(matchup_dir):
                        # Get matchup graph configurations from the config
                        matchup_graphs = mode_config.get("graph_types", {}).get("matchup", [])
                        
                        for graph_config in matchup_graphs:
                            graph_type = graph_config["file"]
                            title = graph_config["display"]
                            
                            # Find the correct path for the selected mode, matchup, and graph
                            graph_path = self.find_graph_path(mode, graph_type, "matchup", mode_dirs, 
                                                             matchup=folder_name, matchup_dir=matchup_dir)
                            
                            self._add_graph_to_display(graph_path, title, canvas_width, loaded_graphs, failed_graphs)
                    else:
                        # Display a message when no matchup data is found
                        self.display_no_data_message(selection, mode_dirs, folder_name)
            
            # Display a summary if in debug mode
            if DEBUG_MODE and (failed_graphs or len(loaded_graphs) == 0):
                self.display_debug_info(selection, mode, loaded_graphs, failed_graphs)
    
    def get_mode_directories(self, mode):
        """Get all relevant directories for the selected mode"""
        mode_config = get_mode_config(mode)
        mode_paths = get_mode_paths(mode)
        visuals_dirs = get_visuals_dirs(mode)
        
        return {
            "data_dir": mode_config.get("data_dir", ""),
            "visuals_dir": mode_config.get("visuals_dir", ""),
            "teams_dir": mode_paths.get("teams_dir", ""),
            "upcoming_dir": mode_paths.get("upcoming_dir", ""),
            "plots_dir": PLOTS_DIR,
            "visuals_teams_dir": visuals_dirs.get("teams_dir", ""),
            "visuals_upcoming_dir": visuals_dirs.get("upcoming_dir", "")
        }
    
    def find_graph_path(self, mode, graph_file, view_type, mode_dirs, team=None, matchup=None, matchup_dir=None):
        """Find the correct path for a graph based on mode, view type, and other parameters"""
        possible_paths = []
        
        # Format the graph file name based on mode and type
        if team and "{team}" in graph_file:
            graph_filename = graph_file.format(team=team)
        else:
            graph_filename = graph_file
            
        # Mode-specific prefixes
        prefix = ""
        if mode == "NRFI" and view_type in ["team", "matchup"] and not graph_file.startswith("F1_"):
            prefix = "F1_"
        elif mode == "ALL_INNINGS":
            for phase in ["F3", "F5", "F9"]:
                if phase in graph_file:
                    prefix = f"{phase}_"
                    break
                    
        # Build different possible paths based on mode and view type
        if view_type == "model":
            # Model graphs could be in several locations
            possible_paths = [
                os.path.join(mode_dirs["visuals_dir"], graph_filename),
                os.path.join(mode_dirs["data_dir"], "visuals", graph_filename),
                os.path.join(mode_dirs["plots_dir"], graph_filename)
            ]
            
            # Special case for AOI_SPORTS league-specific model files
            if mode == "AOI_SPORTS":
                for league in ["mlb", "nba", "nfl", "nhl", "epl", "laliga", "bundesliga", "seriea"]:
                    if league in graph_filename.lower():
                        possible_paths.insert(0, os.path.join(mode_dirs["plots_dir"], graph_filename))
                        break
                        
        elif view_type == "team" and team:
            # Team graphs
            if mode == "PLAYER_PROPS":
                # Player props has a specific structure
                possible_paths = [
                    os.path.join(mode_dirs["data_dir"], "visuals", "teams", f"{prefix}{team}{graph_filename}"),
                    os.path.join(mode_dirs["teams_dir"], f"{prefix}{team}{graph_filename}")
                ]
            elif mode == "ALL_INNINGS" and "{team}" in graph_file:
                # ALL_INNINGS has a specific format for team files
                possible_paths = [
                    os.path.join(mode_dirs["visuals_teams_dir"], graph_filename),
                    os.path.join(mode_dirs["teams_dir"], graph_filename)
                ]
            else:
                # Standard team graph paths
                possible_paths = [
                    os.path.join(mode_dirs["teams_dir"], f"{prefix}{team}{graph_filename}"),
                    os.path.join(mode_dirs["visuals_teams_dir"], f"{prefix}{team}{graph_filename}")
                ]
                
        elif view_type == "matchup" and matchup_dir:
            # Matchup graphs
            if mode == "ALL_INNINGS" and "{team}" in graph_file:
                # Extract team name from matchup folder name
                parts = matchup.split("_")
                team_name = parts[1] if len(parts) >= 3 else ""
                possible_paths = [
                    os.path.join(matchup_dir, graph_file.format(team=team_name)),
                    os.path.join(matchup_dir, graph_filename)
                ]
            else:
                # Standard matchup graph paths
                possible_paths = [
                    os.path.join(matchup_dir, f"{prefix}{graph_filename}"),
                    os.path.join(matchup_dir, graph_filename)
                ]
                
        # Return the first path that exists, or the first path if none exist
        return self._find_existing_path(possible_paths)
    
    def find_matchup_directory(self, mode, folder_name, mode_dirs):
        """Find the correct matchup directory"""
        # Try multiple potential locations for the upcoming directory
        matchup_dirs = [
            os.path.join(mode_dirs["upcoming_dir"], folder_name),
            os.path.join(mode_dirs["visuals_upcoming_dir"], folder_name),
            os.path.join(mode_dirs["data_dir"], "visuals", "upcoming", folder_name)
        ]
        
        # Return the first directory that exists
        for path in matchup_dirs:
            if os.path.exists(path):
                return path
                
        return None
    
    def display_no_data_message(self, selection, mode_dirs, folder_name):
        """Display a message when no matchup data is found"""
        # Create a list of directories that were checked
        matchup_dirs = [
            os.path.join(mode_dirs["upcoming_dir"], folder_name),
            os.path.join(mode_dirs["visuals_upcoming_dir"], folder_name),
            os.path.join(mode_dirs["data_dir"], "visuals", "upcoming", folder_name)
        ]
        
        no_data_frame = tk.Frame(self.graph_display, bg=COLORS['bg_content'])
        no_data_frame.pack(pady=20, fill=tk.X)
        
        no_data_label = tk.Label(
            no_data_frame,
            text=f"No matchup data found for {selection}",
            font=STYLES['heading']['font'],
            bg=COLORS['bg_content'],
            fg=COLORS['accent_warning']
        )
        no_data_label.pack(pady=10)
        
        # List all the directories that were checked
        paths_label = tk.Label(
            no_data_frame,
            text=f"Checked directories:\n" + "\n".join(matchup_dirs),
            font=STYLES['text']['font'],
            bg=COLORS['bg_content'],
            fg=COLORS['text_secondary'],
            justify=tk.LEFT
        )
        paths_label.pack(pady=5)
    
    def display_debug_info(self, selection, mode, loaded_graphs, failed_graphs):
        """Display debug information about loaded and failed graphs"""
        debug_frame = tk.Frame(self.graph_display, bg=COLORS['bg_content'])
        debug_frame.pack(pady=20, fill=tk.X)
        
        if failed_graphs:
            failed_label = tk.Label(
                debug_frame,
                text=f"Failed to load {len(failed_graphs)} graphs:\n" + "\n".join(failed_graphs[:5]),
                font=STYLES['text']['font'],
                bg=COLORS['bg_content'],
                fg=COLORS['accent_warning'],
                justify=tk.LEFT
            )
            failed_label.pack(pady=5, anchor=tk.W)
        
        if len(loaded_graphs) == 0:
            no_graphs_label = tk.Label(
                debug_frame,
                text=f"No graphs were found for {selection} in {mode} mode",
                font=STYLES['text']['font'],
                bg=COLORS['bg_content'],
                fg=COLORS['accent_warning']
            )
            no_graphs_label.pack(pady=5)

    def _find_existing_path(self, paths):
        """Helper function to find the first existing path from a list of possible paths"""
        for path in paths:
            if os.path.exists(path):
                return path
        return paths[0]  # Return the first path even if it doesn't exist
    
    def _add_graph_to_display(self, graph_path, title, canvas_width, loaded_graphs, failed_graphs):
        """Helper function to add a graph to the display"""
        if os.path.exists(graph_path):
            try:
                graph_frame = tk.Frame(self.graph_display, bg=COLORS['bg_content'])
                graph_frame.pack(pady=10, fill=tk.X)
                
                title_label = tk.Label(
                    graph_frame,
                    text=title,
                    font=STYLES['heading']['font'],
                    bg=COLORS['bg_content'],
                    fg=COLORS['text_primary']
                )
                title_label.pack(pady=(0, 5))
                
                # Load and resize image based on canvas width
                img = Image.open(graph_path)
                img = self.resize_image(img, canvas_width)
                
                photo = ImageTk.PhotoImage(img)
                label = ttk.Label(graph_frame, image=photo)
                label.image = photo  # Keep a reference to prevent garbage collection
                label.pack()
                
                # Store original path for future resize operations
                label.original_img_path = graph_path
                
                # Bind resize event to the label with throttling
                self.graph_canvas.bind("<Configure>", 
                    lambda e, img_path=graph_path, lbl=label: self.on_window_resize(e, img_path, lbl))
                
                loaded_graphs.append(f"{title} ({os.path.basename(graph_path)})")
                
            except Exception as e:
                error_label = tk.Label(
                    graph_frame,
                    text=f"Failed to load {os.path.basename(graph_path)}: {e}",
                    fg=COLORS['accent_error'],
                    bg=COLORS['bg_content'],
                    font=STYLES['text']['font']
                )
                error_label.pack(pady=10)
                failed_graphs.append(f"{os.path.basename(graph_path)}: {e}")
        else:
            failed_graphs.append(f"File not found: {graph_path}")    
    def on_window_resize(self, event, img_path, label):
        """Handle window resize events by updating image sizes"""
        try:
            # Check if a resize operation is already in progress and cancel it
            if hasattr(self, '_resize_timer'):
                self.root.after_cancel(self._resize_timer)
            
            # Check if the label still exists and is valid
            try:
                # This will raise TclError if widget doesn't exist anymore
                label.winfo_exists()
                
                # Use the stored original path if available
                path_to_use = getattr(label, 'original_img_path', img_path)
                
                # Define a throttled resize function with widget existence check
                def do_resize():
                    try:
                        # Check again if widget still exists when actually doing resize
                        if label.winfo_exists():
                            # Load original image
                            img = Image.open(path_to_use)
                            # Resize based on new width
                            img = self.resize_image(img, event.width)
                            # Update the label with new image
                            photo = ImageTk.PhotoImage(img)
                            label.configure(image=photo)
                            label.image = photo  # Keep a reference to prevent garbage collection
                    except (tk.TclError, RuntimeError) as widget_err:
                        # Widget was destroyed between scheduling and execution
                        print(f"Widget no longer exists: {widget_err}")
                        # No need to raise, just skip this update
                
                # Schedule resize with a small delay to avoid excessive processing during continuous resize
                self._resize_timer = self.root.after(150, do_resize)
            except (tk.TclError, RuntimeError):
                # Widget was already destroyed, no need to proceed
                print(f"Skipping resize for destroyed widget")
        except Exception as e:
            print(f"Error resizing image: {e}")
            # Try to configure the label, but only if it still exists
            try:
                if label.winfo_exists():
                    label.configure(image='', text=f"Could not load image: {os.path.basename(img_path)}")
            except (tk.TclError, AttributeError):
                # Widget no longer exists, just log the error
                pass

    def clear_graphs(self):
        """Clear all graphs from the display frame"""
        for widget in self.graph_display.winfo_children():
            widget.destroy()

    def show_selected_graph(self, graph_type):
        """Display the selected graph"""
        selection = self.selection_dropdown.get()
        if not selection:
            return
            
        # Clear previous graphs and streaks
        for widget in self.graph_display.winfo_children():
            widget.destroy()
        
        for widget in self.team_streak_display.winfo_children():
            widget.destroy()
        
        for widget in self.game_streak_display.winfo_children():
            widget.destroy()
        
        # Update button states 
        for btn in self.graph_buttons.values():
            btn.configure(relief=tk.RAISED)
        if graph_type in self.graph_buttons:
            self.graph_buttons[graph_type].configure(relief=tk.SUNKEN)
        
        mode = self.analysis_mode.get()
        graph_path = None
        
        if self.view_mode.get() == "team":
            formatted_team = selection.replace(" ", "_")
            teams_dir = get_mode_paths(mode)["teams_dir"]
            prefix = f"{mode}_" if mode != "NRFI" else ""
            graph_path = os.path.join(teams_dir, f"{prefix}{formatted_team}{graph_type}")
            
            # Calculate and display streaks for the selected team
            window = self.streak_window.get()
            today = datetime.now().strftime("%Y-%m-%d")
            team_streak = self.calculate_streaks(selection, today, window, True)
            game_streak = self.calculate_streaks(selection, today, window, False)
            
            # Display team streak
            tk.Label(
                self.team_streak_display,
                text=f"{selection} Team NRFI Streak:",
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary'],
                font=STYLES['text']['font']
            ).pack(side=tk.LEFT, padx=5)
            self.display_streak_boxes(self.team_streak_display, team_streak)
            
            # Display game streak
            tk.Label(
                self.game_streak_display,
                text=f"{selection} Game NRFI Streak:",
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary'],
                font=STYLES['text']['font']
            ).pack(side=tk.LEFT, padx=5)
            self.display_streak_boxes(self.game_streak_display, game_streak)
            
        elif self.view_mode.get() == "upcoming":
            folder_name = self.matchup_folders[selection]
            upcoming_dir = get_mode_paths(mode)["upcoming_dir"]
            matchup_dir = os.path.join(upcoming_dir, folder_name)
            
            # Check if we need to add F1_ prefix for NRFI mode
            if mode == "NRFI" and not graph_type.startswith("F1_"):
                graph_type = f"F1_{graph_type}"
                
            graph_path = os.path.join(matchup_dir, graph_type)
            print(f"Looking for graph at: {graph_path}")  # Debug output
        else:  # Model evaluation mode
            graph_path = os.path.join(get_visuals_dirs(mode)["base_dir"], graph_type)
        
        if os.path.exists(graph_path):
            try:
                img = Image.open(graph_path)
                # Get current canvas width for proper sizing
                canvas_width = self.graph_canvas.winfo_width()
                img = self.resize_image(img, canvas_width)
                
                photo = ImageTk.PhotoImage(img)
                label = ttk.Label(self.graph_display, image=photo)
                label.image = photo  # Keep a reference
                label.pack(pady=10)
                
                # Bind resize event to the label
                self.graph_canvas.bind("<Configure>", 
                    lambda e, img_path=graph_path, lbl=label: self.on_window_resize(e, img_path, lbl))
                
            except Exception as e:
                error_label = tk.Label(
                    self.graph_display,
                    text=f"Failed to load graph: {e}",
                    fg=COLORS['accent_error'],
                    bg=COLORS['bg_content'],
                    font=STYLES['text']['font']
                )
                error_label.pack(pady=10)
        else:
            error_label = tk.Label(
                self.graph_display,
                text=f"Graph not found: {graph_path}",
                fg=COLORS['accent_error'],
                bg=COLORS['bg_content'],
                font=STYLES['text']['font']
            )
            error_label.pack(pady=10)

    def create_log_frame(self):
        """Create frame for process output logs"""
        log_frame = tk.LabelFrame(
            self.root, 
            text="Process Output", 
            font=STYLES['heading']['font'],
            bg=COLORS['bg_content'],
            padx=10, 
            pady=5,
            fg=COLORS['text_primary']
        )
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        
        # Configure log frame grid
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        
        # Create controls with proper spacing
        controls_frame = tk.Frame(log_frame, bg=COLORS['bg_content'])
        controls_frame.grid(row=0, column=0, sticky="ew", pady=5)
        
        # Clear Log Button at top-right in the log frame
        button_frame = tk.Frame(controls_frame, bg=COLORS['bg_content'])
        button_frame.pack(side=tk.RIGHT)
        
        clear_button = tk.Button(
            button_frame, 
            text="Clear Log", 
            command=self.clear_logs,
            bg=self.button_color, 
            fg=self.button_text_color, 
            relief=tk.FLAT, 
            cursor="hand2"
        )
        clear_button.pack(side=tk.RIGHT)
        ToolTip(clear_button, "Clear the process output log (Ctrl+L)")
        
        # Create scrolled text with proper expanding
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            bg=COLORS['bg_dark'],
            fg=COLORS['text_primary'],
            font=('Courier', 9),
            wrap=tk.WORD,
            height=8
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.log_text.configure(state='disabled')
        
        self.redirect = RedirectText(self.log_text)
        
        # Write welcome message
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"=== NRFI Dashboard Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self.log_text.insert(tk.END, "Ready to run NRFI analysis pipeline. Click on a step to begin.\n\n")
        self.log_text.configure(state='disabled')

    def clear_logs(self):
        """Clear the process output log"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')

    def show_about(self):
        """Display an About dialog with Kanye West image"""
        about_win = tk.Toplevel(self.root)
        about_win.title("About")
        about_win.geometry("500x500")
        about_win.configure(bg=self.bg_color)
        
        # Add Kanye West image
        image_path = os.path.join("data", "Dashboard", "mlb", "depositphotos_78976280-stock-photo-musician-kanye-west.webp")
        
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                img = img.resize((400, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                img_label = tk.Label(about_win, image=photo, bg=self.bg_color)
                img_label.image = photo
                img_label.pack(padx=20, pady=20)
                
                caption = tk.Label(
                    about_win,
                    text="NRFI Analysis Dashboard v1.0",
                    bg=self.bg_color,
                    fg=COLORS['text_primary'],
                    font=("Arial", 12, "bold")
                )
                caption.pack(pady=10)
            except Exception as e:
                error_label = tk.Label(
                    about_win,
                    text=f"Error loading image: {str(e)}",
                    bg=self.bg_color,
                    fg=COLORS['accent_error'],
                    font=("Arial", 12)
                )
                error_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        else:
            error_label = tk.Label(
                about_win,
                text=f"Image not found:\n{image_path}",
                bg=self.bg_color,
                fg=COLORS['accent_error'],
                font=("Arial", 12)
            )
            error_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

    def update_step_status(self, step, status):
        """Update the status of a given step and its progress bar"""
        step["status"] = status
        icons = {"running": "⚙️ Running", "completed": "✔️ Completed", "error": "❌ Error", "ready": "🟡 Ready"}
        colors = {"running": self.running_color, "completed": self.completed_color,
                  "error": self.error_color, "ready": COLORS['text_secondary']}
        step["status_label"].config(text=icons.get(status, "🟡 Ready"), fg=colors.get(status, COLORS['text_secondary']))
        
        if status == "running":
            step["button"].config(text="Cancel", bg=self.running_color)
            step["progress_bar"].start(10)
        else:
            step["button"].config(text="Run", bg=self.button_color)
            step["progress_bar"].stop()

    def run_step(self, step):
        """Run or cancel a step - modified for parallel execution"""
        if step["name"] in self.running_steps:
            if messagebox.askyesno("Cancel Process", f"Are you sure you want to cancel {step['name']}?"):
                if step["name"] in self.step_processes:
                    try:
                        self.step_processes[step["name"]].terminate()
                        print(f"\n[INFO] Cancelled {step['name']}")
                        self.update_step_status(step, "ready")
                        del self.running_steps[step["name"]]
                        del self.step_processes[step["name"]]
                    except Exception as e:
                        print(f"\n[ERROR] Failed to terminate process: {e}")
            return

        if len(self.running_steps) < self.max_concurrent:
            self.update_step_status(step, "running")
            self.running_steps[step["name"]] = step
            self.executor.submit(self._run_script, step)
        else:
            print(f"\n[INFO] Maximum concurrent steps reached. Queuing {step['name']}")
            self.step_queue.put(step)

    def _run_script(self, step):
        """Execute script with modified completion handling"""
        try:
            script_path = step["script"]
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, f"\n{'='*50}\n")
            self.log_text.insert(tk.END, f"Starting {step['name']} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_text.insert(tk.END, f"{'='*50}\n\n")
            self.log_text.configure(state='disabled')
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = self.redirect
            sys.stderr = self.redirect

            print(f"[INFO] Running {script_path}...")
            
            python_exe = "python"
            self.process = subprocess.Popen(
                [python_exe, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.step_processes[step["name"]] = self.process
            
            for line in self.process.stdout:
                print(line.strip())
            
            return_code = self.process.wait()
            if return_code == 0:
                print(f"\n[INFO] {step['name']} completed successfully")
                self.root.after(10, lambda: self.update_step_status(step, "completed"))
            else:
                print(f"\n[ERROR] {step['name']} failed with return code {return_code}")
                self.root.after(10, lambda: self.update_step_status(step, "error"))

        except Exception as e:
            print(f"\n[ERROR] Failed to run {step['name']}: {e}")
            self.root.after(10, lambda: self.update_step_status(step, "error"))
        
        finally:
            if step["name"] in self.running_steps:
                del self.running_steps[step["name"]]
            if step["name"] in self.step_processes:
                del self.step_processes[step["name"]]
                
            try:
                next_step = self.step_queue.get_nowait()
                self.run_step(next_step)
            except queue.Empty:
                pass

    def run_all_steps(self):
        """Run all steps sequentially"""
        if self.currently_running:
            print("\n[ERROR] Another process is already running. Please wait or cancel it first.")
            return
        
        threading.Thread(target=self._run_all_steps, daemon=True).start()
    
    def _run_all_steps(self):
        for step in self.steps:
            if self.currently_running:
                break
            
            self.root.after(0, lambda s=step: self.update_step_status(s, "running"))
            self.currently_running = step
            
            try:
                script_path = step["script"]
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, f"\n{'='*50}\n")
                self.log_text.insert(tk.END, f"Starting {step['name']} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_text.insert(tk.END, f"{'='*50}\n\n")
                self.log_text.configure(state='disabled')
                
                old_stdout = sys.stdout
                old_stderr = sys.stdout
                sys.stdout = self.redirect
                sys.stderr = self.redirect
                
                print(f"[INFO] Running {script_path}...")
                
                python_exe = "python"
                self.process = subprocess.Popen(
                    [python_exe, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                for line in self.process.stdout:
                    print(line.strip())
                
                return_code = self.process.wait()
                if return_code == 0:
                    print(f"\n[INFO] {step['name']} completed successfully")
                    self.root.after(10, lambda s=step: self.update_step_status(s, "completed"))
                else:
                    print(f"\n[ERROR] {step['name']} failed with return code {return_code}")
                    self.root.after(10, lambda s=step: self.update_step_status(s, "error"))
                    break
            
            except Exception as e:
                print(f"\n[ERROR] Failed to run {step['name']}: {e}")
                self.root.after(10, lambda s=step: self.update_step_status(s, "error"))
                break
            
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                self.currently_running = None
                time.sleep(1)

    def view_predictions(self):
        """Enhanced NRFI predictions view with advanced filtering and recommendations"""
        # Use config for predictions file path
        predictions_path = MODE_CONFIGS["NRFI"]["predictions_file"]
        # --- DEBUG LOGGING: Print the path being checked and existence ---
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"\n[DEBUG] Checking predictions path: {predictions_path}\n")
        self.log_text.insert(tk.END, f"[DEBUG] os.path.exists: {os.path.exists(predictions_path)}\n")
        # Also check for F1_predictions.csv as fallback
        alt_path = os.path.join(os.path.dirname(predictions_path), "F1_predictions.csv")
        self.log_text.insert(tk.END, f"[DEBUG] Checking alternate path: {alt_path}\n")
        self.log_text.insert(tk.END, f"[DEBUG] os.path.exists (alt): {os.path.exists(alt_path)}\n")
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)
        if not os.path.exists(predictions_path):
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, "\n[ERROR] No predictions file found. Run the NRFI analysis first.\n")
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)
            return
            
        pred_window = tk.Toplevel(self.root)
        pred_window.title("NRFI Predictions")
        pred_window.geometry("1200x600")
        pred_window.configure(bg=self.bg_color)
        
        # Store column relative widths for dynamic resizing
        column_proportions = {}
        
        # Function to handle window resize
        def on_window_resize(event=None):
            if event and event.widget == pred_window:
                window_width = event.width
                available_width = window_width - 40  # Account for scrollbar and padding
                
                # Calculate total proportion units
                total_proportion = sum(column_proportions.values())
                
                # Skip if no proportions or invalid total
                if total_proportion <= 0:
                    return
                    
                # Distribute width proportionally
                for col, proportion in column_proportions.items():
                    new_width = int((proportion / total_proportion) * available_width)
                    # Ensure minimum width
                    new_width = max(30, new_width)
                    tree.column(col, width=new_width)
                    
        # Bind resize event
        pred_window.bind("<Configure>", on_window_resize)
        
        header_frame = tk.Frame(pred_window, bg=self.bg_color, pady=10)
        header_frame.pack(fill=tk.X)
        
        tk.Label(
            header_frame,
            text="NRFI Predictions",
            font=STYLES['heading']['font'],
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=10)
        
        info_frame = tk.Frame(pred_window, bg=self.bg_color)
        info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Label(
            info_frame,
            text="• Green: High confidence NRFI prediction (No Runs in First Inning)",
            font=STYLES['text']['font'],
            bg=self.bg_color,
            fg=COLORS['text_secondary'],
            anchor="w"
        ).pack(fill=tk.X, padx=10)
        
        tk.Label(
            info_frame,
            text="• Red: High confidence YRFI prediction (Yes, Runs in First Inning)",
            font=STYLES['text']['font'],
            bg=self.bg_color,
            fg=COLORS['text_secondary'],
            anchor="w"
        ).pack(fill=tk.X, padx=10)
        
        tk.Label(
            info_frame,
            text="• Confidence scores are calibrated based on historical performance and prediction context",
            font=STYLES['text']['font'],
            bg=self.bg_color,
            fg=COLORS['text_secondary'],
            anchor="w"
        ).pack(fill=tk.X, padx=10)
        
        legend_frame = tk.Frame(pred_window, bg=self.bg_color, pady=5)
        legend_frame.pack(fill=tk.X, padx=10)
        
        legend_left = tk.Frame(legend_frame, bg=self.bg_color)
        legend_left.pack(side=tk.LEFT)
        
        high_nrfi = tk.Frame(legend_left, bg="#144a14", width=20, height=20)
        high_nrfi.pack(side=tk.LEFT, padx=5)
        tk.Label(
            legend_left, 
            text="High Confidence NRFI",
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)
        
        med_nrfi = tk.Frame(legend_left, bg="#1f6e1f", width=20, height=20)
        med_nrfi.pack(side=tk.LEFT, padx=5)
        tk.Label(
            legend_left, 
            text="Medium Confidence NRFI",
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)
        
        legend_right = tk.Frame(legend_frame, bg=self.bg_color)
        legend_right.pack(side=tk.LEFT, padx=20)
        
        high_yrfi = tk.Frame(legend_right, bg="#661414", width=20, height=20)
        high_yrfi.pack(side=tk.LEFT, padx=5)
        tk.Label(
            legend_right, 
            text="High Confidence YRFI",
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)
        
        med_yrfi = tk.Frame(legend_right, bg="#8b1e1e", width=20, height=20)
        med_yrfi.pack(side=tk.LEFT, padx=5)
        tk.Label(
            legend_right, 
            text="Medium Confidence YRFI",
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)
        
        slider_frame = tk.Frame(pred_window, bg=self.bg_color, pady=10)
        slider_frame.pack(fill=tk.X)
        
        tk.Label(
            slider_frame,
            text="Confidence Threshold:",
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=10)
        
        threshold_var = tk.DoubleVar(value=0.15)
        threshold_slider = ttk.Scale(
            slider_frame,
            from_=0.05,
            to=0.5,
            orient="horizontal",
            variable=threshold_var,
            length=200
        )
        threshold_slider.pack(side=tk.LEFT, padx=5)
        
        threshold_label = tk.Label(
            slider_frame,
            text="15%",
            bg=self.bg_color,
            fg=COLORS['text_primary'],
            width=5
        )
        threshold_label.pack(side=tk.LEFT, padx=5)
        
        highlight_var = tk.StringVar(value="both")
        highlight_frame = tk.Frame(pred_window, bg=self.bg_color, pady=5)
        highlight_frame.pack(fill=tk.X, padx=10)

        tk.Label(
            highlight_frame,
            text="Highlight Method:",
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=10)

        for method, desc in [
            ("confidence", "By Calibrated Confidence"), 
            ("probability", "By NRFI/YRFI Probability"), 
            ("both", "Both Methods")
        ]:
            tk.Radiobutton(
                highlight_frame,
                text=desc,
                variable=highlight_var,
                value=method,
                command=lambda: load_predictions(),
                bg=self.bg_color,
                fg=COLORS['text_primary'],
                selectcolor="black"
            ).pack(side=tk.LEFT, padx=5)
        
        sort_frame = tk.Frame(slider_frame, bg=self.bg_color)
        sort_frame.pack(side=tk.RIGHT, padx=10)
        
        tk.Label(
            sort_frame,
            text="Sort by:",
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)
        
        sort_var = tk.StringVar(value="confidence")
        sort_options = ttk.Combobox(
            sort_frame,
            textvariable=sort_var,
            values=["confidence", "nrfi_probability", "runs_probability", "game_time"],
            width=15,
            state="readonly"
        )
        sort_options.pack(side=tk.LEFT, padx=5)
        
        filter_frame = tk.Frame(pred_window, bg=self.bg_color, pady=5)
        filter_frame.pack(fill=tk.X, padx=10)

        tk.Label(
            filter_frame,
            text="Show:",
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=10)

        filter_var = tk.StringVar(value="all")
        filter_options = ttk.Combobox(
            filter_frame,
            textvariable=filter_var,
            values=["all", "nrfi_picks", "yrfi_picks", "high_confidence"],
            width=15,
            state="readonly"
        )
        filter_options.pack(side=tk.LEFT, padx=5)

        filter_options.bind("<<ComboboxSelected>>", lambda _: load_predictions())
        
        status_label = tk.Label(
            pred_window,
            text="",
            bg=self.bg_color,
            fg=COLORS['text_primary'],
            pady=5
        )
        status_label.pack(fill=tk.X, padx=10)

        frame = tk.Frame(pred_window, padx=10, pady=10, bg=self.bg_color)
        frame.pack(fill=tk.BOTH, expand=True)
        
        def load_predictions(event=None):            
            try:
                predictions = pd.read_csv(predictions_path)
                
                for item in tree.get_children():
                    tree.delete(item)
                
                threshold = threshold_var.get()
                highlight_method = highlight_var.get()
                sort_by = sort_var.get()
                filter_type = filter_var.get()
                
                 # --- ADD THIS BLOCK: Always sort by date/time before any filtering ---
                if "date" in predictions.columns and "game_time" in predictions.columns:
                    # Combine date and time for sorting
                    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
                    # Create datetime column with proper format
                    predictions["datetime"] = predictions["date"].dt.strftime('%Y-%m-%d') + " " + predictions["game_time"].astype(str)
                    # Parse with explicit format
                    predictions["datetime"] = pd.to_datetime(predictions["datetime"], format="%Y-%m-%d %H:%M", errors="coerce")
                    predictions = predictions.sort_values("datetime", ascending=True, na_position='last')
                elif "date" in predictions.columns:
                    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
                    predictions = predictions.sort_values("date", ascending=True, na_position='last')
                # ---------------------------------------------------------------------

                if filter_type == "nrfi_picks":
                    predictions = predictions[predictions['nrfi_probability'] > 0.5]
                elif filter_type == "yrfi_picks":
                    predictions = predictions[predictions['nrfi_probability'] <= 0.5]
                elif filter_type == "high_confidence":
                    predictions = predictions[predictions['confidence'] > threshold]
                
                if sort_by == "game_time":
                    predictions = predictions.sort_values("game_time")
                elif sort_by == "date":
                    # Ensure date is datetime for correct sorting
                    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
                    predictions = predictions.sort_values("date", ascending=False)
                else:
                    predictions = predictions.sort_values(sort_by, ascending=False)

                nrfi_count = len(predictions[predictions['nrfi_probability'] > 0.5])
                yrfi_count = len(predictions[predictions['nrfi_probability'] <= 0.5])
                high_conf_count = len(predictions[predictions['confidence'] > threshold])
                
                status_text = f"Total Games: {len(predictions)} | NRFI Picks: {nrfi_count} | YRFI Picks: {yrfi_count} | High Confidence: {high_conf_count}"
                status_label.config(text=status_text)
                
                for _, row in predictions.iterrows():
                    nrfi_prob = row['nrfi_probability']
                    yrfi_prob = row['runs_probability'] if 'runs_probability' in row else 1 - nrfi_prob
                    confidence = row['confidence'] if 'confidence' in row else abs(nrfi_prob - 0.5) * 2
                    row_values = []
                    for col in tree["columns"]:
                        value = row[col] if col in row else ""
                        if col in ['nrfi_probability', 'runs_probability', 'confidence', 'venue_nrfi_rate']:
                            if pd.notnull(value) and isinstance(value, (int, float)):
                                value = f"{value*100:.2f}%"  # Limit to 2 decimal places for percentages
                        elif col in ['temperature']:
                            if pd.notnull(value) and isinstance(value, (int, float)):
                                value = f"{value:.1f}°F"
                        elif isinstance(value, float):
                            # Format any other float values to 3 decimal places
                            value = f"{value:.3f}"
                        row_values.append(value)
                    
                    tags = []
                    
                    if highlight_method in ["confidence", "both"]:
                        if nrfi_prob > 0.5:
                            if confidence >= threshold * 2:
                                tags.append("high_nrfi")
                            elif confidence >= threshold:
                                tags.append("med_nrfi")
                        else:
                            if confidence >= threshold * 2:
                                tags.append("high_yrfi")
                            elif confidence >= threshold:
                                tags.append("med_yrfi")
                    
                    if highlight_method in ["probability", "both"]:
                        if nrfi_prob > 0.65:
                            tags.append("high_nrfi")
                        elif nrfi_prob > 0.58:
                            tags.append("med_nrfi")
                        elif nrfi_prob < 0.35:
                            tags.append("high_yrfi")
                        elif nrfi_prob < 0.42:
                            tags.append("med_yrfi")
                    
                    item_id = tree.insert("", "end", values=row_values, tags=(tags))
                    
                    if 'prediction_desc' in row:
                        ToolTip(tree, row['prediction_desc'], item=item_id)
                
                threshold_label.config(text=f"{threshold*100:.0f}%")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading predictions: {str(e)}")
        
        try:
            predictions = pd.read_csv(predictions_path)
            
            if 'runs_probability' not in predictions.columns:
                predictions['runs_probability'] = 1 - predictions['nrfi_probability']
            
            columns = list(predictions.columns)
            
            preferred_cols = [
                'date', 'game_time', 'home_team', 'away_team', 'home_pitcher_name', 'away_pitcher_name',
                'nrfi_probability', 'runs_probability', 'confidence', 'venue_nrfi_rate', 
                'prediction_label', 'temperature', 'condition', 'venue_name'
            ]
            
            display_cols = [col for col in preferred_cols if col in columns]
            display_cols.extend([col for col in columns if col not in display_cols])
            
            style = ttk.Style()
            style.configure("Treeview",
                          background=COLORS['bg_content'],
                          foreground=COLORS['text_primary'],
                          fieldbackground=COLORS['bg_content'],
                          font=('Arial', 8))  # Smaller base font
            style.configure("Treeview.Heading",
                          background=COLORS['bg_dark'],
                          foreground=COLORS['text_primary'],
                          font=('Arial', 8, 'bold'))  # Smaller header font
            
            tree = ttk.Treeview(frame, columns=display_cols, show='headings')
            
            # Configure tags with adjusted background colors
            tree.tag_configure("high_nrfi", background='#144a14')
            tree.tag_configure("high_yrfi", background='#661414')
            tree.tag_configure("med_nrfi", background='#1f6e1f')
            tree.tag_configure("med_yrfi", background='#8b1e1e')
            
            # Improved column headers with shorter names
            custom_headers = {
                'nrfi_probability': 'NRFI',
                'runs_probability': 'YRFI',
                'confidence': 'Conf',
                'venue_nrfi_rate': 'Venue %',
                'prediction_label': 'Pred',
                'prediction_desc': 'Details',
                'home_team': 'Home',
                'away_team': 'Away',
                'home_pitcher_name': 'Home P',
                'away_pitcher_name': 'Away P',
                'venue_name': 'Venue',
                'temperature': 'Temp',
                'condition': 'Cond'
            }            # Dynamic width calculation based on content type
            for col in display_cols:
                display_name = custom_headers.get(col, col.replace('_', ' ').title())
                tree.heading(col, text=display_name)
                
                # Store relative column width proportions for dynamic resizing
                if col in ['date']:
                    rel_width = 70  # Date needs less space with shorter format
                elif col in ['game_time']:
                    rel_width = 60  # Time can be more compact
                elif col in ['home_team', 'away_team']:
                    rel_width = 80  # Team names can be abbreviated if needed
                elif col in ['home_pitcher_name', 'away_pitcher_name']:
                    rel_width = 100  # Pitcher names need moderate space
                elif col in ['nrfi_probability', 'runs_probability', 'confidence', 'venue_nrfi_rate']:
                    rel_width = 65  # Percentages need a bit more space for 2 decimal places
                    tree.column(col, anchor='center')  # Center align percentages
                elif col == 'prediction_label':
                    rel_width = 80  # Prediction can be shorter
                elif col == 'temperature':
                    rel_width = 50  # Temperature is compact
                    tree.column(col, anchor='center')  # Center align temperature
                elif col == 'condition':
                    rel_width = 60  # Weather condition can be abbreviated
                elif col == 'venue_name':
                    rel_width = 120  # Venue names need more space
                else:
                    rel_width = 70  # Default width for other columns
                
                # Set initial column width    
                tree.column(col, width=rel_width)
                # Store relative width as tag for dynamic resizing
                tree.column(col, minwidth=min(30, rel_width))
                column_proportions[col] = rel_width
            
            y_scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            x_scrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)

            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            def add_tooltip_to_treeview(tree):
                def show_tooltip(event):
                    item = tree.identify('item', event.x, event.y)
                    if item and hasattr(tree, '_tooltips') and item in tree._tooltips:
                        # If we were waiting to show a tooltip for a different item, cancel it
                        if hasattr(tree, '_tooltip_timer') and tree._tooltip_timer:
                            tree.after_cancel(tree._tooltip_timer)
                            tree._tooltip_timer = None
                            
                        # If we already have a tooltip showing, don't create another
                        if hasattr(tree, '_tooltip_window') and tree._tooltip_window:
                            return
                            
                        # Store the current item
                        tree._current_tooltip_item = item
                        
                        # Schedule the tooltip to appear after delay
                        def display_tooltip():
                            if not hasattr(tree, '_current_tooltip_item') or tree._current_tooltip_item != item:
                                return
                                
                            try:
                                x, y, _, _ = tree.bbox(item, column=tree.identify('column', event.x, event.y))
                                y = y + tree.winfo_rooty() + 25
                                x = event.x_root
                                
                                tw = tk.Toplevel(tree)
                                tw.wm_overrideredirect(True)
                                tw.wm_geometry(f"+{x}+{y}")
                                label = tk.Label(
                                    tw, text=tree._tooltips[item], justify=tk.LEFT,
                                    background="#ffffcc", relief=tk.SOLID, borderwidth=1,
                                    font=("tahoma", "9", "normal")
                                )
                                label.pack(ipadx=2, ipady=2)
                                tree._tooltip_window = tw
                                tree._tooltip_timer = None
                            except:
                                pass
                        
                        # Set timer for 2 seconds delay
                        tree._tooltip_timer = tree.after(2000, display_tooltip)
                
                def hide_tooltip(event=None):
                    # Cancel any pending tooltip
                    if hasattr(tree, '_tooltip_timer') and tree._tooltip_timer:
                        tree.after_cancel(tree._tooltip_timer)
                        tree._tooltip_timer = None
                    
                    # Clear the current tooltip item
                    if hasattr(tree, '_current_tooltip_item'):
                        tree._current_tooltip_item = None
                    
                    # Destroy any visible tooltip
                    if hasattr(tree, '_tooltip_window') and tree._tooltip_window:
                        try:
                            tree._tooltip_window.destroy()
                            tree._tooltip_window = None
                        except:
                            pass
                            
                tree.bind('<Motion>', show_tooltip)
                tree.bind('<Leave>', hide_tooltip)
                tree._tooltips = {}
                tree._tooltip_window = None
                
                def set_tooltip(text, item):
                    tree._tooltips[item] = text
                    
                tree.set_tooltip = set_tooltip
                return tree
                
            ToolTip.original_init = ToolTip.__init__
            def new_tooltip_init(self, widget, text=None, item=None):
                if hasattr(widget, 'set_tooltip') and item:
                    widget.set_tooltip(text, item)
                else:
                    ToolTip.original_init(self, widget, text)
            ToolTip.__init__ = new_tooltip_init
            
            add_tooltip_to_treeview(tree)
            
            threshold_slider.configure(command=lambda _: load_predictions())
            sort_options.bind("<<ComboboxSelected>>", load_predictions)
            
            load_predictions()
            
        except Exception as e:
            error_label = tk.Label(
                frame, 
                text=f"Error loading predictions: {str(e)}",
                fg=COLORS['accent_error'],
                bg=self.bg_color,
                padx=20,
                pady=20
            )
            error_label.pack(fill=tk.BOTH, expand=True)

    def create_graph_buttons(self, parent_frame):
        """Create buttons for graph selection"""
        buttons_frame = tk.Frame(parent_frame, bg=COLORS['bg_content'])
        buttons_frame.grid(row=2, column=0, sticky="ew", pady=5)
        buttons_frame.grid_columnconfigure(0, weight=1)
        
        self.graph_buttons = {}
        
        button_grid = tk.Frame(buttons_frame, bg=COLORS['bg_content'])
        button_grid.grid(row=0, column=0, sticky="ew")
        
        # Initialize current_graphs before the loop
        mode = self.analysis_mode.get()
        current_graphs = []  # Default empty list
        
        # Define graphs based on mode
        if mode == "NRFI":
            current_graphs = [
                ("_NRFI_Analysis.png", "Overall Analysis"),
                ("_NRFI_Calendar.png", "Calendar"),
                ("_NRFI_Forecast.png", "Forecast"),
                ("_NRFI_Moving_Averages.png", "Moving Avg"),
                ("_NRFI_Trend_Enhanced.png", "Trends"),
                ("_NRFI_Weekly.png", "Weekly")
            ]
        elif mode == "PROPS":
            current_graphs = []  # Add prop-specific graphs if needed
        # Add other modes as needed
        
        row = 0
        col = 0
        max_cols = 4
        
        for graph_type, display_name in current_graphs:
            btn = tk.Button(
                button_grid,
                text=display_name,
                command=lambda g=graph_type: self.show_selected_graph(g),
                bg=self.button_color,
                fg=self.button_text_color,
                cursor="hand2",
                relief=tk.RAISED,
                padx=5
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            self.graph_buttons[graph_type] = btn
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        for i in range(max_cols):
            button_grid.grid_columnconfigure(i, weight=1)    
    def update_steps_for_mode(self):
        """Define pipeline steps based on current analysis mode"""
        mode = self.analysis_mode.get()
        
        # Get mode configuration from the centralized config
        mode_config = MODE_CONFIGS.get(mode)
        
        # Default values in case the mode is not found in configs
        title = "Baseball Analysis Dashboard"
        subtitle = "Select an analysis mode to begin"
        base_path = ""
        self.steps = []
        
        if mode_config:
            # Use values from the centralized config
            title = mode_config.get("display_name", title)
            subtitle = mode_config.get("subtitle", subtitle)
            base_path = mode_config.get("scripts_dir", "")
            
            # Create steps using the pipeline_steps from the mode_config
            pipeline_steps = mode_config.get("pipeline_steps", [])
            self.steps = []
            
            for step in pipeline_steps:
                self.steps.append({
                    "name": step["name"],
                    "script": os.path.join(base_path, step["script"]),
                    "status": "ready"
                })
        else:
            # Fallback to NRFI if mode not found in configs
            nrfi_config = MODE_CONFIGS.get("NRFI")
            if nrfi_config:
                title = nrfi_config.get("display_name")
                subtitle = nrfi_config.get("subtitle")
                base_path = nrfi_config.get("scripts_dir")
                
                # Create steps for NRFI
                pipeline_steps = nrfi_config.get("pipeline_steps", [])
                for step in pipeline_steps:
                    self.steps.append({
                        "name": step["name"],
                        "script": os.path.join(base_path, step["script"]),
                        "status": "ready"
                    })

        if hasattr(self, 'title_label') and hasattr(self, 'subtitle_label'):
            self.title_label.config(text=title)
            self.subtitle_label.config(text=subtitle)
    # Python
    def log_message(self, message, add_newlines=False):
        """Log a message to the log_text widget."""
        self.log_text.configure(state='normal')
        if add_newlines:
            self.log_text.insert(tk.END, f"\n{message}\n")
        else:
            self.log_text.insert(tk.END, f"{message}")        
            self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)    
    def change_analysis_mode(self):
        # Get the display name from dropdown
        selected_display = self.mode_dropdown.get()
        
        # Find the matching mode code from ANALYSIS_MODES
        new_mode = None
        for code, name in ANALYSIS_MODES:
            if name == selected_display:
                new_mode = code
                break
        
        if not new_mode:
            print(f"Error: Could not find mode code for '{selected_display}'")
            return
            
        # Store the mode in the StringVar
        self.analysis_mode.set(new_mode)
        old_mode = getattr(self, '_previous_mode', None)
        
        # Don't do anything if mode hasn't changed
        if new_mode == old_mode:
            return
        
        # Get mode configuration with error checking
        mode_config = MODE_CONFIGS.get(new_mode)
        if not mode_config:
            print(f"Error: No configuration found for mode {new_mode}")
            return
        
        # Validate required config fields
        required_fields = ['display_name', 'subtitle', 'scripts_dir', 'pipeline_steps']
        missing_fields = [field for field in required_fields if field not in mode_config]
        if missing_fields:
            print(f"Error: Missing required configuration fields: {missing_fields}")
            return
        
        # Update UI title and subtitle
        self.title_label.config(text=mode_config['display_name'])
        self.subtitle_label.config(text=mode_config['subtitle'])
        
        # Store new mode
        self._previous_mode = new_mode
        
        # Clear any running processes
        if self.currently_running:
            if self.process and self.process.poll() is None:
                if messagebox.askyesno("Cancel Process", 
                    "A process is currently running. Do you want to cancel it?"):
                    self.process.terminate()
            self.currently_running = None
        
        print(f"Changing mode to {new_mode}, updating steps...")
        
        # Update steps array based on current mode
        self.update_steps_for_mode()
        
        # Get reference to the steps frame
        steps_frame = None
        for child in self.content_frame.winfo_children():
            if isinstance(child, tk.LabelFrame) and child.cget("text") == "Analysis Pipeline":
                steps_frame = child
                break
        
        if not steps_frame:
            print("Error: Could not find Analysis Pipeline frame")
            return
            
        # Clear existing steps widgets
        for widget in steps_frame.winfo_children():
            widget.destroy()
            
        # Recreate step buttons for the new mode
        for i, step in enumerate(self.steps):
            step_frame = tk.Frame(steps_frame, bg=COLORS['bg_content'])
            step_frame.grid(row=i, column=0, sticky="ew", pady=2)
            
            # Status label
            step["status_label"] = tk.Label(
                step_frame,
                text="🟡 Ready",
                bg=COLORS['bg_content'],
                fg=COLORS['text_secondary']
            )
            step["status_label"].grid(row=0, column=0, padx=5, sticky="w")
            
            # Step button
            step["button"] = tk.Button(
                step_frame,
                text="Run",
                command=lambda s=step: self.run_step(s),
                bg=self.button_color,
                fg=self.button_text_color,
                cursor="hand2"
            )
            step["button"].grid(row=0, column=2, padx=5, sticky="e")
            
            # Progress bar
            step["progress_bar"] = ttk.Progressbar(
                step_frame,
                mode='indeterminate',
                length=200
            )
            step["progress_bar"].grid(row=0, column=1, padx=5, sticky="ew")
            
            # Step name
            tk.Label(
                step_frame,
                text=step["name"],
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary']
            ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 2))
            
            # Configure grid weights
            step_frame.grid_columnconfigure(1, weight=1)
            step_frame.grid_rowconfigure(0, weight=0)
            step_frame.grid_rowconfigure(1, weight=0)
        
        # Add Run All button at bottom of steps frame
        run_all_frame = tk.Frame(steps_frame, bg=COLORS['bg_content'])
        run_all_frame.grid(row=len(self.steps), column=0, sticky="ew", pady=5)
        
        tk.Button(
            run_all_frame,
            text="Run All Steps",
            command=self.run_all_steps,
            bg=self.button_color,
            fg=self.button_text_color,
            cursor="hand2"
        ).grid(row=0, column=0, sticky="e", padx=5)
        
        # Update graph selection dropdown for the new mode
        self.update_dropdown()

    def get_mode_display_name(self, mode_code):
        """Get the display name for a mode code"""
        for code, name in ANALYSIS_MODES:
            if code == mode_code:
                return name
        return mode_code

    def calculate_streaks(self, team, date, window=10, team_streak=True):
        """Calculate NRFI streaks for a team up to a specific date
        team_streak: If True, only look at team's scoring in 1st inning
                    If False, look at overall game NRFI result"""
        try:
            team = team.replace("F1_", "").replace("F1 ", "").strip().upper()
            
            historical_data = pd.read_csv(os.path.join("data", "First Inning NRFI", "F1_historical_schedule.csv"))
            print(f"Calculating streaks for {team} up to {date}")
            
            historical_data['date'] = pd.to_datetime(historical_data['date'])
            target_date = pd.to_datetime(date)
            
            historical_data['home_team'] = historical_data['home_team'].str.strip().str.upper()
            historical_data['away_team'] = historical_data['away_team'].str.strip().str.upper()
            
            historical_data = historical_data[historical_data['date'] < target_date].sort_values('date', ascending=False)
            
            print(f"Looking for team '{team}' in data")
            print(f"First few home teams in data: {list(historical_data['home_team'].head())}")
            print(f"First few away teams in data: {list(historical_data['away_team'].head())}")
            
            streak = []
            if team_streak:
                team_games = historical_data[
                    (historical_data['home_team'] == team) | 
                    (historical_data['away_team'] == team)
                ]
                print(f"Found {len(team_games)} games for {team}")
                
                if len(team_games) == 0:
                    print(f"No games found for {team}")
                    print(f"Available teams: {sorted(set(historical_data['home_team'].unique()))}")
                
                for _, game in team_games.head(window).iterrows():
                    if team == game['home_team']:
                        streak.append(1 if game['home_inning_1_runs'] == 0 else 0)
                    else:
                        streak.append(1 if game['away_inning_1_runs'] == 0 else 0)
            else:
                team_games = historical_data[
                    (historical_data['home_team'] == team) | 
                    (historical_data['away_team'] == team)
                ]
                
                if len(team_games) == 0:
                    print(f"No games found for {team}")
                    print(f"Available teams: {sorted(set(historical_data['home_team'].unique()))}")
                
                for _, game in team_games.head(window).iterrows():
                    streak.append(1 if game['home_inning_1_runs'] == 0 and 
                                game['away_inning_1_runs'] == 0 else 0)
            
            while len(streak) < window:
                streak.append(None)
                
            return streak
        except Exception as e:
            print(f"Error calculating streaks for {team}: {e}")
            return [None] * window

    def create_streak_display(self, parent_frame):
        """Create streak display section"""
        streak_frame = tk.LabelFrame(
            parent_frame,
            text="NRFI Streak Analysis",
            font=STYLES['heading']['font'],
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary'],
            padx=10,
            pady=10
        )
        streak_frame.pack(fill=tk.X, pady=10)
        
        window_frame = tk.Frame(streak_frame, bg=COLORS['bg_content'])
        window_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            window_frame,
            text="Streak Window:",
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)
        
        self.streak_window = tk.IntVar(value=10)
        for window in [3, 5, 10]:
            tk.Radiobutton(
                window_frame,
                text=f"Last {window}",
                variable=self.streak_window,
                value=window,
                command=self.update_streak_display,
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary'],
                selectcolor="black"
            ).pack(side=tk.LEFT, padx=5)
        
        self.home_streak_frame = tk.Frame(streak_frame, bg=COLORS['bg_content'])
        self.home_streak_frame.pack(fill=tk.X, pady=5)
        
        self.away_streak_frame = tk.Frame(streak_frame, bg=COLORS['bg_content'])
        self.away_streak_frame.pack(fill=tk.X, pady=5)
        
        self.team_streak_frame = tk.Frame(streak_frame, bg=COLORS['bg_content'])
        self.team_streak_frame.pack(fill=tk.X, pady=5)
        
        self.game_streak_frame = tk.Frame(streak_frame, bg=COLORS['bg_content'])
        self.game_streak_frame.pack(fill=tk.X, pady=5)
        
        return streak_frame

    def update_streak_display(self):
        """Update the streak display based on selected window size"""
        window = self.streak_window.get()
        
        # Clear existing displays
        for widget in self.team_streak_display.winfo_children():
            widget.destroy()
        for widget in self.game_streak_display.winfo_children():
            widget.destroy()
        
        selection = self.selection_dropdown.get()
        if not selection:
            return
            
        # Calculate and display streaks
        today = datetime.now().strftime("%Y-%m-%d")
        team_streak = self.calculate_streaks(selection, today, window, True)
        game_streak = self.calculate_streaks(selection, today, window, False)
        
        # Display team streak
        tk.Label(
            self.team_streak_display,
            text=f"{selection} Team NRFI Streak:",
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary'],
            font=STYLES['text']['font']
        ).pack(side=tk.LEFT, padx=5)
        self.display_streak_boxes(self.team_streak_display, team_streak)
        
        # Display game streak
        tk.Label(
            self.game_streak_display,
            text=f"{selection} Game NRFI Streak:",
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary'],
            font=STYLES['text']['font']
        ).pack(side=tk.LEFT, padx=5)
        self.display_streak_boxes(self.game_streak_display, game_streak)

    def display_streak_boxes(self, parent_frame, streak):
        """Display streak boxes inline with team analysis"""
        box_frame = tk.Frame(parent_frame, bg=COLORS['bg_content'])
        box_frame.pack(side=tk.LEFT, padx=5)
        
        for i, result in enumerate(streak):
            if result is None:
                color = "#666666"
                text = "?"
            elif result == 1:
                color = COLORS['accent_success']
                text = "✓"
            else:
                color = COLORS['accent_error']
                text = "✗"
            
            box = tk.Label(
                box_frame,
                text=text,
                width=2,
                bg=color,
                fg=COLORS['text_primary'],
                relief=tk.RAISED,
                borderwidth=1
            )
            box.pack(side=tk.LEFT, padx=1)
            
            games_ago = i + 1
            ToolTip(box, f"{games_ago} games ago")

    def view_props(self):
        """Display player prop recommendations from analyzed_props.csv"""
        props_file = os.path.join(DATA_DIR, "analyzed_props.csv")
        
        if not os.path.exists(props_file):
            messagebox.showerror("Error", "Props data not found. Please run the props model first.")
            return
            
        try:
            df = pd.read_csv(props_file)
            
            if df.empty:
                messagebox.showinfo("Info", "No player props data available.")
                return
                
            props_window = tk.Toplevel(self.root)
            props_window.title("Player Props Recommendations")
            props_window.geometry("1100x800")
            props_window.configure(bg=COLORS['bg_content'])
            
            main_frame = tk.Frame(props_window, bg=COLORS['bg_content'])
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            header_frame = tk.Frame(main_frame, bg=COLORS['bg_content'])
            header_frame.pack(fill=tk.X, pady=(0, 10))
            
            tk.Label(
                header_frame,
                text="Player Props Recommendations",
                font=STYLES['heading']['font'],
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary']
            ).pack(side=tk.LEFT)
            
            filter_frame = tk.Frame(main_frame, bg=COLORS['bg_content'])
            filter_frame.pack(fill=tk.X, pady=(0, 10))
            
            tk.Label(
                filter_frame,
                text="Minimum Score:",
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary']
            ).pack(side=tk.LEFT, padx=(0, 5))
            
            score_var = tk.DoubleVar(value=1.0)
            score_options = [0.5, 0.75, 1.0, 1.5, 2.0]
            score_dropdown = ttk.Combobox(
                filter_frame,
                textvariable=score_var,
                values=score_options,
                width=5,
                state="readonly"
            )
            score_dropdown.pack(side=tk.LEFT, padx=(0, 20))
            
            tk.Label(
                filter_frame,
                text="Minimum EV:",
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary']
            ).pack(side=tk.LEFT, padx=(0, 5))
            
            ev_var = tk.DoubleVar(value=1.0)
            ev_options = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            ev_dropdown = ttk.Combobox(
                filter_frame,
                textvariable=ev_var,
                values=ev_options,
                width=5,
                state="readonly"
            )
            ev_dropdown.pack(side=tk.LEFT, padx=(0, 20))
            
            tk.Label(
                filter_frame,
                text="Prop Type:",
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary']
            ).pack(side=tk.LEFT, padx=(0, 5))
            
            prop_types = ['All'] + sorted(df['Prop'].unique().tolist())
            prop_var = tk.StringVar(value='All')
            prop_dropdown = ttk.Combobox(
                filter_frame,
                textvariable=prop_var,
                values=prop_types,
                width=20,
                state="readonly"
            )
            prop_dropdown.pack(side=tk.LEFT, padx=(0, 20))
            
            legend_frame = tk.Frame(main_frame, bg=COLORS['bg_content'])
            legend_frame.pack(fill=tk.X, pady=(0, 10))
            
            legend_labels = [
                ("🔥 Strong Bet (Score > 2.0)", "#19a55d"),
                ("✅ Good Bet (Score > 1.0)", "#4CAF50"),
                ("🟨 Fair Bet (Score > 0.75)", "#FFA500"),
                ("❌ Poor Bet (Score < 0.75)", "#e74c3c")
            ]
            
            for text, color in legend_labels:
                legend_item = tk.Frame(legend_frame, bg=COLORS['bg_content'])
                legend_item.pack(side=tk.LEFT, padx=10)
                
                tk.Label(
                    legend_item,
                    text=text,
                    font=STYLES['text']['font'],
                    bg=COLORS['bg_content'],
                    fg=color
                ).pack()
            
            columns = ['Player', 'Prop', 'Line', 'Best Side', 'Best EV', 'Score', 
                      'Confidence', 'Current Streak', 'Last 5 Avg', 'Season Avg', 'Trend']
            
            tree_frame = tk.Frame(main_frame, bg=COLORS['bg_content'])
            tree_frame.pack(fill=tk.BOTH, expand=True)
            
            tree_scroll_y = ttk.Scrollbar(tree_frame)
            tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            
            tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
            tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            
            tree = ttk.Treeview(
                tree_frame,
                columns=columns,
                show="headings",
                yscrollcommand=tree_scroll_y.set,
                xscrollcommand=tree_scroll_x.set
            )
            
            
            tree.pack(fill=tk.BOTH, expand=True)
            
            status_frame = tk.Frame(main_frame, bg=COLORS['bg_content'])
            status_frame.pack(fill=tk.X, pady=(10, 0))
            
            status_label = tk.Label(
                status_frame,
                text=f"Showing {len(df)} props",
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary'],
                anchor=tk.W
            )
            status_label.pack(side=tk.LEFT)
            
            export_button = tk.Button(
                status_frame,
                text="Export to CSV",
                bg=COLORS['accent_primary'],
                fg=COLORS['text_primary'],
                command=lambda: self.export_filtered_props(df)
            )
            export_button.pack(side=tk.RIGHT)
            
            def update_tree(*args):
                for item in tree.get_children():
                    tree.delete(item)
                
                min_score = score_var.get()
                min_ev = ev_var.get()
                selected_prop = prop_var.get()
                
                filtered_df = df
                
                if selected_prop != 'All':
                    filtered_df = filtered_df[filtered_df['Prop'] == selected_prop]
                
                filtered_df = filtered_df[
                    (filtered_df['Score'] >= min_score) & 
                    (filtered_df['Best EV'].abs() >= min_ev)
                ]
                
                filtered_df = filtered_df.sort_values('Score', ascending=False)
                
                for _, row in filtered_df.iterrows():
                    best_side = "Over" if abs(row['EV Over']) > abs(row['EV Under']) else "Under"
                    best_ev = row['EV Over'] if best_side == "Over" else row['EV Under']
                    
                    tag = ""
                    if row['Score'] >= 2.0:
                        tag = "strong"
                    elif row['Score'] >= 1.0:
                        tag = "good"
                    elif row['Score'] >= 0.75:
                        tag = "fair"
                    else:
                        tag = "poor"
                    
                    tree.insert('', tk.END, values=(
                        row['Player'],
                        row['Prop'],
                        row['Line'],
                        f"{best_side} ({row['Over %' if best_side == 'Over' else 'Under %']}%)",
                        best_ev,
                        row['Score'],
                        row['Confidence'],
                        row['Current Streak'],
                        row['Last 5 Avg'],
                        row['Season Avg'],
                        row.get('Trend', 'N/A')
                    ), tags=(tag,))
                
                status_label.config(text=f"Showing {len(filtered_df)} props")
            
            tree.tag_configure("strong", background="#19a55d", foreground="white")
            tree.tag_configure("good", background="#4CAF50", foreground="white")
            tree.tag_configure("fair", background="#FFA500", foreground="black")
            tree.tag_configure("poor", background="#e74c3c", foreground="white")
            
            score_dropdown.bind("<<ComboboxSelected>>", update_tree)
            ev_dropdown.bind("<<ComboboxSelected>>", update_tree)
            prop_dropdown.bind("<<ComboboxSelected>>", update_tree)
            
            def on_tree_double_click(event):
                item = tree.identify('item', event.x, event.y)
                if item:
                    values = tree.item(item, 'values')
                    player = values[0]
                    prop = values[1]
                    line = values[2]
                    
                    prop_details = df[(df['Player'] == player) & (df['Prop'] == prop) & (df['Line'].astype(str) == line)]
                    if not prop_details.empty:
                        self.show_prop_details(prop_details.iloc[0])
            
            tree.bind("<Double-1>", on_tree_double_click)
            
            update_tree()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load props data: {e}")

    def export_filtered_props(self, df):
        """Export the filtered props to a CSV file"""
        try:
            file_path = os.path.join(DATA_DIR, "exported_props.csv")
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Props data exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {e}")
            
    def show_prop_details(self, prop_data):
        """Show detailed information for a single prop"""
        details_window = tk.Toplevel(self.root)
        details_window.title(f"{prop_data['Player']} - {prop_data['Prop']}")
        details_window.geometry("600x500")
        details_window.configure(bg=COLORS['bg_content'])
        
        main_frame = tk.Frame(details_window, bg=COLORS['bg_content'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        header_frame = tk.Frame(main_frame, bg=COLORS['bg_content'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(
            header_frame,
            text=f"{prop_data['Player']} - {prop_data['Prop']} {prop_data['Line']}",
            font=STYLES['heading']['font'],
            bg=COLORS['bg_content'],
            fg=COLORS['text_primary']
        ).pack(anchor=tk.W)
        
        sections = [
            ("Betting Info", ['Line', 'Over %', 'Under %', 'Raw Over %', 'Raw Under %', 'EV Over', 'EV Under', 'Best EV', 'Score', 'Confidence']),
            ("Performance Metrics", ['Current Streak', 'Last 5 Avg', 'Last 10 Avg', 'Season Avg', 'Games Analyzed']),
            ("Advanced Metrics", ['recent_avg', 'recent_slg', 'recent_obp', 'contact_quality', 'plate_discipline', 
                                 'momentum', 'barrel_rate', 'contact_rate', 'chase_rate', 'whiff_rate', 'recent_form', 
                                 'Trend', 'momentum_status', 'heat_status'])
        ]
        
        for section_title, fields in sections:
            section_frame = tk.LabelFrame(
                main_frame,
                text=section_title,
                font=STYLES['heading']['font'],
                bg=COLORS['bg_content'],
                fg=COLORS['text_primary'],
                padx=10,
                pady=10
            )
            section_frame.pack(fill=tk.X, pady=(0, 15))
            
            for i, field in enumerate(fields):
                if field in prop_data and not pd.isna(prop_data[field]):
                    row, col = divmod(i, 2)
                    
                    tk.Label(
                        section_frame,
                        text=f"{field}:",
                        font=STYLES['text']['font'],
                        bg=COLORS['bg_content'],
                        fg=COLORS['text_secondary'],
                        anchor=tk.E,
                        width=15
                    ).grid(row=row, column=col*2, sticky=tk.E, padx=(5, 0), pady=5)
                    
                    value = prop_data[field]
                    color = COLORS['text_primary']
                    
                    if field in ['EV Over', 'EV Under', 'Best EV']:
                        if abs(value) > 1.5:
                            color = "#19a55d"
                        elif abs(value) > 1.0:
                            color = "#4CAF50"
                        elif abs(value) > 0.5:
                            color = "#FFA500"
                    elif field == 'Score':
                        if value > 2.0:
                            color = "#19a55d"
                        elif value > 1.0:
                            color = "#4CAF50"
                        elif value > 0.75:
                            color = "#FFA500"
                        else:
                            color = "#e74c3c"
                    elif field == 'Confidence':
                        if value > 0.9:
                            color = "#19a55d"
                        elif value > 0.8:
                            color = "#4CAF50"
                        elif value > 0.7:
                            color = "#FFA500"
                    
                    tk.Label(
                        section_frame,
                        text=str(value),
                        font=STYLES['text']['font'],
                        bg=COLORS['bg_content'],
                        fg=color,
                        anchor=tk.W
                    ).grid(row=row, column=col*2+1, sticky=tk.W, padx=(5, 10), pady=5)
        
        rec_frame = tk.Frame(main_frame, bg=COLORS['bg_content'])
        rec_frame.pack(fill=tk.X, pady=(10, 0))
        
        best_side = "OVER" if abs(prop_data['EV Over']) > abs(prop_data['EV Under']) else "UNDER"
        best_ev = prop_data['EV Over'] if best_side == "OVER" else prop_data['EV Under']
        
        recommendation = ""
        rec_color = COLORS['text_primary']
        
        if prop_data['Score'] > 2.0:
            recommendation = f"STRONG {best_side} RECOMMENDATION"
            rec_color = "#19a55d"
        elif prop_data['Score'] > 1.0:
            recommendation = f"GOOD {best_side} BET"
            rec_color = "#4CAF50"
        elif prop_data['Score'] > 0.75:
            recommendation = f"CONSIDER {best_side}"
            rec_color = "#FFA500"
        else:
            recommendation = "PASS ON THIS BET"
            rec_color = "#e74c3c"
        
        tk.Label(
            rec_frame,
            text=recommendation,
            font=("Arial", 14, "bold"),
            bg=COLORS['bg_content'],
            fg=rec_color
        ).pack(pady=10)

    def view_backtesting_results(self):
        """Display NRFI backtesting results with filtering and analysis features"""
        try:
            backtesting_file = os.path.join("data", "First Inning NRFI", "F1_backtesting_results.csv")
            if not os.path.exists(backtesting_file):
                messagebox.showerror("Error", f"Backtesting results file not found at {backtesting_file}")
                return
                
            backtest_window = tk.Toplevel(self.root)
            backtest_window.title("NRFI Backtesting Results")
            backtest_window.geometry("1200x800")
            backtest_window.configure(bg=self.bg_color)
            
            df = pd.read_csv(backtesting_file)
            
            controls_frame = tk.Frame(backtest_window, bg=self.bg_color)
            controls_frame.pack(fill=tk.X, padx=20, pady=10)
            
            filter_frame = tk.LabelFrame(controls_frame, text="Filter Results", font=STYLES['heading']['font'], 
                                       bg=self.bg_color, fg=COLORS['text_primary'], padx=10, pady=10)
            filter_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
            
            tk.Label(filter_frame, text="Prediction:", bg=self.bg_color, fg=COLORS['text_primary']).grid(row=0, column=0, padx=5, pady=5)
            prediction_var = tk.StringVar(value="all")
            prediction_options = [("All", "all"), ("NRFI", "0"), ("RFI", "1")]
            col = 1
            for text, value in prediction_options:
                tk.Radiobutton(filter_frame, text=text, variable=prediction_var, value=value,
                              bg=self.bg_color, fg=COLORS['text_primary'], selectcolor="black").grid(row=0, column=col, padx=5, pady=5)
                col += 1
            
            tk.Label(filter_frame, text="Date Range:", bg=self.bg_color, fg=COLORS['text_primary']).grid(row=0, column=4, padx=5, pady=5)
            date_var = tk.StringVar(value="all")
            date_options = [("All", "all"), ("Last Week", "last_week"), ("Last Month", "last_month"), 
                           ("Last 3 Months", "last_3_months")]
            for text, value in date_options:
                tk.Radiobutton(filter_frame, text=text, variable=date_var, value=value,
                              bg=self.bg_color, fg=COLORS['text_primary'], selectcolor="black").grid(row=0, column=col, padx=5, pady=5)
                col += 1
            
            tk.Label(filter_frame, text="Min. Confidence:", bg=self.bg_color, fg=COLORS['text_primary']).grid(row=1, column=0, padx=5, pady=5)
            confidence_slider = ttk.Scale(filter_frame, from_=0, to=1, orient=tk.HORIZONTAL, length=200)
            confidence_slider.set(0.0)
            confidence_slider.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
            
            confidence_label = tk.Label(filter_frame, text="0.0", bg=self.bg_color, fg=COLORS['text_primary'])
            confidence_label.grid(row=1, column=3, padx=5, pady=5)
            
            last_update = {'time': 0}
            def throttled_update(val):
                current_time = time.time()
                if current_time - last_update['time'] > 0.1:
                    confidence_label.config(text=f"{float(val):.2f}")
                    load_data()
                    last_update['time'] = current_time
            
            confidence_slider.configure(command=throttled_update)

            tree_columns = ("Date", "Game", "Result", "NRFI Prob", "Prediction", "Confidence", "Correct")
            tree = ttk.Treeview(backtest_window, columns=tree_columns, show='headings', height=20)
            
            tree.heading("Date", text="Date")
            tree.heading("Game", text="Game")
            tree.heading("Result", text="Result")
            tree.heading("NRFI Prob", text="NRFI Prob")
            tree.heading("Prediction", text="Prediction")
            tree.heading("Confidence", text="Confidence")
            tree.heading("Correct", text="✓/✗")
            
            tree.column("Date", width=100)
            tree.column("Game", width=200)
            tree.column("Result", width=80)
            tree.column("NRFI Prob", width=100)
            tree.column("Prediction", width=100)
            tree.column("Confidence", width=100)
            tree.column("Correct", width=50, anchor="center")
            
            scrollbar = ttk.Scrollbar(backtest_window, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=10)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10, padx=(0, 20))
            
            metrics_frame = tk.Frame(backtest_window, bg=self.bg_color)
            metrics_frame.pack(fill=tk.X, padx=20, pady=10)
            
            metric_labels = []
            for i in range(6):
                metric_frame = tk.Frame(metrics_frame, bg=self.bg_color)
                metric_frame.pack(side=tk.LEFT, expand=True, padx=5)
                
                title_label = tk.Label(metric_frame, text="", bg=self.bg_color, fg=COLORS['text_primary'])
                title_label.pack()
                
                value_label = tk.Label(metric_frame, text="", bg=self.bg_color, fg=COLORS['text_primary'], 
                                     font=("Arial", 16, "bold"))
                value_label.pack()
                
                metric_labels.append((title_label, value_label))

            def update_confidence_label(val):
                confidence_label.config(text=f"{float(val):.2f}")
                backtest_window.after(100, load_data)
            
            confidence_slider.configure(command=update_confidence_label)

            def load_data(*args):
                try:
                    tree.delete(*tree.get_children())
                    
                    prediction_filter = prediction_var.get()
                    date_filter = date_var.get()
                    confidence_threshold = float(confidence_slider.get())
                    
                    filtered_df = df.copy()
                    
                    filtered_df['prediction'] = (filtered_df['nrfi_probability'] >= 0.5).astype(int)
                    filtered_df['prediction_label'] = filtered_df['prediction'].map({1: 'NRFI', 0: 'RFI'})
                    
                    filtered_df['confidence'] = filtered_df.apply(
                        lambda row: max(row['nrfi_probability'], 1 - row['nrfi_probability']), 
                        axis=1
                    )
                    
                    if prediction_filter != "all":
                        filtered_df = filtered_df[filtered_df["prediction"] == int(prediction_filter)]
                    
                    filtered_df = filtered_df[filtered_df["confidence"] >= confidence_threshold]
                    
                    today = pd.Timestamp.now()
                    if date_filter == "last_week":
                        start_date = today - pd.Timedelta(days=7)
                        filtered_df = filtered_df[pd.to_datetime(filtered_df["date"]) >= start_date]
                    elif date_filter == "last_month":
                        start_date = today - pd.Timedelta(days=30)
                        filtered_df = filtered_df[pd.to_datetime(filtered_df["date"]) >= start_date]
                    elif date_filter == "last_3_months":
                        start_date = today - pd.Timedelta(days=90)
                        filtered_df = filtered_df[pd.to_datetime(filtered_df["date"]) >= start_date]
                    
                    total_games = len(filtered_df)
                    if total_games > 0:
                        accuracy = (filtered_df["prediction"] == filtered_df["nrfi"]).mean() * 100
                        
                        try:
                            from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
                            roc_auc = roc_auc_score(filtered_df["nrfi"], filtered_df["nrfi_probability"]) * 100
                            precision = precision_score(filtered_df["nrfi"], filtered_df["prediction"]) * 100
                            recall = recall_score(filtered_df["nrfi"], filtered_df["prediction"]) * 100
                            f1 = f1_score(filtered_df["nrfi"], filtered_df["prediction"]) * 100
                        except:
                            roc_auc = precision = recall = f1 = 0.0
                        
                        metrics_data = [
                            ("Total Games", f"{total_games}", COLORS['text_primary']),
                            ("Accuracy", f"{accuracy:.1f}%", COLORS['accent_success']),
                            ("ROC AUC", f"{roc_auc:.1f}%", COLORS['accent_secondary']),
                            ("Precision", f"{precision:.1f}%", "#ff9500"),
                            ("Recall", f"{recall:.1f}%", COLORS['accent_error']),
                            ("F1 Score", f"{f1:.1f}%", "#ffcc00")
                        ]
                        
                        for i, (title, value, color) in enumerate(metrics_data):
                            if i < len(metric_labels):
                                title_lbl, value_lbl = metric_labels[i]
                                title_lbl.config(text=title)
                                value_lbl.config(text=value, fg=color)
                    
                    for idx, row in filtered_df.iterrows():
                        date_val = row["date"]
                        game_val = f"{row['away_team']} @ {row['home_team']}"
                        nrfi_val = "NRFI" if row["nrfi"] == 1 else "RFI"
                        nrfi_prob = f"{row['nrfi_probability']:.3f}"
                        pred_val = row["prediction_label"]
                        conf_val = f"{row['confidence']:.3f}"
                        correct = "✓" if row["prediction"] == row["nrfi"] else "✗"
                        
                        tag = "correct" if row["prediction"] == row["nrfi"] else "incorrect"
                        tree.insert("", tk.END, values=(
                            date_val, game_val, nrfi_val, nrfi_prob, pred_val, conf_val, correct
                        ), tags=(tag,))
                    
                    tree.tag_configure("correct", background="#1e3a2f")
                    tree.tag_configure("incorrect", background="#3a1e1e")
                    
                    filter_text = f"NRFI Backtesting Results - {total_games} games"
                    if prediction_filter != "all":
                        filter_text += f" ({prediction_filter})"
                    if date_filter != "all":
                        filter_text += f" - {date_filter.replace('_', ' ').title()}"
                    backtest_window.title(filter_text)
                
                except Exception as e:
                    print(f"Error updating display: {e}")
                    tree.delete(*tree.get_children())
            
            confidence_slider.configure(command=lambda val: update_confidence_label(val) or load_data())
            prediction_var.trace("w", load_data)
            date_var.trace("w", load_data)
            
            load_data()
            
        except Exception as e:
            error_label = tk.Label(
                backtest_window if 'backtest_window' in locals() else self.root,
                text=f"Error loading backtesting data: {str(e)}",
                fg=COLORS['accent_error'],
                bg=self.bg_color,
                font=STYLES['text']['font'],
                wraplength=600,
                justify=tk.LEFT,
                padx=20,
                pady=20
            )
            error_label.pack(fill=tk.BOTH, expand=True)
            print(f"Error in view_backtesting_results: {e}")

    def create_model_evaluation_tab(self, parent):
        """Create the model evaluation tab with various metrics and graphs"""
        frame = tk.Frame(parent, bg=self.bg_color)
        frame.pack(fill=tk.BOTH, expand=True)

        self.create_title(frame, "Model Evaluation & Performance Metrics")

        button_frame = tk.Frame(frame, bg=self.bg_color)
        button_frame.pack(pady=10)

        backtest_button = self.create_button(
            button_frame,
            text="View Backtesting Results",
            command=self.view_backtesting_results,
            width=20,
            height=1,
            bg=COLORS['accent_primary']        
        )
        backtest_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.graph_frame = tk.Frame(frame, bg=self.bg_color)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        stats_frame = tk.Frame(frame, bg=self.bg_color)
        stats_frame.pack(fill=tk.X, padx=20, pady=10)

        return frame

    def view_game_predictions(self):
        """View sports game predictions from AOI_Moneyline.py outputs"""
        import os
        import pandas as pd
        import tkinter as tk
        from tkinter import ttk
        
        # Find predictions directory
        predictions_dir = os.path.join(DATA_DIR, "AOI Sports", "predictions")
        if not os.path.exists(predictions_dir):
            predictions_dir = os.path.join("predictions")
            if not os.path.exists(predictions_dir):
                predictions_dir = os.path.join(DATA_DIR, "predictions")
                if not os.path.exists(predictions_dir):
                    self.log_text.configure(state='normal')
                    self.log_text.insert(tk.END, "\n[ERROR] No predictions directory found.\n")
                    self.log_text.configure(state='disabled')
                    self.log_text.see(tk.END)
                    return
                    
        # Create mappings between file prefixes and display names
        prediction_files = []
        file_to_sport_map = {}  # Map filenames to display names
        sport_to_prefix_map = {}  # Map display names to file prefixes
        
        # Define display name mapping
        sport_display_names = {
            "NBA": "NBA Basketball",
            "NFL": "NFL Football",
            "MLB": "MLB Baseball",
            "NHL": "NHL Hockey",
            "EPL": "Premier League",
            "LALIGA": "La Liga",
            "BUNDESLIGA": "Bundesliga",
            "SERIEA": "Serie A",
            "LIGUE1": "Ligue 1",
            "UCL": "Champions League",
            "UEL": "Europa League",
            "MLS": "MLS Soccer",
            "LIGAMX": "Liga MX",
            "WNBA": "WNBA Basketball",
            "UFC": "UFC/MMA",
            "CFB": "College Football",
            "CBB": "College Basketball"
        }
        
        # Get all prediction files and create mappings
        for f in os.listdir(predictions_dir):
            if f.endswith('_upcoming_predictions.csv') or f.endswith('_predictions.csv'):
                prediction_files.append(f)
                
                # Get prefix (like NBA_, EPL_, etc.)
                if '_upcoming_predictions.csv' in f:
                    prefix = f.split('_upcoming_predictions.csv')[0]
                else:
                    prefix = f.split('_predictions.csv')[0]
                
                # Create display name and mappings
                display_name = sport_display_names.get(prefix, prefix)
                file_to_sport_map[f] = display_name
                sport_to_prefix_map[display_name] = prefix
        
        if not prediction_files:
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, "\n[ERROR] No prediction files found. Run the sports model first.\n")
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)
            return
            
        # Create the predictions window
        pred_window = tk.Toplevel(self.root)
        pred_window.title("Sports Predictions")
        pred_window.geometry("1400x800")
        pred_window.configure(bg=self.bg_color)
        
        # Create header and sport selection
        header_frame = tk.Frame(pred_window, bg=self.bg_color, pady=10)
        header_frame.pack(fill=tk.X)
        
        tk.Label(
            header_frame,
            text="Sports Game Predictions",
            font=STYLES['heading']['font'],
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=10)
        
        sport_frame = tk.Frame(header_frame, bg=self.bg_color)
        sport_frame.pack(side=tk.RIGHT, padx=10)
        
        tk.Label(
            sport_frame,
            text="Select Sport:",
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=5)
        
        # Sort sports alphabetically by display name
        display_names = sorted(sport_to_prefix_map.keys())
        sport_var = tk.StringVar(value=display_names[0] if display_names else "")
        
        sport_dropdown = ttk.Combobox(
            sport_frame,
            textvariable=sport_var,
            values=display_names,
            width=20,
            state="readonly"
        )
        sport_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Add a sort dropdown
        sort_var = tk.StringVar(value="Date/Time")
        sort_options = ttk.Combobox(
            header_frame,
            textvariable=sort_var,
            values=["Date/Time", "Edge", "Home%", "Away%"],  # Add more if needed
            width=15,
            state="readonly"
        )
        sort_options.pack(side=tk.RIGHT, padx=5)
        tk.Label(
            header_frame,
            text="Sort by:",
            bg=self.bg_color,
            fg=COLORS['text_primary']
        ).pack(side=tk.RIGHT, padx=5)
        
        # Status label for feedback
        status_label = tk.Label(
            pred_window,
            text="",
            bg=self.bg_color,
            fg=COLORS['text_primary'],
            pady=5
        )
        status_label.pack(fill=tk.X, padx=10)
        
        # Main content frame
        frame = tk.Frame(pred_window, padx=10, pady=10, bg=self.bg_color)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Define columns for prediction display
        columns = [
            "Date/Time", "Matchup", "Home%", "Away%", "HomeML", "AwayML", "VegasHomeML", "VegasAwayML",
            "VegasSpread", "PredSpread", "Bet?", "BetTeam", "Edge", "BetSize", "KellySize", "Notes"
        ]
        
        # Create treeview for displaying predictions
        tree = ttk.Treeview(frame, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=tk.CENTER)
        
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Style for different bet types
        tree.tag_configure("strong_bet", background="#144a14")
        tree.tag_configure("good_bet", background="#1f6e1f")
        tree.tag_configure("weak_bet", background="#3a3a1e")
        tree.tag_configure("normal", background=COLORS['bg_dark'])
        
        # Format game time helper function
        def format_game_time(time_str):
            try:
                if pd.isna(time_str) or not isinstance(time_str, str):
                    return ""
                if ":" in time_str:
                    hours, minutes = map(int, time_str.split(":"))
                    period = "PM" if hours >= 12 else "AM"
                    hours = hours % 12
                    hours = 12 if hours == 0 else hours
                    return f"{hours}:{minutes:02d} {period}"
                return time_str
            except:
                return time_str
                  # Function to load predictions based on selected sport
        def load_predictions(*args):
            try:
                # Clear existing items
                for item in tree.get_children():
                    tree.delete(item)
                
                # Get selected sport and corresponding file prefix
                selected_display_name = sport_var.get()
                selected_prefix = sport_to_prefix_map.get(selected_display_name)
                
                if not selected_prefix:
                    status_label.config(text=f"Error: Could not find prefix for {selected_display_name}")
                    return
                
                # List of possible directories to check for prediction files
                possible_dirs = [
                    predictions_dir,
                    os.path.join(DATA_DIR, "predictions"),
                    os.path.join("predictions")
                ]
                
                # List of possible file naming patterns
                file_patterns = [
                    f"{selected_prefix}_upcoming_predictions.csv",
                    f"{selected_prefix}_predictions.csv",
                ]
                
                # Try to find the prediction file
                prediction_file = None
                for directory in possible_dirs:
                    for pattern in file_patterns:
                        potential_file = os.path.join(directory, pattern)
                        if os.path.exists(potential_file):
                            prediction_file = potential_file
                            print(f"[DEBUG] Found prediction file: {prediction_file}")
                            break
                    if prediction_file:
                        break
                
                if not prediction_file:
                    searched_paths = [os.path.join(d, p) for d in possible_dirs for p in file_patterns]
                    paths_str = "\n - ".join(searched_paths)
                    status_label.config(text=f"No prediction file found for {selected_display_name}")
                    print(f"[DEBUG] Searched in:\n - {paths_str}")
                    return
                
                # Load predictions from CSV
                predictions = pd.read_csv(prediction_file)
                
                print(f"[DEBUG] CSV columns: {predictions.columns.tolist()}")
                print(f"[DEBUG] CSV row count: {len(predictions)}")
                
                # Process dates and times
                if 'Date' in predictions.columns:
                    predictions['Date'] = pd.to_datetime(predictions['Date'], errors='coerce')
                    
                # Setup datetime field for sorting
                if 'game_time' in predictions.columns and 'Date' in predictions.columns:
                    predictions['display_time'] = predictions['game_time'].apply(format_game_time)
                    # Make sure to handle NaT values in Date
                    date_strings = predictions['Date'].dt.strftime('%Y-%m-%d').fillna('')
                    predictions['datetime'] = date_strings + ' ' + predictions['game_time'].fillna('00:00')
                elif 'Date' in predictions.columns:
                    predictions['datetime'] = predictions['Date']
                    predictions['display_time'] = ''
                else:
                    # If no date/time columns, create placeholder
                    predictions['datetime'] = pd.Timestamp.now()
                    predictions['display_time'] = ''
                
                # Don't filter out past games if there would be no games left
                try:
                    current_time = pd.Timestamp.now()
                    future_games = predictions[pd.to_datetime(predictions['datetime'], errors='coerce') >= current_time]
                    if len(future_games) > 0:
                        predictions = future_games
                    predictions = predictions.sort_values('datetime', ascending=True, na_position='first')
                except Exception as e:
                    print(f"[DEBUG] Error filtering/sorting by date: {e}")
                    # Continue without filtering if there's an error
                
                # Sort predictions based on sort dropdown
                sort_by = sort_var.get()
                if sort_by == "Date/Time":
                    predictions = predictions.sort_values('datetime', ascending=True, na_position='first')
                elif sort_by == "Edge":
                    predictions = predictions.sort_values('Edge', ascending=False)
                elif sort_by == "Home%":
                    predictions = predictions.sort_values('Home%', ascending=False)
                elif sort_by == "Away%":
                    predictions = predictions.sort_values('Away%', ascending=False)
                
                # Update status
                row_count = len(predictions)
                status_label.config(text=f"Showing {row_count} games for {selected_display_name}")
                
                # Populate treeview
                for _, row in predictions.iterrows():
                    date_str = row['Date'].strftime('%Y-%m-%d') if 'Date' in row and not pd.isna(row['Date']) else ''
                    time_str = row['display_time'] if 'display_time' in row else ''
                    datetime_display = f"{date_str} {time_str}".strip()
                    
                    matchup = f"{row.get('Away','')} @ {row.get('Home','')}"
                    
                    # Determine row tag for styling
                    tag = "normal"
                    if row.get('Bet?', False) == True:
                        edge = float(row.get('Edge', 0))
                        if abs(edge) >= 0.1:
                            tag = "strong_bet"
                        elif abs(edge) >= 0.05:
                            tag = "good_bet"
                        else:
                            tag = "weak_bet"
                    
                    # Prepare row values
                    values = [
                        datetime_display,
                        matchup,
                        f"{float(row.get('Home%', 0))*100:.1f}%" if 'Home%' in row else '',
                        f"{float(row.get('Away%', 0))*100:.1f}%" if 'Away%' in row else '',
                        row.get('HomeML', ''),
                        row.get('AwayML', ''),
                        row.get('VegasHomeML', ''),
                        row.get('VegasAwayML', ''),
                        row.get('VegasSpread', ''),
                        row.get('PredSpread', ''),
                        "Yes" if row.get('Bet?', False) == True else "No",
                        row.get('BetTeam', ''),
                        f"{float(row.get('Edge', 0))*100:.1f}%" if 'Edge' in row else '',
                        row.get('BetSize', ''),
                        row.get('KellySize', ''),
                        row.get('Notes', '')
                    ]
                    
                    tree.insert('', tk.END, values=values, tags=(tag,))
                    
            except Exception as e:
                status_label.config(text=f"Error loading predictions: {str(e)}")
                print(f"Error loading predictions: {e}")
        
        # Bind dropdown selection to load predictions
        sport_dropdown.bind("<<ComboboxSelected>>", lambda _: load_predictions())
        sort_options.bind("<<ComboboxSelected>>", lambda _: load_predictions())
        
        # Load predictions for initially selected sport
        load_predictions()

    def view_aoi_sports_dashboard(self):
        """Display the AOI Sports Analytics Dashboard"""
        try:
            # Create new window for AOI Sports Dashboard
            aoi_window = tk.Toplevel(self.root)
            aoi_window.title("AOI Sports Analytics Dashboard")
            aoi_window.geometry("1200x800")
            aoi_window.configure(bg=self.bg_color)

            # Add header
            header_frame = tk.Frame(aoi_window, bg=self.bg_color, pady=10)
            header_frame.pack(fill=tk.X)
            
            tk.Label(
                header_frame,
                text="AOI Sports Analytics Dashboard",
                font=STYLES['heading']['font'],
                bg=self.bg_color,
                fg=COLORS['text_primary']
            ).pack(side=tk.LEFT, padx=10)

            # Add placeholder text for now
            tk.Label(
                aoi_window,
                text="AOI Sports Analytics Dashboard coming soon...",
                font=STYLES['text']['font'],
                bg=self.bg_color,
                fg=COLORS['text_primary']
            ).pack(expand=True)

        except Exception as e:
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, f"\n[ERROR] Failed to open AOI Sports Dashboard: {e}\n")
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)

class ResponsiveLayout:
    def __init__(self, parent):
        self.parent = parent
        self.min_width = MIN_WINDOW_WIDTH
        self.min_height = MIN_WINDOW_HEIGHT
        
        # Configure grid weights
        self.parent.grid_rowconfigure(1, weight=3)  # Content area expands most
        self.parent.grid_rowconfigure(2, weight=1)  # Log area expands less
        self.parent.grid_columnconfigure(0, weight=1)
        
        # Bind resize event
        self.parent.bind('<Configure>', self.on_resize)
        
    def on_resize(self, event):
        # Ensure minimum dimensions
        if event.widget == self.parent:
            width = max(event.width, self.min_width)
            height = max(event.height, self.min_height)
            
            if width != event.width or height != event.height:
                self.parent.geometry(f"{width}x{height}")
                
            # Calculate proportional dimensions
            content_height = int(height * 0.6)  # 60% for content
            log_height = int(height * 0.2)     # 20% for log
            header_height = height - content_height - log_height
            
            # Update minimum sizes
            for widget in self.parent.winfo_children():
                grid_info = widget.grid_info()
                if grid_info:
                    row = grid_info['row']
                    if row == 0:  # Header
                        widget.configure(height=header_height)
                    elif row == 1:  # Content
                        widget.configure(height=content_height)
                    elif row == 2:  # Log
                        widget.configure(height=log_height)

# --- Main Execution with Persistent Geometry ---
def main():
    root = tk.Tk()
    config_file = "window_config.txt"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            root.geometry(f.read().strip())
    
    app = NRFIDashboard(root)
    
    def on_closing():
        geometry = root.winfo_geometry()
        with open(config_file, "w") as f:
            f.write(geometry)
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()

