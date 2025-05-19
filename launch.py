#!/usr/bin/env python
"""
Launcher script for Baseball Analysis Dashboard
Automatically sets up and activates a virtual environment before running the dashboard.
Works on both Windows and Mac.
"""
import os
import sys
import platform
import subprocess
import shutil

# Base directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")

def is_venv():
    """Check if the script is running in a virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def is_windows():
    """Check if running on Windows"""
    return platform.system() == "Windows"

def is_mac():
    """Check if running on Mac"""
    return platform.system() == "Darwin"

def create_venv():
    """Create a virtual environment in the .venv directory"""
    print(f"Creating virtual environment in {VENV_DIR}...")
    
    # Check if venv module is available
    try:
        import venv
    except ImportError:
        print("Error: The 'venv' module is not available. Please install Python 3.3+ or virtualenv.")
        sys.exit(1)
    
    # Create the virtual environment
    try:
        venv.create(VENV_DIR, with_pip=True)
        return True
    except Exception as e:
        print(f"Error creating virtual environment: {e}")
        return False

def update_requirements():
    """Run the requirements analyzer to ensure all imports are in requirements.txt"""
    update_script = os.path.join(BASE_DIR, "update_requirements.py")
    
    # Only run if the update script exists
    if os.path.exists(update_script):
        try:
            print("Checking for missing requirements...")
            subprocess.check_call([sys.executable, update_script])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Note: Failed to update requirements: {e}")
            # Continue anyway with the existing requirements.txt
    return False

def install_requirements():
    """Install requirements in the virtual environment"""
    print("Installing required packages...")
    
    # Fix requirements file formatting if needed
    with open(REQUIREMENTS_FILE, 'r') as f:
        requirements = f.read()
    
    # Check if file starts with "// filepath:" and remove that line if present
    if requirements.startswith("// filepath:"):
        with open(REQUIREMENTS_FILE, 'w') as f:
            f.write('\n'.join(requirements.split('\n')[1:]))
    
    # Get the path to pip in the virtual environment
    if is_windows():
        pip_path = os.path.join(VENV_DIR, "Scripts", "pip")
    else:
        pip_path = os.path.join(VENV_DIR, "bin", "pip")
    
    # Install requirements
    try:
        subprocess.check_call([pip_path, "install", "-r", REQUIREMENTS_FILE])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def launch_dashboard():
    """Launch the Dashboard.py script from within the virtual environment"""
    dashboard_path = os.path.join(BASE_DIR, "Dashboard.py")
    
    # Get the path to the Python interpreter in the virtual environment
    if is_windows():
        python_path = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        python_path = os.path.join(VENV_DIR, "bin", "python")
    
    print(f"Launching dashboard using {python_path}...")
    
    # Execute the dashboard script with the virtual environment's Python
    try:
        subprocess.check_call([python_path, dashboard_path])
    except subprocess.CalledProcessError as e:
        print(f"Error launching dashboard: {e}")
        sys.exit(1)

def main():
    """Main entry point for the launcher"""
    print("Baseball Analysis Dashboard Launcher")
    print("===================================")
    
    # If we're already in a virtual environment, just run the dashboard
    if is_venv():
        print("Already running in virtual environment")
        launch_dashboard()
        return
    
    # Update requirements.txt if needed
    update_requirements()
    
    # Check if virtual environment exists
    venv_exists = os.path.exists(VENV_DIR)
    venv_valid = False
    
    if is_windows():
        venv_valid = os.path.exists(os.path.join(VENV_DIR, "Scripts", "python.exe"))
    else:
        venv_valid = os.path.exists(os.path.join(VENV_DIR, "bin", "python"))
    
    # Create or update the virtual environment
    if not venv_exists or not venv_valid:
        if os.path.exists(VENV_DIR):
            print("Removing existing virtual environment...")
            shutil.rmtree(VENV_DIR)
        
        if not create_venv():
            print("Failed to create virtual environment. Exiting.")
            sys.exit(1)
            
        if not install_requirements():
            print("Failed to install required packages. Exiting.")
            sys.exit(1)
    
    # Launch Dashboard.py using the virtual environment
    launch_dashboard()

if __name__ == "__main__":
    main()
