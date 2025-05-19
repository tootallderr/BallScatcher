import os
import csv
import glob
from datetime import datetime
import re
from collections import defaultdict

# Define sports categories and their corresponding leagues
LEAGUE_NAME_MAP = {
    'eng.1': 'epl', 'esp.1': 'laliga', 'ger.1': 'bundesliga', 'ita.1': 'seriea',
    'fra.1': 'ligue1', 'usa.1': 'mls', 'mex.1': 'ligamx', 'por.1': 'primeiraliga',
    'uefa.champions': 'champions-league', 'uefa.europa': 'europa-league',
    'mens-college-basketball': 'mens-college-basketball',
    'womens-college-basketball': 'womens-college-basketball',
    'college-football': 'college-football',
    'nba': 'nba', 'nhl': 'nhl', 'mlb': 'mlb', 'wnba': 'wnba', 'nfl': 'nfl',
    'mma': 'mma'
}

# Group leagues by sport category
SPORT_CATEGORIES = {
    'soccer': ['epl', 'laliga', 'bundesliga', 'seriea', 'ligue1', 'mls', 'ligamx', 
               'primeiraliga', 'champions-league', 'europa-league'],
    'basketball': ['nba', 'wnba', 'mens-college-basketball', 'womens-college-basketball'],
    'football': ['nfl', 'college-football'],
    'baseball': ['mlb'],
    'hockey': ['nhl'],
    'other': ['mma']
}

# Files to exclude from the reports
EXCLUDED_FILES = [
    'debug_team_metrics.csv',
    'mlb_game_logs_partial.csv',
    'confusion_matrix.csv',
    'sample_predictions.csv',
    'MLB_upcoming_predictions.csv'
]

# Create reverse mapping for easier lookup
LEAGUE_TO_CATEGORY = {}
for category, leagues in SPORT_CATEGORIES.items():
    for league in leagues:
        LEAGUE_TO_CATEGORY[league] = category

def should_include_file(file_path):
    """
    Check if the file should be included in the report.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        bool: True if the file should be included, False otherwise
    """
    filename = os.path.basename(file_path)
    return filename.lower() not in [f.lower() for f in EXCLUDED_FILES]

def find_csv_files(root_dir='.'):
    """
    Find all CSV files in the given directory and its subdirectories.
    
    Args:
        root_dir (str): The root directory to start the search
        
    Returns:
        list: List of paths to CSV files
    """
    csv_files = []
    for dirpath, _, _ in os.walk(root_dir):
        # Find all .csv files in the current directory
        files = glob.glob(os.path.join(dirpath, '*.csv'))
        # Filter out excluded files
        files = [f for f in files if should_include_file(f)]
        csv_files.extend(files)
    return csv_files

def get_csv_structure(csv_path):
    """
    Extract header, a sample row, and missing data statistics from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        tuple: (headers, sample_row, missing_stats, error_message)
    """
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader, None)  # Get the header row
            
            if not headers:
                return None, None, None, "No headers found"
                
            # Initialize counters for missing values
            total_rows = 0
            missing_counts = [0] * len(headers)
            
            # Try to get the first data row
            sample_row = next(reader, None)
            
            # If we have a sample row, start counting missing values
            if sample_row:
                total_rows = 1
                for i, value in enumerate(sample_row):
                    if i < len(missing_counts) and (not value or value.strip() == ''):
                        missing_counts[i] += 1
            
            # Continue reading the file to count missing values
            for row in reader:
                total_rows += 1
                for i, value in enumerate(row):
                    if i < len(missing_counts) and (not value or value.strip() == ''):
                        missing_counts[i] += 1
            
            # Calculate missing percentages
            missing_stats = []
            if total_rows > 0:
                for i, count in enumerate(missing_counts):
                    if i < len(headers):
                        missing_pct = (count / total_rows) * 100
                        missing_stats.append((headers[i], count, missing_pct))
            
            return headers, sample_row, missing_stats, None
    except Exception as e:
        return None, None, None, str(e)

def identify_league_from_path(filepath):
    """
    Identify the league based on the filepath and filename.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        tuple: (league, category)
    """
    filepath_lower = filepath.lower()
    
    # Check for exact league matches in path
    for league in LEAGUE_NAME_MAP.values():
        if league.lower() in filepath_lower:
            category = LEAGUE_TO_CATEGORY.get(league, 'other')
            return league, category
    
    # If no direct match, try to infer from directory structure
    path_parts = os.path.normpath(filepath).split(os.sep)
    for part in path_parts:
        part_lower = part.lower()
        for league in LEAGUE_NAME_MAP.values():
            if league.lower() in part_lower:
                category = LEAGUE_TO_CATEGORY.get(league, 'other')
                return league, category
    
    # If still no match, check for sport category matches
    for category, keywords in {
        'soccer': ['football', 'soccer'],
        'basketball': ['basketball', 'bball', 'hoops'],
        'football': ['football', 'nfl', 'ncaa'],
        'baseball': ['baseball', 'mlb'],
        'hockey': ['hockey', 'nhl'],
    }.items():
        for keyword in keywords:
            if keyword in filepath_lower:
                return 'unknown', category
    
    return 'unknown', 'other'

def get_header_signature(headers):
    """
    Create a signature for headers to group similar structures.
    
    Args:
        headers (list): List of header names
        
    Returns:
        tuple: Hashable signature of headers
    """
    if not headers:
        return tuple()
    return tuple(headers)

def write_structure_report_organized(csv_files, output_dir):
    """
    Write structure reports organized by sport category and league.
    
    Args:
        csv_files (list): List of paths to CSV files
        output_dir (str): Base directory for output reports
    """
    # Create main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize files by category and league
    categorized_files = defaultdict(lambda: defaultdict(list))
    for csv_path in csv_files:
        league, category = identify_league_from_path(csv_path)
        categorized_files[category][league].append(csv_path)
    
    # Also create a category index file
    with open(os.path.join(output_dir, "index.txt"), 'w', encoding='utf-8') as index_file:
        index_file.write("CSV Structure Reports Index\n")
        index_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        index_file.write("=" * 80 + "\n\n")
        
        # Process each category
        for category in sorted(categorized_files.keys()):
            # Create category directory
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            index_file.write(f"{category.upper()} CATEGORY\n")
            index_file.write("-" * 80 + "\n")
            
            # Generate category summary
            with open(os.path.join(category_dir, f"{category}_summary.txt"), 'w', encoding='utf-8') as cat_summary:
                cat_summary.write(f"{category.upper()} CSV Structure Summary\n")
                cat_summary.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                cat_summary.write("=" * 80 + "\n\n")
                
                # Process each league in the category
                for league, league_files in sorted(categorized_files[category].items()):
                    index_file.write(f"  - {league}: {len(league_files)} files\n")
                    cat_summary.write(f"{league.upper()}: {len(league_files)} files\n")
                    
                    # Create league directory
                    league_dir = os.path.join(category_dir, league)
                    os.makedirs(league_dir, exist_ok=True)
                    
                    # Group files by header structure
                    header_groups = defaultdict(list)
                    file_structures = {}
                    
                    for csv_path in league_files:
                        headers, sample_row, missing_stats, error = get_csv_structure(csv_path)
                        if not error:
                            header_sig = get_header_signature(headers)
                            header_groups[header_sig].append(csv_path)
                            file_structures[csv_path] = (headers, sample_row, missing_stats)
                        else:
                            # Add files with errors to a special group
                            header_groups['errors'].append((csv_path, error))
                    
                    # Write detailed report for this league
                    league_report_path = os.path.join(league_dir, f"{league}_structure_report.txt")
                    with open(league_report_path, 'w', encoding='utf-8') as league_report:
                        league_report.write(f"{league.upper()} CSV Structure Report\n")
                        league_report.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        league_report.write("=" * 80 + "\n\n")
                        
                        # Process each header group
                        for i, (header_sig, group_files) in enumerate(header_groups.items(), 1):
                            if header_sig == 'errors':
                                # Handle files with errors
                                league_report.write(f"Files with Errors:\n")
                                league_report.write("-" * 80 + "\n")
                                for csv_path, error in group_files:
                                    league_report.write(f"   {os.path.basename(csv_path)}: {error}\n")
                                league_report.write("\n" + "=" * 80 + "\n\n")
                                continue
                            
                            # Skip empty groups
                            if not group_files:
                                continue
                            
                            league_report.write(f"Header Structure {i}:\n")
                            league_report.write("-" * 80 + "\n")
                            
                            # List the files with this structure
                            league_report.write(f"Files with this structure ({len(group_files)}):\n")
                            for j, filepath in enumerate(group_files, 1):
                                league_report.write(f"   {j}. {os.path.basename(filepath)}\n")
                            league_report.write("\n")
                            
                            # Show the headers
                            headers = file_structures[group_files[0]][0]
                            if headers:
                                league_report.write("Headers:\n")
                                for j, header in enumerate(headers):
                                    league_report.write(f"   {j+1}. {header}\n")
                            else:
                                league_report.write("   No headers found\n")
                            
                            # Show a sample data row from the first file
                            sample_file = group_files[0]
                            sample_row = file_structures[sample_file][1]
                            missing_stats = file_structures[sample_file][2]
                            
                            # Display missing value statistics
                            if missing_stats:
                                league_report.write("\nMissing Data Analysis:\n")
                                league_report.write("   Column                            Missing Count     Missing %\n")
                                league_report.write("   " + "-" * 65 + "\n")
                                
                                for header, count, percentage in missing_stats:
                                    # Format the output with proper alignment
                                    league_report.write(f"   {header:<30} {count:>10}    {percentage:>8.2f}%\n")
                            

                            league_report.write("\n" + "=" * 80 + "\n\n")
                            
                            # Add a note to the category summary
                            if j == 1:  # First header pattern
                                cat_summary.write(f"  - {len(header_sig)} columns including: ")
                                if headers and len(headers) > 0:
                                    preview_headers = headers[:5] if len(headers) > 5 else headers
                                    cat_summary.write(", ".join(preview_headers))
                                    if len(headers) > 5:
                                        cat_summary.write(f", ... ({len(headers)-5} more)")
                                cat_summary.write("\n")
                
                cat_summary.write("\n" + "=" * 80 + "\n")
            
            index_file.write("\n")
        
    # Generate a combined report with all files
    combined_output = os.path.join(output_dir, "all_csv_structure_report.txt")
    write_structure_report(csv_files, combined_output)
    print(f"Combined report generated: {combined_output}")

def write_structure_report(csv_files, output_file):
    """
    Write the structure information of all CSV files to a text file.
    
    Args:
        csv_files (list): List of paths to CSV files
        output_file (str): Path to the output text file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"CSV Structure Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, csv_path in enumerate(sorted(csv_files), 1):
            f.write(f"{i}. File: {csv_path}\n")
            f.write("-" * 80 + "\n")
            
            headers, sample_row, missing_stats, error = get_csv_structure(csv_path)
            
            if error:
                f.write(f"   Error reading file: {error}\n\n")
                continue
            
            if headers:
                f.write("   Headers:\n")
                for j, header in enumerate(headers):
                    f.write(f"   {j+1}. {header}\n")
            else:
                f.write("   No headers found\n")
            
            # Display missing value statistics
            if missing_stats:
                f.write("\n   Missing Data Analysis:\n")
                f.write("      Column                            Missing Count     Missing %\n")
                f.write("      " + "-" * 65 + "\n")
                
                for header, count, percentage in missing_stats:
                    # Format the output with proper alignment
                    f.write(f"      {header:<30} {count:>10}    {percentage:>8.2f}%\n")
            
            if sample_row:
                f.write("\n   Sample Data Row:\n")
                for j, (header, value) in enumerate(zip(headers if headers else [], sample_row)):
                    header_text = f"{header}: " if headers else ""
                    missing_marker = " [MISSING]" if not value or value.strip() == '' else ""
                    f.write(f"   {j+1}. {header_text}{value}{missing_marker}\n")
            else:
                f.write("\n   No data rows found\n")
            
            f.write("\n" + "=" * 80 + "\n\n")

def analyze_csv_content(csv_content, filename):
    """
    Analyze CSV content provided as a string
    """
    import io
    csvfile = io.StringIO(csv_content)
    reader = csv.reader(csvfile)
    headers = next(reader, None)  # Get the header row
    sample_row = next(reader, None)  # Get first data row
    return headers, sample_row, None

# Example usage:
batters_content = """..."""  # Content from 1stInningBattersHistorical.csv
pitchers_content = """..."""  # Content from 1stInningPitchersHistorical.csv

# Analyze each file
for filename, content in [
    ("1stInningBattersHistorical.csv", batters_content),
    ("1stInningPitchersHistorical.csv", pitchers_content)
]:
    headers, sample_row, error = analyze_csv_content(content, filename)
    if headers:
        print(f"\nAnalyzing {filename}:")
        print("Headers:", headers)
        print("Sample row:", sample_row)

def main():
    # Define the root directory - the current directory by default
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all CSV files
    print(f"Searching for CSV files in {root_dir} and subdirectories...")
    csv_files = find_csv_files(root_dir)
    
    if not csv_files:
        print("No CSV files found.")
        return
    
    print(f"Found {len(csv_files)} CSV files (excluding specified debug and test files).")
    
    # Output directory path
    output_dir = os.path.join(root_dir, "csv_reports")
    
    # Generate the organized structure reports
    print(f"Generating organized structure reports...")
    write_structure_report_organized(csv_files, output_dir)
    
    print(f"Reports generated successfully in: {output_dir}")
    print("Overall structure:")
    print(f"  - Main index: {os.path.join(output_dir, 'index.txt')}")
    print(f"  - Complete report: {os.path.join(output_dir, 'all_csv_structure_report.txt')}")
    print("  - Organized by category and league in subfolders")

if __name__ == "__main__":
    main()
