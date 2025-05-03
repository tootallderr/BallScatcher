import requests
import json
from typing import Dict, Any, Union, List
import os
from datetime import datetime

def get_data_type(value: Any) -> str:
    """Determine the data type of a value."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "number"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, list):
        if not value:  # Empty list
            return "array (empty)"
        # For arrays, we'll check the first item to represent the type
        sample = value[0]
        if isinstance(sample, dict):
            return "array of objects"
        else:
            return f"array of {get_data_type(sample)}"
    elif isinstance(value, dict):
        return "object"
    else:
        return str(type(value).__name__)

def extract_schema(data: Any, max_depth: int = None, current_depth: int = 0) -> Union[Dict, str]:
    """
    Recursively extract schema from JSON data.
    
    Args:
        data: The JSON data to analyze
        max_depth: Maximum depth to traverse (None for unlimited)
        current_depth: Current recursion depth
    
    Returns:
        Schema representation
    """
    if max_depth is not None and current_depth >= max_depth:
        return f"{get_data_type(data)} (max depth reached)"
    
    if isinstance(data, dict):
        schema = {}
        for key, value in data.items():
            if isinstance(value, dict):
                schema[key] = extract_schema(value, max_depth, current_depth + 1)
            elif isinstance(value, list):
                if not value:  # Empty list
                    schema[key] = "array (empty)"
                elif isinstance(value[0], (dict, list)):
                    # For arrays of objects, we'll analyze just the first item for brevity
                    schema[key] = [extract_schema(value[0], max_depth, current_depth + 1)]
                else:
                    schema[key] = f"array of {get_data_type(value[0])}"
            else:
                schema[key] = get_data_type(value)
        return schema
    elif isinstance(data, list):
        if not data:
            return "array (empty)"
        if isinstance(data[0], (dict, list)):
            # For arrays, we'll analyze just the first item for brevity
            return [extract_schema(data[0], max_depth, current_depth + 1)]
        else:
            return f"array of {get_data_type(data[0])}"
    else:
        return get_data_type(data)

def format_schema(schema: Union[Dict, List, str], indent: int = 0) -> str:
    """Format the schema in a readable format."""
    indent_str = "  " * indent
    
    if isinstance(schema, dict):
        output = "{\n"
        for i, (key, value) in enumerate(schema.items()):
            output += f"{indent_str}  \"{key}\": "
            output += format_schema(value, indent + 1)
            if i < len(schema) - 1:
                output += ","
            output += "\n"
        output += f"{indent_str}}}"
        return output
    elif isinstance(schema, list):
        if isinstance(schema[0], (dict, list)):
            output = "[\n"
            output += f"{indent_str}  "
            output += format_schema(schema[0], indent + 1)
            output += "\n"
            output += f"{indent_str}]"
            return output
        else:
            return f"[{schema[0]}]"
    else:
        return f"\"{schema}\""

def fetch_and_analyze_json(url: str, max_depth: int = None) -> Dict:
    """
    Fetch JSON from URL and analyze its structure.
    
    Args:
        url: The URL to fetch JSON data from
        max_depth: Maximum depth to traverse (None for unlimited)
    
    Returns:
        Schema representation
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        json_data = response.json()
        schema = extract_schema(json_data, max_depth)
        return schema
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return {}
    except json.JSONDecodeError:
        print("Error: Invalid JSON received from URL")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}

def main():
    print("\n===== JSON Schema Extractor =====")
    url = input("\nEnter the URL to extract JSON schema from: ")
    
    try:
        max_depth_input = input("Enter maximum recursion depth (leave blank for unlimited): ")
        max_depth = int(max_depth_input) if max_depth_input.strip() else None
    except ValueError:
        print("Invalid depth value. Using unlimited depth.")
        max_depth = None
    
    print("\nFetching and analyzing JSON structure...")
    schema = fetch_and_analyze_json(url, max_depth)
    if schema:
        formatted_schema = format_schema(schema)
        
        # Generate a filename from the URL
        from urllib.parse import urlparse
        
        # Extract domain and path components for the filename
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace('.', '_').replace(':', '_')
        
        # Get path and remove trailing slash if present
        path = parsed_url.path.strip('/')
        
        # Take only the last part of the path or use 'api' if no path
        path_component = path.split('/')[-1] if path else 'api'
        
        # Clean up any special characters that might be problematic in filenames
        path_component = ''.join(c if c.isalnum() or c in '_-' else '_' for c in path_component)
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the filename
        filename = f"{domain}_{path_component}_{timestamp}.txt"
        
        # Save to file
        with open(filename, "w") as f:
            f.write(f"URL: {url}\n")
            f.write(f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(formatted_schema)
        
        print(f"\nSchema extraction complete! Saved to {os.path.abspath(filename)}")
        
        # Also display the schema
        print("\nSchema Preview:")
        print(formatted_schema[:1000] + "..." if len(formatted_schema) > 1000 else formatted_schema)
    else:
        print("\nFailed to extract schema.")

if __name__ == "__main__":
    main()
