import requests
import json
from datetime import datetime, timezone

# Constants for NASA POWER API
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Function to fetch temperature data
def fetch_temperature_data(lat: float, lon: float, start_date: str, end_date: str):
    params = {
        "start": start_date,
        "end": end_date,
        "latitude": lat,
        "longitude": lon,
        "community": "SB",
        "parameters": "T2M",
        "format": "CSV",  # Change to CSV format
        "time-standard": "UTC"
    }
    
    try:
        print(f"Fetching temperature data for location ({lat}, {lon}) from {start_date} to {end_date}")
        response = requests.get(NASA_POWER_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.text  # Get the response as text for CSV
        print("Successfully fetched temperature data")
        
        # Print temperature data
        print("Temperature Data (CSV):")
        print(data)
    except Exception as e:
        print(f"Error fetching temperature data: {e}")

# Sample execution
if __name__ == "__main__":
    # Example coordinates and date range
    latitude = 0  # Example latitude
    longitude = 0  # Example longitude
    start_date = "20250310"  # Example start date
    end_date = "20250311"  # Example end date
    
    fetch_temperature_data(latitude, longitude, start_date, end_date) 