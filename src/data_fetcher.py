import requests
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import streamlit as st

class DataFetcher:
    def __init__(self):
        self.windborne_base_url = "https://a.windbornesystems.com/treasure"
        self.opensky_base_url = "https://opensky-network.org/api"
        self.cache = {}
        self.last_fetch_time = None
        self.fetch_interval = 300  # 5 minutes in seconds
        print("DataFetcher initialized")

    def fetch_windborne_data(self, hours_ago):
        """Fetch balloon data from WindBorne API."""
        url = f"{self.windborne_base_url}/{hours_ago:02d}.json"
        print(f"\nFetching balloon data for {hours_ago} hours ago from {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"Successfully fetched data for {hours_ago}h ago. Found {len(data)} balloons.")
            print(f"Sample data: {data[:2] if data else 'No data'}")
            return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching data for {hours_ago}h ago: {e}")
            return None

    def fetch_opensky_data(self, bounds=None):
        """Fetch aircraft data from OpenSky Network API."""
        params = {}
        if bounds:
            # Format: [min_latitude, max_latitude, min_longitude, max_longitude]
            params.update({
                'lamin': bounds[0],
                'lamax': bounds[1],
                'lomin': bounds[2],
                'lomax': bounds[3]
            })
        
        try:
            response = requests.get(f"{self.opensky_base_url}/states/all", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching OpenSky data: {e}")
            return None

    def process_windborne_data(self, raw_data, timestamp):
        """Process raw WindBorne data into a DataFrame."""
        if not raw_data:
            return pd.DataFrame()

        records = []
        for balloon in raw_data:
            try:
                if isinstance(balloon, list) and len(balloon) >= 3:
                    lat, lon, alt = balloon
                    records.append({
                        'id': f'balloon_{len(records)}',
                        'latitude': float(lat),
                        'longitude': float(lon),
                        'altitude': float(alt) * 1000,  # Convert km to meters
                        'timestamp': timestamp,
                        'type': 'balloon'
                    })
                elif isinstance(balloon, dict) and all(k in balloon for k in ['lat', 'lon', 'alt']):
                    records.append({
                        'id': balloon.get('id', f'balloon_{len(records)}'),
                        'latitude': float(balloon['lat']),
                        'longitude': float(balloon['lon']),
                        'altitude': float(balloon['alt']) * 1000,  # Convert km to meters
                        'timestamp': timestamp,
                        'type': 'balloon'
                    })
            except (ValueError, TypeError, IndexError) as e:
                print(f"Error processing balloon data: {e}")
                continue

        return pd.DataFrame(records)

    def process_opensky_data(self, raw_data):
        """Process raw OpenSky data into a DataFrame."""
        if not raw_data or 'states' not in raw_data:
            return pd.DataFrame()

        records = []
        for state in raw_data['states']:
            try:
                if state[6] and state[5]:  # Check if latitude and longitude are not None
                    records.append({
                        'id': state[0],  # ICAO24 address
                        'latitude': float(state[6]),
                        'longitude': float(state[5]),
                        'altitude': float(state[7]) if state[7] else 0,  # geometric altitude in meters
                        'timestamp': state[3],  # last contact
                        'type': 'aircraft',
                        'velocity': state[9],  # velocity in m/s
                        'heading': state[10],  # true track in decimal degrees
                        'vertical_rate': state[11]  # vertical rate in m/s
                    })
            except (ValueError, TypeError, IndexError) as e:
                print(f"Error processing aircraft data: {e}")
                continue

        return pd.DataFrame(records)

    def get_current_data(self):
        """Get current data from both WindBorne and OpenSky."""
        current_time = datetime.now(timezone.utc)
        
        # Check if we need to fetch new data
        if (self.last_fetch_time is None or 
            (current_time - self.last_fetch_time).total_seconds() >= self.fetch_interval):
            
            print("Fetching new data...")
            
            # Get WindBorne data for the last 24 hours
            balloon_frames = []
            for hours_ago in range(24):
                raw_data = self.fetch_windborne_data(hours_ago)
                timestamp = current_time - timedelta(hours=hours_ago)
                if raw_data:
                    df = self.process_windborne_data(raw_data, timestamp)
                    balloon_frames.append(df)
            
            # Combine all balloon data
            balloon_data = pd.concat(balloon_frames, ignore_index=True) if balloon_frames else pd.DataFrame()
            
            # Get current aircraft positions
            # Use the bounding box of balloon positions to limit aircraft data
            if not balloon_data.empty:
                bounds = [
                    balloon_data['latitude'].min() - 5,
                    balloon_data['latitude'].max() + 5,
                    balloon_data['longitude'].min() - 5,
                    balloon_data['longitude'].max() + 5
                ]
            else:
                bounds = None
            
            raw_aircraft_data = self.fetch_opensky_data(bounds)
            aircraft_data = self.process_opensky_data(raw_aircraft_data)
            
            # Combine both datasets
            combined_data = pd.concat([balloon_data, aircraft_data], ignore_index=True)
            
            # Update cache
            self.cache['data'] = combined_data
            self.last_fetch_time = current_time
            
            print(f"Fetched {len(balloon_data)} balloon positions and {len(aircraft_data)} aircraft positions")
            
            return combined_data
        
        return self.cache['data']

    def get_statistics(self):
        """Calculate statistics about balloons and aircraft."""
        data = self.get_current_data()
        if data.empty:
            return {}

        stats = {
            'total_balloons': len(data[data['type'] == 'balloon']['id'].unique()),
            'total_aircraft': len(data[data['type'] == 'aircraft']['id'].unique()),
            'avg_balloon_altitude': data[data['type'] == 'balloon']['altitude'].mean(),
            'avg_aircraft_altitude': data[data['type'] == 'aircraft']['altitude'].mean(),
            'last_update': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        }

        return stats 