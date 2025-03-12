import requests
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import streamlit as st
import os
import pickle
import xarray as xr

class DataFetcher:
    def __init__(self):
        self.windborne_base_url = "https://a.windbornesystems.com/treasure"
        self.opensky_base_url = "https://opensky-network.org/api"
        self.cache = {}
        self.last_fetch_time = None
        self.fetch_interval = 300  # 5 minutes in seconds
        self.weather_cache = self._load_weather_cache()
        self.weather_cache_timestamp = None
        print("DataFetcher initialized")
    
    def _load_weather_cache(self) -> Dict:
        """Load weather cache from file if it exists."""
        if os.path.exists("weather_cache.pkl"):
            try:
                with open("weather_cache.pkl", 'rb') as f:
                    cache = pickle.load(f)
                print(f"Loaded weather cache with {len(cache)} entries")
                return cache
            except Exception as e:
                print(f"Error loading cache: {e}")
        return {}
    
    def _save_weather_cache(self):
        """Save weather cache to file."""
        try:
            with open("weather_cache.pkl", 'wb') as f:
                pickle.dump(self.weather_cache, f)
            print("Weather cache saved successfully")
        except Exception as e:
            print(f"Error saving cache: {e}")

    def fetch_windborne_data(self, hours_ago):
        """Fetch balloon data from WindBorne API."""
        url = f"{self.windborne_base_url}/{hours_ago:02d}.json"
        print(f"\nFetching balloon data for {hours_ago} hours ago from {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"Successfully fetched data for {hours_ago}h ago. Found {len(data)} balloons.")
            print(f"Sample data: {data[:2] if data else 'No data'}")  # Print first two entries
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
                if isinstance(balloon, list) and len(balloon) >= 3:  # WindBorne data is in list format [lat, lon, alt]
                    lat, lon, alt = balloon
                    records.append({
                        'id': f'balloon_{len(records)}',  # Generate sequential balloon IDs
                        'latitude': float(lat),
                        'longitude': float(lon),
                        'altitude': float(alt) * 1000,  # Convert km to meters to match aircraft altitude
                        'timestamp': timestamp,
                        'type': 'balloon'
                    })
                elif isinstance(balloon, dict) and all(k in balloon for k in ['lat', 'lon', 'alt']):
                    records.append({
                        'id': balloon.get('id', f'balloon_{len(records)}'),
                        'latitude': float(balloon['lat']),
                        'longitude': float(balloon['lon']),
                        'altitude': float(balloon['alt']) * 1000,  # Convert km to meters to match aircraft altitude
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

    def fetch_weather_data(self, lat: float, lon: float, start_date: str, end_date: str) -> Dict:
        """
        Fetch weather data from GFS via NCEI for a specific location.
        """
        cache_key = f"{lat}_{lon}_{start_date}_{end_date}"
        print(f"\nFetching weather data for location ({lat}, {lon})")
        
        # Return cached data if less than 1 hour old
        if (self.weather_cache_timestamp and 
            (datetime.utcnow() - self.weather_cache_timestamp) < timedelta(hours=1) and
            cache_key in self.weather_cache):
            print("Using cached weather data")
            return self.weather_cache[cache_key]
        
        # Example of accessing GFS data
        try:
            # Construct the URL for the GFS data
            date_str = datetime.utcnow().strftime('%Y%m%d')
            hour_str = datetime.utcnow().strftime('%H')
            gfs_url = "https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date}/gfs_0p25_{hour}z".format(date=date_str, hour=hour_str)
            
            print(f"Accessing GFS data from {gfs_url}")
            ds = xr.open_dataset(gfs_url)
            
            # Extract relevant data (e.g., wind speed, temperature)
            # This is a simplified example; actual extraction will depend on the dataset structure
            wind_speed = ds['wind_speed'].sel(lat=lat, lon=lon, method='nearest').values
            temperature = ds['temperature'].sel(lat=lat, lon=lon, method='nearest').values
            
            # Cache the results
            self.weather_cache[cache_key] = {
                'wind_speed': wind_speed,
                'temperature': temperature
            }
            self.weather_cache_timestamp = datetime.utcnow()
            
            print("Successfully fetched GFS weather data")
            return self.weather_cache[cache_key]
        except Exception as e:
            print(f"Error fetching GFS weather data: {e}")
            return {}

    def get_historical_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch and process 24 hours of balloon data.
        Returns a DataFrame with columns: timestamp, balloon_id, lat, lon, alt
        """
        print("\nStarting historical data collection...")
        current_time = datetime.utcnow()
        
        # Return cached data if it's less than 5 minutes old
        if not force_refresh and self.last_fetch_time and \
           (current_time - self.last_fetch_time) < timedelta(minutes=5):
            print("Using cached historical data")
            return self.cache['data']
        
        print("Fetching fresh data for all time points...")
        all_data = []
        total_balloons = 0
        invalid_records = 0
        
        # Create a progress bar
        progress_text = "Fetching balloon constellation data..."
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            for hours_ago in range(24):
                status.text(f"{progress_text} ({hours_ago}/24 hours processed)")
                progress_bar.progress((hours_ago + 1) / 24)
                
                data = self.fetch_windborne_data(hours_ago)
                timestamp = current_time - timedelta(hours=hours_ago)
                
                if data:
                    for balloon_id, coords in enumerate(data):
                        try:
                            lat, lon, alt = coords
                            # Validate coordinates
                            lat = float(lat)
                            lon = float(lon)
                            alt = float(alt)
                            
                            # Check for valid ranges and NaN values
                            if (not -90 <= lat <= 90 or 
                                not -180 <= lon <= 180 or 
                                alt < 0 or 
                                np.isnan(lat) or np.isnan(lon) or np.isnan(alt)):
                                invalid_records += 1
                                continue
                                
                            all_data.append({
                                'timestamp': timestamp,
                                'balloon_id': balloon_id,
                                'lat': lat,
                                'lon': lon,
                                'alt': alt
                            })
                            total_balloons += 1
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"Error processing balloon {balloon_id} data: {e}")
                            invalid_records += 1
                            continue
                
                # Add small delay to avoid overwhelming the API
                time.sleep(0.1)
            
            print(f"\nProcessed {total_balloons} total balloon positions across 24 hours")
            print(f"Filtered out {invalid_records} invalid records")
            
            if not all_data:
                raise ValueError("No valid balloon data found")
                
            status.text("Creating DataFrame...")
            df = pd.DataFrame(all_data)
            
            # Add weather data for each unique location
            if len(df) > 0:
                status.text("Fetching weather data...")
                df = self._add_weather_data(df)
                print("Weather data collection complete")
            else:
                print("No balloon data to process for weather")
            
            # Cache the results
            self.cache['data'] = df
            self.last_fetch_time = current_time
            
            print(f"\nFinal DataFrame shape: {df.shape}")
            print(f"Columns available: {df.columns.tolist()}")
            
            # Clear the status indicators
            status.empty()
            progress_bar.empty()
            
            return df
            
        except Exception as e:
            status.error(f"Error during data collection: {str(e)}")
            progress_bar.empty()
            raise e

    def _add_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather data to the balloon dataframe."""
        print("\nPreparing to add weather data...")
        
        weather_status = st.empty()
        weather_progress = st.progress(0)
        weather_status.text(f"Processing weather data (using 20° grid resolution - hemispheric scale)...")
        
        try:
            # Filter out any remaining invalid coordinates
            df = df.dropna(subset=['lat', 'lon'])
            
            # Round coordinates to just North/South hemispheres
            df['lat_round'] = np.where(pd.isna(df['lat']), 0, np.sign(df['lat']) * 20)  # Use 20° instead of 90° for better average
            df['lon_round'] = 0  # Single longitude reference point
            
            # Get unique grid cells (should be at most 2: North and South hemispheres)
            unique_locations = df[['lat_round', 'lon_round']].drop_duplicates()
            total_locations = len(unique_locations)
            print(f"Found {total_locations} hemispheres to fetch weather data for")
            
            if total_locations > 2:  # Reduced threshold for warning
                weather_status.warning(f"Large number of hemispheres ({total_locations}) even with 20° resolution.")
            
            # Get date range
            start_date = df['timestamp'].min().strftime('%Y%m%d')
            end_date = df['timestamp'].max().strftime('%Y%m%d')
            print(f"Date range: {start_date} to {end_date}")
            
            # Fetch weather data for each grid cell
            weather_data = {}
            locations_to_fetch = []
            
            # First check cache
            cached_count = 0
            for idx, loc in unique_locations.iterrows():
                if pd.isna(loc['lat_round']) or pd.isna(loc['lon_round']):
                    continue
                    
                cache_key = f"{loc['lat_round']}_{loc['lon_round']}_{start_date}_{end_date}"
                if cache_key in self.weather_cache:
                    weather_data[(loc['lat_round'], loc['lon_round'])] = self.weather_cache[cache_key]
                    cached_count += 1
                    weather_progress.progress(cached_count / total_locations)
                else:
                    locations_to_fetch.append(loc)
            
            print(f"Found {len(weather_data)} locations in cache, need to fetch {len(locations_to_fetch)} new locations")
            
            # Fetch new locations
            for idx, loc in enumerate(locations_to_fetch):
                if pd.isna(loc['lat_round']) or pd.isna(loc['lon_round']):
                    continue
                    
                weather_status.text(f"Fetching weather data for grid cell {idx + 1}/{len(locations_to_fetch)}")
                progress = min(1.0, (cached_count + idx + 1) / total_locations)
                weather_progress.progress(progress)
                
                data = self.fetch_weather_data(loc['lat_round'], loc['lon_round'], start_date, end_date)
                if data and 'properties' in data and 'parameter' in data['properties']:
                    cache_key = f"{loc['lat_round']}_{loc['lon_round']}_{start_date}_{end_date}"
                    self.weather_cache[cache_key] = data['properties']['parameter']
                    weather_data[(loc['lat_round'], loc['lon_round'])] = data['properties']['parameter']
                    if (idx + 1) % 10 == 0:  # Save cache periodically
                        self._save_weather_cache()
                
                if idx < len(locations_to_fetch) - 1:  # Don't sleep after last request
                    time.sleep(0.5)  # Reduced delay between requests
            
            # Save final cache
            self._save_weather_cache()
            
            # Add weather data to the dataframe
            weather_status.text("Merging weather data with balloon positions...")
            weather_cols = ['WS50M', 'WD50M', 'T2M', 'PS', 'RH2M']
            for col in weather_cols:
                df[col] = None
            
            # Update progress while merging data
            for idx, ((lat, lon), data) in enumerate(weather_data.items()):
                weather_progress.progress(min(0.99, idx / len(weather_data)))  # Leave room for final step
                mask = (df['lat_round'] == lat) & (df['lon_round'] == lon)
                for col in weather_cols:
                    if col in data:
                        try:
                            times = pd.to_datetime(list(data[col].keys()), format='%Y%m%d%H')
                            values = list(data[col].values())
                            if len(times) > 0:
                                df.loc[mask, col] = np.interp(
                                    df[mask]['timestamp'].astype(np.int64),
                                    times.astype(np.int64),
                                    values
                                )
                        except (ValueError, TypeError) as e:
                            print(f"Error interpolating {col}: {e}")
                            continue
            
            # Final progress update
            weather_progress.progress(1.0)
            
            # Drop temporary columns and invalid data
            df = df.drop(['lat_round', 'lon_round'], axis=1)
            df = df.dropna(subset=['lat', 'lon', 'alt'])  # Remove any rows with NaN coordinates
            print("\nWeather data integration complete")
            
            # Clear the status indicators
            weather_status.empty()
            weather_progress.empty()
            
            return df
            
        except Exception as e:
            weather_status.error(f"Error processing weather data: {str(e)}")
            weather_progress.empty()
            raise e

    def get_balloon_trajectories(self) -> Dict[int, pd.DataFrame]:
        """Get trajectory data grouped by balloon ID, handling map wrap-around."""
        print("\nGetting balloon trajectories...")
        df = self.get_historical_data()
        
        # Filter to show only the top 10 balloons with the most data points
        top_balloons = df['balloon_id'].value_counts().nlargest(10).index
        df = df[df['balloon_id'].isin(top_balloons)]
        
        trajectories = {}
        for bid, group in df.groupby('balloon_id'):
            group = group.sort_values('timestamp')
            # Handle wrap-around by adjusting longitudes
            lon_diff = group['lon'].diff().fillna(0)
            group.loc[lon_diff > 180, 'lon'] -= 360
            group.loc[lon_diff < -180, 'lon'] += 360
            trajectories[bid] = group
        print(f"Found trajectories for {len(trajectories)} balloons")
        return trajectories

    def calculate_balloon_stats(self) -> Dict[int, Dict[str, float]]:
        """Calculate statistics for each balloon."""
        print("\nCalculating balloon statistics...")
        trajectories = self.get_balloon_trajectories()
        stats = {}
        
        for balloon_id, traj in trajectories.items():
            print(f"\nProcessing statistics for balloon {balloon_id}")
            if len(traj) < 2:
                print(f"Insufficient data points for balloon {balloon_id}")
                continue
                
            # Calculate various metrics
            total_distance = self._calculate_distance_traveled(traj)
            avg_speed = total_distance / ((traj['timestamp'].max() - traj['timestamp'].min()).total_seconds() / 3600)
            avg_alt = traj['alt'].mean()
            alt_change = traj['alt'].max() - traj['alt'].min()
            
            # Add weather-related stats if available
            weather_stats = {}
            for col in ['WS50M', 'WD50M', 'T2M', 'PS', 'RH2M']:
                if col in traj.columns and not traj[col].isna().all():
                    weather_stats.update({
                        f'avg_{col.lower()}': traj[col].mean(),
                        f'{col.lower()}_std': traj[col].std()
                    })
            
            stats[balloon_id] = {
                'total_distance_km': total_distance,
                'avg_speed_kmh': avg_speed,
                'avg_altitude_km': avg_alt,
                'altitude_range_km': alt_change,
                'last_lat': traj['lat'].iloc[-1],
                'last_lon': traj['lon'].iloc[-1],
                'last_alt': traj['alt'].iloc[-1],
                **weather_stats
            }
            print(f"Completed statistics for balloon {balloon_id}")
        
        print(f"\nFinished calculating statistics for {len(stats)} balloons")
        return stats

    def _calculate_distance_traveled(self, traj: pd.DataFrame) -> float:
        """Calculate total distance traveled in kilometers."""
        from geopy.distance import geodesic
        
        total_distance = 0
        for i in range(len(traj) - 1):
            point1 = (traj['lat'].iloc[i], traj['lon'].iloc[i])
            point2 = (traj['lat'].iloc[i + 1], traj['lon'].iloc[i + 1])
            total_distance += geodesic(point1, point2).kilometers
            
        return total_distance 