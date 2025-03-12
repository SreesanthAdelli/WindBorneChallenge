import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.spatial.distance import pdist, squareform
from datetime import datetime, timedelta

class ConstellationAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        print(f"\nInitializing ConstellationAnalyzer with DataFrame of shape {data.shape}")
        
    def analyze_constellation_coverage(self) -> Dict[str, float]:
        """
        Analyze the coverage of the balloon constellation.
        Returns metrics about the constellation's geographic distribution.
        """
        print("\nAnalyzing constellation coverage...")
        latest_positions = self.data.sort_values('timestamp').groupby('balloon_id').last()
        print(f"Found {len(latest_positions)} balloons in latest positions")
        
        # Calculate the convex hull area of the constellation
        if len(latest_positions) >= 3:
            from scipy.spatial import ConvexHull
            points = latest_positions[['lat', 'lon']].values
            try:
                hull = ConvexHull(points)
                coverage_area = hull.area  # Approximate area in square degrees
                print(f"Calculated convex hull area: {coverage_area:.2f} sq deg")
            except Exception as e:
                print(f"Error calculating convex hull: {e}")
                coverage_area = 0
        else:
            print("Insufficient points for convex hull calculation")
            coverage_area = 0
            
        # Calculate average distance between balloons
        if len(latest_positions) >= 2:
            distances = pdist(latest_positions[['lat', 'lon']].values)
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            print(f"Average separation: {avg_distance:.2f}°, Max separation: {max_distance:.2f}°")
        else:
            print("Insufficient points for distance calculations")
            avg_distance = 0
            max_distance = 0
            
        return {
            'coverage_area_sq_deg': coverage_area,
            'avg_balloon_separation_deg': avg_distance,
            'max_balloon_separation_deg': max_distance,
            'active_balloons': len(latest_positions)
        }
        
    def identify_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns in balloon movement and behavior.
        """
        print("\nIdentifying constellation patterns...")
        patterns = {}
        
        # Analyze altitude patterns
        print("Analyzing altitude patterns...")
        altitude_stats = self.data.groupby('balloon_id')['alt'].agg(['mean', 'std', 'min', 'max'])
        patterns['altitude'] = {
            'avg_fleet_altitude': altitude_stats['mean'].mean(),
            'altitude_variation': altitude_stats['std'].mean(),
            'altitude_range': altitude_stats['max'].max() - altitude_stats['min'].min()
        }
        print(f"Fleet average altitude: {patterns['altitude']['avg_fleet_altitude']:.2f} km")
        
        # Analyze temporal patterns
        print("\nAnalyzing movement patterns...")
        time_patterns = {}
        for balloon_id in self.data['balloon_id'].unique():
            print(f"Processing balloon {balloon_id}...")
            balloon_data = self.data[self.data['balloon_id'] == balloon_id].sort_values('timestamp')
            if len(balloon_data) >= 2:
                # Calculate velocity
                time_diff = balloon_data['timestamp'].diff().dt.total_seconds() / 3600  # hours
                lat_diff = balloon_data['lat'].diff()
                lon_diff = balloon_data['lon'].diff()
                
                # Simple velocity calculation (degrees per hour)
                velocity = np.sqrt(lat_diff**2 + lon_diff**2) / time_diff
                
                # Add weather-based analysis if available
                weather_metrics = {}
                if 'WS50M' in balloon_data.columns and not balloon_data['WS50M'].isna().all():
                    # Calculate wind influence
                    wind_speed = balloon_data['WS50M']
                    wind_direction = balloon_data['WD50M']
                    weather_metrics.update({
                        'avg_wind_speed': wind_speed.mean(),
                        'wind_speed_std': wind_speed.std(),
                        'wind_direction_mode': wind_direction.mode().iloc[0] if not wind_direction.isna().all() else None
                    })
                    print(f"Added weather metrics for balloon {balloon_id}")
                
                time_patterns[balloon_id] = {
                    'avg_velocity': velocity.mean(),
                    'max_velocity': velocity.max(),
                    'velocity_std': velocity.std(),
                    **weather_metrics
                }
                print(f"Average velocity for balloon {balloon_id}: {velocity.mean():.2f} deg/hour")
        
        patterns['movement'] = time_patterns
        
        # Add weather pattern analysis
        if 'T2M' in self.data.columns:
            print("\nAnalyzing weather patterns...")
            weather_patterns = self._analyze_weather_patterns()
            patterns['weather'] = weather_patterns
            print("Weather pattern analysis complete")
        else:
            print("No weather data available for pattern analysis")
        
        return patterns
    
    def _analyze_weather_patterns(self) -> Dict[str, Any]:
        """Analyze weather patterns and their impact on balloon behavior."""
        print("\nAnalyzing detailed weather patterns...")
        weather_patterns = {}
        
        # Group data by balloon and calculate correlations
        for balloon_id in self.data['balloon_id'].unique():
            print(f"Analyzing weather correlations for balloon {balloon_id}...")
            balloon_data = self.data[self.data['balloon_id'] == balloon_id].sort_values('timestamp')
            
            if len(balloon_data) < 2:
                print(f"Insufficient data points for balloon {balloon_id}")
                continue
                
            # Calculate altitude changes
            balloon_data['alt_change'] = balloon_data['alt'].diff()
            
            # Analyze correlations with weather parameters
            correlations = {}
            for param in ['WS50M', 'T2M', 'PS', 'RH2M']:
                if param in balloon_data.columns and not balloon_data[param].isna().all():
                    corr = balloon_data['alt_change'].corr(balloon_data[param])
                    correlations[param] = corr
                    print(f"Correlation between altitude change and {param}: {corr:.3f}")
            
            weather_patterns[balloon_id] = correlations
        
        return weather_patterns
        
    def get_constellation_health(self) -> Dict[str, str]:
        """
        Assess the overall health of the constellation based on various metrics.
        """
        print("\nAssessing constellation health...")
        latest_data = self.data[self.data['timestamp'] == self.data['timestamp'].max()]
        print(f"Latest data timestamp: {self.data['timestamp'].max()}")
        
        # Check data freshness
        time_since_update = datetime.utcnow() - self.data['timestamp'].max()
        data_freshness = 'Good' if time_since_update < timedelta(hours=1) else 'Stale'
        print(f"Data freshness: {data_freshness} (last update: {time_since_update})")
        
        # Check constellation spread
        coverage = self.analyze_constellation_coverage()
        coverage_status = 'Good' if coverage['coverage_area_sq_deg'] > 1000 else 'Limited'
        print(f"Coverage status: {coverage_status}")
        
        # Check number of active balloons
        balloon_count_status = 'Good' if coverage['active_balloons'] >= 3 else 'Degraded'
        print(f"Constellation size status: {balloon_count_status}")
        
        # Check weather conditions if available
        weather_status = 'Unknown'
        if 'WS50M' in latest_data.columns and not latest_data['WS50M'].isna().all():
            avg_wind_speed = latest_data['WS50M'].mean()
            weather_status = 'Warning' if avg_wind_speed > 15 else 'Good'
            print(f"Weather status: {weather_status} (avg wind speed: {avg_wind_speed:.1f} m/s)")
        else:
            print("Weather data not available for health assessment")
        
        return {
            'data_freshness': data_freshness,
            'coverage_status': coverage_status,
            'constellation_size': balloon_count_status,
            'weather_conditions': weather_status,
            'active_balloons': coverage['active_balloons']
        }
        
    def get_operational_recommendations(self) -> List[str]:
        """
        Generate operational recommendations based on constellation analysis.
        """
        print("\nGenerating operational recommendations...")
        health = self.get_constellation_health()
        patterns = self.identify_patterns()
        recommendations = []
        
        if health['data_freshness'] == 'Stale':
            recommendations.append("Data feed may be delayed - check constellation communication systems")
            
        if health['coverage_status'] == 'Limited':
            recommendations.append("Consider adjusting balloon positions to improve geographic coverage")
            
        if health['constellation_size'] == 'Degraded':
            recommendations.append("Constellation operating with reduced capacity - consider launching additional balloons")
            
        # Analyze altitude distribution
        alt_stats = patterns['altitude']
        if alt_stats['altitude_variation'] > 5:  # km
            recommendations.append("High altitude variation detected - check for atmospheric disturbances")
        
        # Add weather-based recommendations
        if 'weather' in patterns:
            for balloon_id, correlations in patterns['weather'].items():
                if 'WS50M' in correlations and abs(correlations['WS50M']) > 0.7:
                    recommendations.append(f"Balloon {balloon_id} showing strong response to wind conditions - monitor closely")
                if 'PS' in correlations and abs(correlations['PS']) > 0.7:
                    recommendations.append(f"Balloon {balloon_id} showing sensitivity to pressure changes")
        
        print(f"Generated {len(recommendations)} recommendations")
        return recommendations 