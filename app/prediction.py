from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import requests
import os

class TripPredictor:
    def __init__(self, model_path='app/models/final_model_v2.joblib'):
        model_info = joblib.load(model_path)
        # Extract the actual model from the model_info dictionary
        self.model = model_info['model']
        self.api_key = os.getenv('WORLDWEATHER_API_KEY')
        self.weather_cache = {}  # Cache weather data to avoid excessive API calls
        self.cache_duration = timedelta(hours=1)  # Update weather every hour
        self.last_weather_update = None
        
    def _get_weather_data(self):
        current_time = datetime.now()
        
        # Return cached data if it's still valid
        if (self.last_weather_update and 
            current_time - self.last_weather_update < self.cache_duration and 
            self.weather_cache):
            return self.weather_cache
            
        # Malang coordinates
        lat, lon = -7.9666, 112.6326
        
        try:
            url = f"https://api.worldweatheronline.com/premium/v1/weather.ashx"
            params = {
                'key': self.api_key,
                'q': f"{lat},{lon}",
                'format': 'json',
                'num_of_days': 1,
                'tp': 1  # 1 hour intervals
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            current = data['data']['current_condition'][0]
            
            self.weather_cache = {
                'temperature': float(current['temp_C']),
                'cloud_cover': float(current['cloudcover']),
                'wind_speed': float(current['windspeedKmph']),
                'temp_range': float(current['FeelsLikeC']) - float(current['temp_C'])
            }
            
            self.last_weather_update = current_time
            return self.weather_cache
            
        except Exception as e:
            print(f"Weather API error: {e}")
            # Fallback to default values if API fails
            return {
                'temperature': 25,
                'cloud_cover': 50,
                'wind_speed': 5,
                'temp_range': 2
            }
    
    def prepare_features(self, distance, departure_time):
        # Time-based features
        minutes = departure_time.hour * 60 + departure_time.minute
        minutes_sin = np.sin(2 * np.pi * minutes / (24 * 60))
        minutes_cos = np.cos(2 * np.pi * minutes / (24 * 60))
        
        # Month-based features
        month_num = departure_time.month
        month_num_sin = np.sin(2 * np.pi * month_num / 12)
        month_num_cos = np.cos(2 * np.pi * month_num / 12)
        
        # Day of week features
        day_of_week = departure_time.weekday()
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Get real weather data
        weather = self._get_weather_data()
        
        features = {
            'distance': distance,
            'minutes_sin': minutes_sin,
            'minutes_cos': minutes_cos,
            'month_num_sin': month_num_sin,
            'cloud_cover': weather['cloud_cover'],
            'month_num_cos': month_num_cos,
            'temp_range': weather['temp_range'],
            'temperature': weather['temperature'],
            'wind_speed': weather['wind_speed'],
            'day_of_week_cos': day_of_week_cos
        }
        
        # Ensure features are in the same order as during training
        feature_order = [
            'distance', 'minutes_sin', 'minutes_cos', 'month_num_sin', 
            'cloud_cover', 'month_num_cos', 'temp_range', 'temperature',
            'wind_speed', 'day_of_week_cos'
        ]
        
        return pd.DataFrame([{key: features[key] for key in feature_order}])
    
    def predict_duration(self, distance, departure_time):
        features = self.prepare_features(distance, departure_time)
        duration = self.model.predict(features)[0]
        error_margin = self._calculate_error_margin(duration)
        return duration, error_margin
    
    def _calculate_error_margin(self, duration):
        if duration <= 5:
            return 2
        elif duration <= 10:
            return 3
        return 5

    def generate_schedule(self, stops, current_time):
        schedule = []
        current_time = current_time
        cumulative_distance = 0
        
        for i, stop in enumerate(stops):
            segment_distance = stop['distance']
            cumulative_distance += segment_distance
            duration, error_margin = self.predict_duration(segment_distance, current_time)
            current_time += timedelta(minutes=duration)
            
            schedule.append({
                'stop_id': stop['stop_id'],
                'name': stop['name'],
                'estimated_time': current_time.strftime('%H:%M'),
                'error_margin': f"±{error_margin} min"
            })
            
        return schedule