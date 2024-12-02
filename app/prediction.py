import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

class TripPredictor:
    def __init__(self, model_path='app/models/final_model_no_weather_trigmonth2.joblib'):
        model_info = joblib.load(model_path)
        # Extract the actual model from the model_info dictionary
        self.model = model_info['model']
        
    def _calculate_operation_minutes(self, time):
        """Calculate minutes of operation based on morning/afternoon sessions"""
        # Create timestamps for morning and afternoon start times
        morning_start = pd.Timestamp(time.date()).replace(hour=6, minute=30)
        afternoon_start = pd.Timestamp(time.date()).replace(hour=15, minute=0)
        
        if time < afternoon_start:  # Morning session
            return max(0, (time - morning_start).total_seconds() // 60 + 1)
        else:  # Afternoon session
            return (time - afternoon_start).total_seconds() // 60 + 181
        
    def prepare_features(self, distance, departure_time, origin_id):
        # Convert departure_time to pandas Timestamp if it's not already
        if not isinstance(departure_time, pd.Timestamp):
            departure_time = pd.Timestamp(departure_time)
            
        # Calculate minutes of operation
        minutes_of_operation = self._calculate_operation_minutes(departure_time)
        
        # Time-based features using minutes of operation (360 minutes total per session)
        minutes_sin = np.sin(2 * np.pi * minutes_of_operation / 360)
        minutes_cos = np.cos(2 * np.pi * minutes_of_operation / 360)
        
        # Day of week features
        day_of_week = departure_time.dayofweek
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Month features
        month_num = departure_time.month
        month_sin = np.sin(2 * np.pi * month_num / 12)
        month_cos = np.cos(2 * np.pi * month_num / 12)
        
        features = {
            'distance': distance,
            'minutes_sin': minutes_sin,
            'minutes_cos': minutes_cos,
            'origin_id': origin_id,
            'month_cos': month_cos,
            'month_sin': month_sin,
            'day_of_week_cos': day_of_week_cos,
            'day_of_week_sin': day_of_week_sin
        }
        
        # Ensure features are in the same order as during training
        feature_order = [
            'distance', 'minutes_sin', 'minutes_cos', 'origin_id',
            'month_cos', 'month_sin', 'day_of_week_sin', 'day_of_week_cos'
        ]
        
        return pd.DataFrame([{key: features[key] for key in feature_order}])
    
    def predict_duration(self, distance, departure_time, origin_id):
        features = self.prepare_features(distance, departure_time, origin_id)
        # Get log-transformed prediction
        log_duration = self.model.predict(features)[0]
        # Apply inverse transformation (exp) to get actual duration
        duration = np.exp(log_duration)
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
        cumulative_distance = 0
        
        # Calculate minutes since start of the day for display
        def get_minutes_display(base_time, current):
            base_minutes = base_time.hour * 60 + base_time.minute
            current_minutes = current.hour * 60 + current.minute
            
            # If current time is in the next day (crossed midnight)
            if current_minutes < base_minutes:
                current_minutes += 24 * 60  # Add 24 hours worth of minutes
                
            minutes_diff = current_minutes - base_minutes
            return f"(in {int(minutes_diff)} min)"
        
        base_time = current_time  # Keep track of initial time for display
        
        for i, stop in enumerate(stops):
            segment_distance = stop['distance']
            cumulative_distance += segment_distance
            # Use the full stop_id as origin_id
            origin_id = stop['stop_id']
            duration, error_margin = self.predict_duration(segment_distance, current_time, origin_id)
            current_time += timedelta(minutes=duration)
            
            # Format the display time
            minutes_display = get_minutes_display(base_time, current_time)
            
            schedule.append({
                'stop_id': stop['stop_id'],
                'name': stop['name'],
                'estimated_time': current_time.strftime('%H:%M'),
                'minutes_display': minutes_display,
                'error_margin': f"±{error_margin} min"
            })
            
        return schedule