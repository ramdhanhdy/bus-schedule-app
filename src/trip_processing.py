# src/trip_processing.py

import pandas as pd
import numpy as np
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN
from src.calculate_distance import calculate_distance, HaversineCalculator
from src.data_processing import DataProcessingStrategy

class ClusterStopsStrategy(DataProcessingStrategy):
    def __init__(self, eps=0.1, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def process(self, df):
        coords = df[['latitude', 'longitude']].values
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='haversine')
        df['cluster'] = dbscan.fit_predict(np.radians(coords))
        stop_locations = df.groupby('cluster').agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        stop_locations['stop_id'] = range(len(stop_locations))
        df = df.merge(stop_locations[['cluster', 'stop_id']], on='cluster', how='left')
        return df

class CreateTripSegments(DataProcessingStrategy):
    def __init__(self):
        self.calculator = HaversineCalculator()

    def process(self, df):
        trip_segments = []
        current_trip = []
        
        for i, row in df.iterrows():
            if not current_trip or (row['time'] - current_trip[-1]['time']).total_seconds() <= 1800:
                current_trip.append(row)
            else:
                if len(current_trip) > 1:
                    trip_segments.append(current_trip)
                current_trip = [row]
        
        if len(current_trip) > 1:
            trip_segments.append(current_trip)
        
        processed_segments = []
        for trip in trip_segments:
            for i in range(len(trip) - 1):
                start = trip[i]
                end = trip[i + 1]
                duration = (end['time'] - start['time']).total_seconds() / 60
                distance = calculate_distance(start['latitude'], start['longitude'], end['latitude'], end['longitude'], self.calculator)
                segment = LineString([(start['longitude'], start['latitude']), (end['longitude'], end['latitude'])])
                
                segment_data = {
                    'route_id': start['route_id'],
                    'start_time': start['time'],
                    'end_time': end['time'],
                    'start_hour': start['hour'],
                    'end_hour': end['hour'],
                    'start_lat': start['latitude'],
                    'start_lon': start['longitude'],
                    'end_lat': end['latitude'],
                    'end_lon': end['longitude'],
                    'start_stop_id': start['stop_id'],
                    'end_stop_id': end['stop_id'],
                    'duration': duration,
                    'distance': distance,
                    'segment': segment
                }
                
                processed_segments.append(segment_data)
        
        return pd.DataFrame(processed_segments)

class FilterTripSegments(DataProcessingStrategy):
    def process(self, df):
        return df[df['start_stop_id'] != df['end_stop_id']]

class RemoveLongDurationTrips(DataProcessingStrategy):
    def process(self, df):
        return df[df['duration'] <= 60]  # Remove trips longer than 60 minutes

class TripProcessor:
    def __init__(self, strategies):
        self.strategies = strategies

    def process(self, df):
        for strategy in self.strategies:
            df = strategy.process(df)
        return df