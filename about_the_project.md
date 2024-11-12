# Bus Schedule Prediction Project

## Project Overview
This project aims to predict school bus trip durations in Malang, Indonesia, using historical GPS data and weather information. The system uses machine learning (specifically CatBoost) to predict arrival times at designated bus stops, achieving a mean absolute error of around 1 minute.

## Bus Operation Details
- **Route Structure**: 7 fixed bus stops (BT01-BT07)
- **Operating Schedule**:
  - Morning Route: BT01 → BT07 (7:00 AM - 8:00 AM)
  - Afternoon Route: BT07 → BT01 (3:00 PM - 5:00 PM)
- **Operating Days**: School days only (excluding weekends and public holidays)

## Data Sources
1. **GPS Location Data**
   - Collected from February 2024 to September 2024, each month has a separate file
   - **Data Availability Note**: August 2024 data currently unusable due to significant coordinate system discrepancy
   - Effective temporal coverage: 7 months (Feb-Jul, Sep 2024)
   - Records bus arrivals at each stop
   - Raw data format example:
     ```
     route_id  imei          latitude   longitude  speed  time            flag
     11        fb:fd:2a:a8   -7.93338  112.6035   10     9/1/2024 19:12  1
     10        fb:fd:2a:a8   -7.93324  112.6035   10     9/2/2024 7:17   1
     10        fb:fd:2a:a8   -7.94329  112.6105   10     9/2/2024 7:19   1
     11        f2:ab:73:1    -7.93338  112.6035   10     9/2/2024 17:56  1
     11        f2:ab:73:1    -7.94331  112.6104   10     9/2/2024 18:06  0
     ```
   - Fields:
     - `route_id`: Bus route identifier
     - `imei`: Device identifier
     - `latitude/longitude`: GPS coordinates
     - `speed`: Vehicle speed in km/h
     - `time`: Timestamp of GPS reading
     - `flag`: Binary indicator (possibly for valid/invalid readings)
   - Data Quality Issues:
     - Contains readings outside operating hours
     - Duplicate records present
     - Weekend/holiday data included
     - GPS coordinate noise
     - Coordinate system inconsistency in August data
     - Possible GPS device reconfiguration in August

2. **Weather Data**
   - Features include:
     - Basic measurements: temperature, precipitation, humidity, visibility, pressure
     - Wind conditions: wind speed in km/h
     - Cloud cover percentage
     - Advanced metrics:
       - Feels like temperature
       - Temperature range (feels like - actual)
       - Dew point (calculated using Magnus-Tetens formula)
       - Heat index (calculated using Rothfusz regression)
   - Collected at regular 3-hour intervals
   - Comfort indicators derived:
     - Humidity comfort levels: Comfortable (30-50%), Humid (50-70%), Very Humid (>70%)
     - Heat stress levels: Safe (<27°C), Caution (27-32°C), Extreme Caution (32-41°C)

## Data Processing Pipeline
1. **Data Cleaning** (`01b_data_preparation.ipynb`)
   - Removes data outside operating hours (7-8 AM, 3-5 PM)
   - Filters out weekends and public holidays
   - Removes duplicate records
   - Drops unnecessary columns (flag, speed, imei, bus_id)
   - Converts latitude/longitude to numeric format
   - Processes timestamps
   - Extracts time features (hour, weekday, etc.)
   

2. **Stop Location Processing**
   - Uses DBSCAN clustering (eps=0.1, min_samples=5) to standardize bus stop coordinates
   - Assigns consistent stop IDs (BT01-BT07)
   - Validates correct stop sequence:
     - Morning: BT01 → BT07
     - Afternoon: BT07 → BT01
   - Trip segmentation:
     - Maximum 30-minute gap between consecutive readings
     - Removes segments longer than 60 minutes
     - Filters out segments where start and end stops are identical

3. **Feature Engineering** (`02_feature_engineering.ipynb`)
   - Combines trip data with weather information
   - Creates derived features:
     - Time-based cyclic features:
       - Minutes of day (sin/cos transformed)
       - Month of year (sin/cos transformed)
       - Day of week (cos transformed)
     - Distance-based: using Haversine formula
     - Weather features:
       - Temperature and feels like temperature
       - Cloud cover percentage
       - Wind speed
       - Temperature range
   - Selected features based on importance:
     - Primary features: distance, time-based features
     - Weather features: cloud cover, temperature, wind speed
     - All features normalized/scaled appropriately

## Data Quality Metrics
- Raw data completeness: ~95% valid GPS readings (flag=1)
- Temporal coverage: 7 usable months (Feb-Jul, Sep 2024)
  - August data excluded due to coordinate system mismatch

## Model Performance
- **Mean Absolute Error**: 1.04 minutes
- **Median Absolute Error**: 0.66 minutes
- **90th percentile error**: 2.15 minutes
- **95th percentile error**: 2.77 minutes

## Error Analysis
- **Error Distribution**:
  - Roughly bell-shaped and centered near zero
  - Most errors fall between -2 and +2 minutes
  - Slight tendency to overestimate durations
  - Very few errors beyond ±4 minutes

- **Error by Trip Duration**:
  - Short trips (0-5 minutes): Errors mostly under 2 minutes
  - Medium trips (5-10 minutes): Errors between 2-4 minutes
  - Long trips (>15 minutes): Errors up to 7 minutes
  - Heteroscedasticity present (error increases with trip duration)

## Key Characteristics
- Predictions are more accurate for shorter trips
- Error increases with trip duration
- Model shows slight tendency to overestimate durations
- Most predictions are within ±2 minutes of actual duration

## Technical Stack
- Python
- Primary libraries: pandas, numpy, scikit-learn, CatBoost
- Data storage: CSV files
- Model serialization: joblib

## Current Challenges/Areas for Improvement
1. Trip segmentation needs refinement for better stop detection
2. Some heteroscedasticity in predictions (larger errors for longer trips)
3. Handling of extreme weather conditions or unusual traffic patterns
4. Better handling of public holidays and school calendar events

## Project Goals
1. Accurate prediction of bus arrival times at each stop
2. Robust handling of various weather conditions
3. Real-time prediction capabilities
4. Integration with school transportation system
5. User-friendly interface for students and parents

