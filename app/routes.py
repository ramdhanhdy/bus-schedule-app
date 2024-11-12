from flask import Blueprint, render_template, jsonify, request
from datetime import datetime
from app.prediction import TripPredictor

main = Blueprint('main', __name__)
predictor = TripPredictor()

# Route configuration
ROUTE_STOPS = [
    {'stop_id': 'BT01', 'name': 'SPBU Tlogomas', 'distance': 0},
    {'stop_id': 'BT02', 'name': 'SD Dinoyo 2', 'distance': 1.35},  # Distance from BT01
    {'stop_id': 'BT03', 'name': 'SMA 9', 'distance': 0.91},  # Distance from BT02
    {'stop_id': 'BT04', 'name': 'SMA 8', 'distance': 0.81},  # Distance from BT03
    {'stop_id': 'BT05', 'name': 'MAN 2', 'distance': 1.59},  # Distance from BT04
    {'stop_id': 'BT06', 'name': 'SMA Dempo', 'distance': 1.09},  # Distance from BT05
    {'stop_id': 'BT07', 'name': 'SMP 4', 'distance': 1.37},  # Distance from BT06
]

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/api/current-location')
def get_current_location():
    """Get the current bus location and next stops"""
    try:
        current_time = datetime.now()
        is_morning = 6 <= current_time.hour <= 9
        is_afternoon = 14 <= current_time.hour <= 17
        
        if not (is_morning or is_afternoon):
            return jsonify({'error': 'Bus not in service at this time'}), 400
            
        # For demo, assume we're at a specific stop based on time
        # In production, this would come from real-time GPS
        current_stop = get_current_stop(current_time)
        
        if current_stop is None:
            return jsonify({'error': 'Cannot determine current location'}), 400
            
        next_stops = get_next_stops(current_stop, is_morning)
        schedule = predictor.generate_schedule(next_stops, current_time)
        
        return jsonify({
            'current_time': current_time.strftime('%H:%M:%S'),
            'current_stop': current_stop,
            'next_stops': schedule
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_current_stop(current_time):
    """Simulate current stop based on time - replace with actual GPS logic"""
    hour = current_time.hour
    minute = current_time.minute
    
    if 7 <= hour < 8:  # Morning route
        progress = (minute + (hour - 7) * 60) / 60  # 0 to 1 progress through route
        stop_index = int(progress * 6)  # 0 to 6
        return ROUTE_STOPS[min(stop_index, 6)]
    elif 15 <= hour < 17:  # Afternoon route
        progress = (minute + (hour - 15) * 60) / 120  # 0 to 1 progress through route
        stop_index = 6 - int(progress * 6)  # 6 to 0
        return ROUTE_STOPS[max(stop_index, 0)]
    return None

def get_next_stops(current_stop, is_morning):
    """Get remaining stops in the route"""
    current_index = next(i for i, stop in enumerate(ROUTE_STOPS) 
                        if stop['stop_id'] == current_stop['stop_id'])
    
    if is_morning:
        return ROUTE_STOPS[current_index + 1:]
    else:
        return list(reversed(ROUTE_STOPS[:current_index]))

@main.route('/demo')
def demo():
    return render_template('demo.html')

@main.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        current_stop = int(data['current_stop'])
        current_time = datetime.strptime(data['current_time'], '%Y-%m-%dT%H:%M')
        
        # Validate current_stop index
        if current_stop < 0 or current_stop >= len(ROUTE_STOPS):
            return jsonify({'error': 'Invalid stop index'}), 400
            
        # Get next stops based on current stop and time
        is_morning = 6 <= current_time.hour <= 9
        next_stops = get_next_stops(ROUTE_STOPS[current_stop], is_morning)
        
        # Generate schedule with stop names
        schedule = predictor.generate_schedule(next_stops, current_time)
        
        response = {
            'current_time': current_time.strftime('%H:%M:%S'),
            'current_stop': {
                'stop_id': ROUTE_STOPS[current_stop]['stop_id'],
                'name': ROUTE_STOPS[current_stop]['name'],
                'distance': ROUTE_STOPS[current_stop]['distance']
            },
            'next_stops': schedule
        }
        
        return jsonify(response)
    except ValueError as ve:
        return jsonify({'error': 'Invalid date/time format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500