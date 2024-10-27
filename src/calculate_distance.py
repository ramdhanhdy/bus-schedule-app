import math
from abc import ABC, abstractmethod

class DistanceCalculator(ABC):
    @abstractmethod
    def calculate(self, lat1, lon1, lat2, lon2):
        pass

class HaversineCalculator(DistanceCalculator):
    def calculate(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two points on the Earth's surface
        using the Haversine formula.
        
        :param lat1: Latitude of the first point (in degrees)
        :param lon1: Longitude of the first point (in degrees)
        :param lat2: Latitude of the second point (in degrees)
        :param lon2: Longitude of the second point (in degrees)
        :return: Distance between the two points in kilometers
        """
        # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in kilometers
        radius = 6371
        
        # Calculate the distance
        distance = radius * c
        
        return distance

def calculate_distance(lat1, lon1, lat2, lon2, calculator: DistanceCalculator):
    """
    Calculate the distance between two points on the Earth's surface
    given their latitude and longitude coordinates.
    
    :param lat1: Latitude of the first point (in degrees)
    :param lon1: Longitude of the first point (in degrees)
    :param lat2: Latitude of the second point (in degrees)
    :param lon2: Longitude of the second point (in degrees)
    :param calculator: DistanceCalculator strategy to use
    :return: Distance between the two points in kilometers
    """
    return calculator.calculate(lat1, lon1, lat2, lon2)

# Example usage
if __name__ == "__main__":
    # Example coordinates (New York City and Los Angeles)
    nyc_lat, nyc_lon = 40.7128, -74.0060
    la_lat, la_lon = 34.0522, -118.2437
    
    haversine_calculator = HaversineCalculator()
    distance = calculate_distance(nyc_lat, nyc_lon, la_lat, la_lon, haversine_calculator)
    print(f"The distance between the two points is {distance:.2f} km")
