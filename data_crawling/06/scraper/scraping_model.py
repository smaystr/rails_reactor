from datetime import date


class Apartment:
    title: str = ''
    creation_date: date = None
    price_usd: int = 0
    price_uah: int = 0
    description: str = ''
    rooms_count: int = 0
    floor: int = None
    seller: str = None
    wall_type: str = None
    construction_year: int = None
    heating: str = None
    is_verified_price: bool = False
    is_verified_flat: bool = False
    realty_id: int = None
    street_name: str = None
    city_name: str = None
    district_name: str = None
    longitude: float = None
    latitude: float = None
    total_square_meters: float = None
    living_square_meters: float = None
    kitchen_square_meters: float = None
    photos = []
    water: str = None
    building_condition: str = None
    elevator_count: int = None
    dist_to_center: str = None
    dist_to_school: str = None
    dist_to_kindergarten: str = None
    dist_to_bus_terminal: str = None
    dist_to_railway_station: str = None
    dist_to_airport: str = None
    dist_to_hospital: str = None
    dist_to_shop: str = None
    dist_to_parking: str = None
    dist_to_rest_area: str = None
