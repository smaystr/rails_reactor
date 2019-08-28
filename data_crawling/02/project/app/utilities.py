from datetime import date
from decimal import Decimal


def alchemyencoder(obj):
    """JSON encoder function for SQLAlchemy special classes."""
    if isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)