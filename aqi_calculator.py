import numpy as np
from config import NO2_CONVERSION_FACTOR, SO2_CONVERSION_FACTOR, CO_CONVERSION_FACTOR, O3_CONVERSION_FACTOR
from aqi_breakpoints import (PM25_BREAKPOINTS, PM10_BREAKPOINTS, NO2_BREAKPOINTS, 
                           CO_BREAKPOINTS, SO2_BREAKPOINTS, O3_BREAKPOINTS, O3_1H_BREAKPOINTS)

def calc_aqi(Cp, BPl, BPh, Il, Ih):
    """
    Calculate AQI using EPA's linear interpolation formula.
    
    Parameters:
    Cp: Pollutant concentration
    BPl, BPh: Concentration breakpoints
    Il, Ih: Index breakpoints
    
    Returns:
    AQI value
    """
    a = (Ih - Il)
    b = (BPh - BPl)
    c = (Cp - BPl)
    return round((a / b) * c + Il)

def find_breakpoint(value, breakpoints):
    """Find the appropriate breakpoint interval for a given concentration value."""
    for (BPl, BPh, Il, Ih) in breakpoints:
        if BPl <= value <= BPh:
            return (BPl, BPh, Il, Ih)
    
    # If value exceeds the highest breakpoint, use the highest breakpoint
    if value > breakpoints[-1][1]:
        return breakpoints[-1]
    
    return None


def pollutant_aqi(pollutant, value, is_1h_ozone=False):
    """Calculate AQI for a specific pollutant given its concentration value."""
    # Handle missing, null or negative values
    if value is None or np.isnan(value) or value < 0:
        return None
        
    if pollutant == 'pm2_5':
        bp = find_breakpoint(value, PM25_BREAKPOINTS)
        if bp:
            return calc_aqi(value, *bp)
    elif pollutant == 'pm10':
        bp = find_breakpoint(value, PM10_BREAKPOINTS)
        if bp:
            return calc_aqi(value, *bp)
    elif pollutant == 'nitrogen_dioxide':
        # Convert μg/m³ to ppb
        ppb = value / NO2_CONVERSION_FACTOR
        bp = find_breakpoint(ppb, NO2_BREAKPOINTS)
        if bp:
            return calc_aqi(ppb, *bp)
    elif pollutant == 'carbon_monoxide':
        # Convert μg/m³ to ppm
        ppm = value / CO_CONVERSION_FACTOR
        bp = find_breakpoint(ppm, CO_BREAKPOINTS)
        if bp:
            return calc_aqi(ppm, *bp)
    elif pollutant == 'sulphur_dioxide':
        # Convert μg/m³ to ppb
        ppb = value / SO2_CONVERSION_FACTOR
        bp = find_breakpoint(ppb, SO2_BREAKPOINTS)
        if bp:
            return calc_aqi(ppb, *bp)
    elif pollutant == 'ozone':
        # Convert μg/m³ to ppm
        ppm = value / O3_CONVERSION_FACTOR
        if is_1h_ozone:
            bp = find_breakpoint(ppm, O3_1H_BREAKPOINTS)
        else:
            bp = find_breakpoint(ppm, O3_BREAKPOINTS)
        if bp:
            return calc_aqi(ppm, *bp)
    return None

def get_aqi_category(aqi):
    """Convert numeric AQI value to EPA category."""
    if aqi is None:
        return None
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"
