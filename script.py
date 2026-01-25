import pandas as pd

# Read csv file
#df = pd.read_csv('new-weather-data.csv')
df = pd.read_csv('Shoulder_Season_Data.csv')

# Dropping Columns for Dataset Usage
"""
df = df.drop(columns=['vapor_pressure_deficit_hpa_2_m_above_gnd_max', 'daylight_duration_min_surface_sum', 
                      'reference_evapotranspiration_mm_2_m_above_gnd_sum', 'shortwave_radiation_w_per_m2_surface_sum', 
                      'wind_speed_km_per_h_2_m_above_gnd_avg', 'wind_speed_km_per_h_2_m_above_gnd_max',
                      'temperature_c_2_m_above_gnd_sum', 'precipitation_total_mm_surface_sum',
                      'wind_gust_km_per_h_surface_max', 'temperature_c_surface_avg', 'soil_transpirable_water_fraction_0to100cm_below_gnd_avg'])
"""

# df = df.drop(columns=['loc', 'start_date', 'end_date', 'input_lat', 'input_lon', 'centroid_lat', 'centroid_lon', 'distance_km'])
#df = df.drop(columns=['place_id'])

# df.to_csv('new-weather-data.csv', index=False)

df = df.drop(columns=['maxLW', 'maxST', 'maxSM', 'maxRF', 'minST', 'minSM'])