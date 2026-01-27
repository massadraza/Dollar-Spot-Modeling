import pandas as pd 

# Read the CSV files
conversion_df = pd.read_csv('NewConversion.csv')
weather_df = pd.read_csv('New-Weather-Data-Final.csv')

# Get weather feature columns (all columns except 'date')
weather_features = ['maxAT', 'meanAT', 'minAT', 'meanRH', 'maxRH', 'minRH', 
                    'meanDP', 'meanLW', 'meanSM', 'meanST', 'meanRF']

# Add empty columns for weather features to conversion_df
for feature in weather_features:
    conversion_df[feature] = None

# Create a dictionary from weather data for fast lookup (date -> row data)
weather_dict = weather_df.set_index('date').to_dict('index')

# Go through each row in conversion_df
for idx, row in conversion_df.iterrows():
    date = row['Date']
    
    # Check if this date exists in weather data
    if date in weather_dict:
        # Copy weather features to this row
        for feature in weather_features:
            conversion_df.at[idx, feature] = weather_dict[date][feature]
        print(f"Found match for date: {date}")

conversion_df.to_csv('FinalNewFociDataset.csv', index=False)
print("\nUpdated FinalNewFociDataset.csv with weather data")
