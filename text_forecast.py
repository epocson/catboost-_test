import pandas as pd
import numpy as np

try:
    # Load the data
    print("Loading data...")
    data = pd.read_csv('data/population_2015_2024.csv')
    print("Data loaded successfully!")
    print(f"Years in data: {data['year'].min()} to {data['year'].max()}")
    
    # Display historical data
    print("\nHistorical Population Data:")
    for _, row in data.iterrows():
        print(f"{int(row['year'])}: {int(row['population']):,} (Birth Rate: {row['birth_rate']}, Death Rate: {row['death_rate']}, Migration: {row['migration_rate']})")
    
    # Simple linear regression for forecasting
    print("\nCalculating population trend...")
    years = data['year'].values
    population = data['population'].values
    trend = np.polyfit(years, population, 1)
    print(f"Population trend: {trend[0]:,.0f} people per year")
    
    # Generate future years (2025-2030)
    future_years = range(data['year'].max() + 1, data['year'].max() + 7)
    
    # Create a table header
    print("\nPopulation Forecast (2025-2030):")
    print("=" * 60)
    print(f"{'Year':<10}{'Population':<20}{'Change (%)':<15}{'Population (M)':<15}")
    print("-" * 60)
    
    # Calculate and display forecasts
    last_pop = data['population'].iloc[-1]
    for year in future_years:
        # Predict population using the trend
        pred_pop = np.polyval(trend, year)
        
        # Calculate year over year change
        yoy_change = (pred_pop - last_pop) / last_pop * 100
        
        # Print formatted row
        print(f"{year:<10}{int(pred_pop):,d}{'':>10}{yoy_change:.2f}%{'':>10}{pred_pop/1_000_000:.2f}M")
        
        # Update for next iteration
        last_pop = pred_pop
    
    print("=" * 60)
    print("\nForecast complete!")
    
except Exception as e:
    print(f"An error occurred: {str(e)}")   