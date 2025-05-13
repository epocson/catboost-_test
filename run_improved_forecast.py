from pathlib import Path
from src.improved_forecast import EnhancedPopulationForecaster, plot_forecast_with_intervals, plot_demographic_indicators, create_forecast_table, create_historical_and_forecast_table

if __name__ == "__main__":
    # Set up paths
    base_path = Path(".")
    data_path = base_path / "data" / "population_2015_2024.csv"
    model_path = base_path / "model" / "catboost_advanced_model.cbm"
    forecast_path = base_path / "forecasts" / "population_forecast_2025_2030.csv"
    
    # Create forecasts directory if it doesn't exist
    forecast_dir = base_path / "forecasts"
    forecast_dir.mkdir(exist_ok=True)
    
    # Initialize and train forecaster
    print("Initializing Enhanced Population Forecaster...")
    forecaster = EnhancedPopulationForecaster(data_path)
    historical_data = forecaster.load_data()
    
    # Train model
    print("\nTraining advanced CatBoost model...")
    metrics, feature_importances = forecaster.train_model(
        params={
            'iterations': 2000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE'
        }
    )
    
    # Save the trained model
    forecaster.save_model(model_path)
    
    # Generate future predictions
    print("\nGenerating population forecasts for 2025-2030...")
    future_predictions = forecaster.forecast_future(years=6)
    
    # Generate report
    report = forecaster.generate_report(future_predictions, save_path=forecast_path)
    
    # Print forecast summary
    print("\nPopulation Forecast Summary:")
    print(f"Current Population ({historical_data['year'].max()}): {report['current_population']:,.0f}")
    print(f"Forecasted Population (2030): {report['forecast_population']:,.0f}")
    print(f"Total Change: {report['total_change']:,.0f} ({report['percent_change']:.2f}%)")
    print(f"Average Annual Growth Rate: {report['avg_annual_growth']:.2f}%")
    
    # Display forecast table (original)
    table_original_forecast = create_forecast_table(future_predictions)
    print("\nDetailed Population Forecasts (2025-2030):")
    print(table_original_forecast.to_string(index=False))

    # Display combined historical and forecast table
    combined_table = create_historical_and_forecast_table(historical_data, future_predictions)
    print("\nPopulation Dynamics (Historical & Forecast):")
    print(combined_table.to_string(index=False))
    
    # Plot forecasts
    print("\nGenerating visualization plots...")
    plot_forecast_with_intervals(historical_data, future_predictions, 
                               save_path=base_path / "forecasts" / "population_forecast_plot.png")
    
    plot_demographic_indicators(historical_data, future_predictions,
                              save_path=base_path / "forecasts" / "demographic_indicators_plot.png")
    
    print("\nForecasting completed successfully!") 