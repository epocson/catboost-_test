import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class EnhancedPopulationForecaster:
    def __init__(self, data_path):
        """Initialize the forecaster with data path."""
        self.data_path = data_path
        self.model = None
        self.data = None
        self.feature_names = None
        self.metrics = {}
        
    def load_data(self):
        """Load and prepare the population data."""
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded data from {self.data_path} with {len(self.data)} rows")
        return self.data
    
    def create_features(self, df=None, lookback=3):
        """Create advanced features for time series forecasting."""
        if df is None:
            if self.data is None:
                raise ValueError("No data available. Please load data first.")
            df = self.data.copy()
            
        # Original features
        features = df.copy()
        
        # Add time-based features
        features['year_squared'] = features['year'] ** 2
        
        # Ensure 'population' is numeric, coercing None to NaN for calculations
        features['population'] = pd.to_numeric(features['population'], errors='coerce')
        
        # Create lagged features
        for i in range(1, lookback + 1):
            # Population lags
            features[f'population_lag_{i}'] = features['population'].shift(i)
            
            # Rate lags
            for rate in ['birth_rate', 'death_rate', 'migration_rate']:
                features[f'{rate}_lag_{i}'] = features[rate].shift(i)
            
            # Year-over-year changes
            if i == 1:
                features['population_yoy_change'] = features['population'].pct_change() * 100
                features['population_diff'] = features['population'].diff()
                
                for rate in ['birth_rate', 'death_rate', 'migration_rate']:
                    features[f'{rate}_yoy_change'] = features[rate].pct_change() * 100
                    features[f'{rate}_diff'] = features[rate].diff()
        
        # Interactions between features - handle potential division by zero
        features['birth_death_ratio'] = features['birth_rate'] / features['death_rate'].replace(0, np.nan)
        features['natural_growth'] = features['birth_rate'] - features['death_rate']
        features['total_growth'] = features['natural_growth'] + features['migration_rate']
        
        # Moving averages
        for window in [2, 3]:
            features[f'population_ma_{window}'] = features['population'].rolling(window=window).mean()
            for rate in ['birth_rate', 'death_rate', 'migration_rate']:
                features[f'{rate}_ma_{window}'] = features[rate].rolling(window=window).mean()
        
        # Fill NaN values with 0 for feature engineering columns
        for col in features.columns:
            if col not in ['year', 'population', 'birth_rate', 'death_rate', 'migration_rate']:
                features[col] = features[col].fillna(0)
        
        # Store feature names for prediction
        if self.feature_names is None:
            self.feature_names = [col for col in features.columns if col not in ['year', 'population']]
        
        return features
    
    def prepare_training_data(self, test_size=0.2):
        """Prepare data for training with advanced features."""
        # Create features
        features_df = self.create_features()
        
        if len(features_df) == 0:
            raise ValueError("Not enough data for training after feature creation")
        
        # Split features and target
        X = features_df[self.feature_names]
        y = features_df['population']
        
        # Split data respecting time order
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, test_size=0.2, params=None):
        """Train an optimized CatBoost model with custom parameters."""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_training_data(test_size)
        
        # Default parameters (can be overridden)
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE',
            'verbose': 100,
            'early_stopping_rounds': 50
        }
        
        # Override defaults with provided params
        if params:
            default_params.update(params)
        
        # Create CatBoost datasets
        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)
        
        # Initialize and train the model
        print("Training CatBoost model...")
        self.model = CatBoostRegressor(**default_params, random_seed=42)
        self.model.fit(train_pool, eval_set=test_pool)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        self.metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        print("\nModel Performance Metrics:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:,.4f}")
        
        # Feature importance
        feature_importance = self.model.get_feature_importance(train_pool)
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        return self.metrics, feature_importance_df
    
    def save_model(self, model_path):
        """Save the trained model."""
        if self.model is not None:
            self.model.save_model(model_path)
            print(f"Model saved to {model_path}")
        else:
            raise ValueError("Model hasn't been trained yet")
    
    def load_model(self, model_path):
        """Load a trained model."""
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def forecast_future(self, years=6):
        """Forecast population for future years with error bounds."""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")
        
        # Get the historical data
        hist_data = self.data.copy()
        last_year = hist_data['year'].max()
        
        # DataFrame to store forecasts
        forecasts = []
        
        # Current state for prediction
        current_data = hist_data.copy()
        
        # Generate predictions for each future year
        for i in range(years):
            # Year to predict
            next_year = last_year + i + 1
            
            # Create a copy for feature generation
            pred_data = current_data.copy()
            
            # Add a new row for the next year
            new_row = pd.DataFrame([{
                'year': next_year,
                'population': None,  # Will be predicted
                'birth_rate': self._extrapolate_rate('birth_rate', next_year),
                'death_rate': self._extrapolate_rate('death_rate', next_year),
                'migration_rate': self._extrapolate_rate('migration_rate', next_year)
            }])
            
            # Append the new row
            pred_data = pd.concat([pred_data, new_row], ignore_index=True)
            
            # Generate features for prediction
            features_df = self.create_features(pred_data)
            
            # Get the last row with all features
            pred_features = features_df.iloc[-1:][self.feature_names]
            
            # Predict population
            pred_pop = self.model.predict(pred_features)[0]
            
            # Update the new row with the prediction
            new_row['population'] = pred_pop
            
            # Add to forecasts
            forecasts.append(new_row)
            
            # Update current data for next iteration
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        # Combine all forecasts
        forecast_df = pd.concat(forecasts, ignore_index=True)
        
        # Add confidence intervals
        error_margin = self.metrics.get('test_rmse', 0)
        if error_margin:
            forecast_df['lower_bound'] = forecast_df['population'] - 1.96 * error_margin
            forecast_df['upper_bound'] = forecast_df['population'] + 1.96 * error_margin
        else:
            forecast_df['lower_bound'] = forecast_df['population'] * 0.98  # Fallback: 2% below
            forecast_df['upper_bound'] = forecast_df['population'] * 1.02  # Fallback: 2% above
        
        # Add year-over-year change
        previous_pop = hist_data['population'].iloc[-1]
        for i, row in forecast_df.iterrows():
            yoy_change = (row['population'] - previous_pop) / previous_pop * 100
            forecast_df.loc[i, 'year_over_year_change'] = yoy_change
            previous_pop = row['population']
        
        # Add population in millions
        forecast_df['population_millions'] = forecast_df['population'] / 1_000_000
        
        return forecast_df
    
    def _extrapolate_rate(self, rate_col, target_year):
        """Extrapolate demographic rates using linear regression."""
        years = self.data['year'].values
        rates = self.data[rate_col].values
        trend = np.polyfit(years, rates, 1)
        return np.polyval(trend, target_year)
    
    def generate_report(self, forecast_df, save_path=None):
        """Generate a comprehensive forecast report."""
        report = {}
        
        # Basic statistics
        report['start_year'] = self.data['year'].min()
        report['end_year'] = self.data['year'].max()
        report['forecast_years'] = forecast_df['year'].min(), forecast_df['year'].max()
        report['current_population'] = self.data['population'].iloc[-1]
        report['forecast_population'] = forecast_df['population'].iloc[-1]
        
        # Growth metrics
        total_change = (report['forecast_population'] - report['current_population']) 
        report['total_change'] = total_change
        report['percent_change'] = (total_change / report['current_population']) * 100
        report['avg_annual_growth'] = report['percent_change'] / len(forecast_df)
        
        # Model performance
        report['model_metrics'] = self.metrics
        
        # Add forecasts
        report['forecasts'] = forecast_df
        
        # Save report
        if save_path:
            forecast_df.to_csv(save_path, index=False)
            print(f"Forecast report saved to {save_path}")
        
        return report

def plot_forecast_with_intervals(historical_data, forecast_data, save_path=None):
    """Plot historical data and forecasts with confidence intervals."""
    plt.figure(figsize=(12, 8))
    
    # Plot historical data
    plt.plot(historical_data['year'], historical_data['population'], 
             'o-', color='blue', label='Historical Data')
    
    # Plot forecast
    plt.plot(forecast_data['year'], forecast_data['population'], 
             's--', color='red', label='Forecast')
    
    # Plot confidence intervals
    plt.fill_between(forecast_data['year'], 
                     forecast_data['lower_bound'], 
                     forecast_data['upper_bound'], 
                     color='red', alpha=0.2, label='95% Confidence Interval')
    
    # Format the plot
    plt.title('Population Forecast with Confidence Intervals', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Population', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Format y-axis to show millions
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M'))
    
    # Add value labels
    for x, y in zip(historical_data['year'][-3:], historical_data['population'][-3:]):
        plt.annotate(f'{int(y/1e6)}M', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    for x, y in zip(forecast_data['year'], forecast_data['population']):
        plt.annotate(f'{int(y/1e6)}M', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center',
                    color='darkred')
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_demographic_indicators(historical_data, forecast_data, save_path=None):
    """Plot demographic indicators (birth rate, death rate, migration rate)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical indicators
    ax.plot(historical_data['year'], historical_data['birth_rate'], 'o-', label='Birth Rate (Historical)')
    ax.plot(historical_data['year'], historical_data['death_rate'], 's-', label='Death Rate (Historical)')
    ax.plot(historical_data['year'], historical_data['migration_rate'], '^-', label='Migration Rate (Historical)')
    
    # Plot forecasted indicators
    ax.plot(forecast_data['year'], forecast_data['birth_rate'], 'o--', label='Birth Rate (Forecast)')
    ax.plot(forecast_data['year'], forecast_data['death_rate'], 's--', label='Death Rate (Forecast)')
    ax.plot(forecast_data['year'], forecast_data['migration_rate'], '^--', label='Migration Rate (Forecast)')
    
    # Format the plot
    ax.set_title('Demographic Indicators: Historical and Forecast', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def create_forecast_table(forecast_data):
    """Create a formatted table with forecast data."""
    table_data = forecast_data[['year', 'population', 'year_over_year_change', 
                              'lower_bound', 'upper_bound', 'population_millions']].copy()
    
    # Format columns
    table_data = table_data.round({
        'population': 0, 
        'year_over_year_change': 2, 
        'lower_bound': 0, 
        'upper_bound': 0,
        'population_millions': 2
    })
    
    # Rename columns for display
    table_data.columns = ['Year', 'Population', 'Change (%)', 
                         'Lower Bound (95%)', 'Upper Bound (95%)', 'Population (M)']
    
    return table_data

def create_historical_and_forecast_table(historical_data, forecast_data):
    """Create a formatted table with historical and forecasted population data and dynamics."""
    # Select and rename historical columns
    hist_df = historical_data[['year', 'population']].copy()
    hist_df.rename(columns={'year': 'Year', 'population': 'Population'}, inplace=True)
    hist_df['Type'] = 'Historical'

    # Select and rename forecast columns
    fore_df = forecast_data[['year', 'population']].copy()
    fore_df.rename(columns={'year': 'Year', 'population': 'Population'}, inplace=True)
    fore_df['Type'] = 'Forecast'

    # Combine data
    combined_df = pd.concat([hist_df, fore_df], ignore_index=True)
    combined_df.sort_values(by='Year', inplace=True)

    # Calculate Year-over-Year Change
    combined_df['Population'] = pd.to_numeric(combined_df['Population'], errors='coerce')
    combined_df['YoY Change (%)'] = combined_df['Population'].pct_change() * 100
    
    # Format for display
    combined_df['Population'] = combined_df['Population'].round(0).astype(int)
    combined_df['YoY Change (%)'] = combined_df['YoY Change (%)'].round(2)
    
    # Reorder columns for the final table
    display_table = combined_df[['Year', 'Population', 'YoY Change (%)', 'Type']]
    
    return display_table

if __name__ == "__main__":
    # Set up paths
    base_path = Path(__file__).parent.parent
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