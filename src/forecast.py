import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class PopulationForecaster:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.data = None
        self.feature_names = None
        
    def load_data(self):
        """Load and prepare the population data."""
        self.data = pd.read_csv(self.data_path)
        return self.data
    
    def prepare_features(self, df=None, lookback=3):
        """Prepare features for the model using historical data."""
        if df is None:
            df = self.data.copy()
        
        # Create lagged features
        for i in range(1, lookback + 1):
            df[f'population_lag_{i}'] = df['population'].shift(i)
            df[f'birth_rate_lag_{i}'] = df['birth_rate'].shift(i)
            df[f'death_rate_lag_{i}'] = df['death_rate'].shift(i)
            df[f'migration_rate_lag_{i}'] = df['migration_rate'].shift(i)
        
        # Store feature names for prediction
        if self.feature_names is None:
            self.feature_names = [col for col in df.columns if col not in ['year', 'population']]
        
        # Drop rows with NaN values (due to lag creation)
        df = df.dropna()
        
        if len(df) > 0:
            # Prepare X and y
            X = df[self.feature_names]
            y = df['population'] if 'population' in df.columns else None
            return X, y
        return None, None
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train the CatBoost model."""
        X, y = self.prepare_features()
        if X is None or y is None:
            raise ValueError("Not enough data for training")
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            random_state=random_state,
            verbose=False
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
        }
        
        return metrics
    
    def forecast_future(self, years=5):
        """Forecast population for future years."""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")
        
        # Get the last available data
        last_data = self.data.iloc[-1:].copy()
        forecasts = []
        
        current_data = last_data.copy()
        for _ in range(years):
            # Create a copy of the current data for prediction
            pred_data = current_data.copy()
            
            # Prepare features for prediction
            X, _ = self.prepare_features(pred_data)
            
            if X is None:
                break
                
            # Make prediction
            pred = self.model.predict(X)
            
            # Create new row with the prediction
            new_row = current_data.copy()
            new_row['year'] += 1
            new_row['population'] = pred
            
            # Estimate other rates based on trends (simple linear extrapolation)
            for rate in ['birth_rate', 'death_rate', 'migration_rate']:
                trend = np.polyfit(self.data['year'], self.data[rate], 1)
                new_row[rate] = np.polyval(trend, new_row['year'].iloc[0])
            
            forecasts.append(new_row)
            current_data = new_row
        
        return pd.concat(forecasts)

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

def plot_forecast(historical_data, forecast_data, save_path=None):
    """Plot historical data and forecasts."""
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data['year'], historical_data['population'], 
             label='Historical', marker='o')
    plt.plot(forecast_data['year'], forecast_data['population'], 
             label='Forecast', marker='o', linestyle='--')
    
    plt.title('Population Forecast')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Set up paths
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "population_2015_2024.csv"
    model_path = base_path / "model" / "catboost_model.cbm"
    
    # Initialize and train forecaster
    forecaster = PopulationForecaster(data_path)
    data = forecaster.load_data()
    
    # Train model and print metrics
    metrics = forecaster.train_model()
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:,.0f}")
    
    # Save the trained model
    forecaster.save_model(model_path)
    
    # Make future predictions
    future_predictions = forecaster.forecast_future(years=5)
    print("\nPopulation Forecasts:")
    print(future_predictions[['year', 'population']].to_string(index=False))
    
    # Plot results
    plot_forecast(data, future_predictions) 