from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_training_data(data):
    X_train = data.iloc[:-1].drop(columns=['Date', 'Close', 'Adj Close'])
    y_train = data.iloc[:-1]['Close']
    return X_train, y_train

def prepare_testing_data(data):
    X_test = data.iloc[-1:].drop(columns=['Date', 'Close', 'Adj Close'])
    y_test = data.iloc[-1:]['Close']
    return X_test, y_test

def impute_missing_values(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed, imputer

def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

def prepare_new_data_point(new_data_point, X_train, imputer):
    new_data_point = new_data_point.reindex(columns=X_train.columns, fill_value=None)
    new_data_point_imputed = imputer.transform(new_data_point)
    return new_data_point_imputed

def predict_price(model, new_data_point_imputed):
    predicted_price = model.predict(new_data_point_imputed)
    print("Predicted Price for 2024-04-23:", predicted_price)

# Main script
def main():
    # Load the dataset
    data = load_data("data/TQQQ_with_indicators.csv")
    
    # Prepare training and testing data
    X_train, y_train = prepare_training_data(data)
    X_test, y_test = prepare_testing_data(data)
    
    # Impute missing values
    X_train_imputed, X_test_imputed, imputer = impute_missing_values(X_train, X_test)
    
    # Train the model
    model = train_model(X_train_imputed, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test_imputed, y_test)
    
    # Define and prepare new data point
    new_data_point = pd.DataFrame({ ... })  # Your new data point here
    new_data_point_imputed = prepare_new_data_point(new_data_point, X_train, imputer)
    
    # Make predictions
    predict_price(model, new_data_point_imputed)

if __name__ == "__main__":
    main()
