import xarray as xr
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.preprocessing import StandardScaler, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional

class ClimateDataProcessor:
    """Class to handle loading and processing of climate data."""
    def __init__(self, dataset_path: str, data_vars: List[str]):
        self.dataset_path = dataset_path
        self.data_vars = data_vars
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> Optional[xr.Dataset]:
        """Load the dataset from the given path."""
        try:
            return xr.open_zarr(self.dataset_path)
        except FileNotFoundError:
            print("Dataset file not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def preprocess_dataset(self) -> None:
        """Preprocess the dataset by adjusting longitude and sorting."""
        if self.dataset is not None and "longitude" in self.dataset.coords:
            self.dataset.coords['longitude'] = (self.dataset.coords['longitude'] + 180) % 360 - 180
            self.dataset = self.dataset.sortby(self.dataset.longitude)
        else:
            print("Dataset not loaded or 'longitude' coordinate not found.")

    def merge_data_vars(self) -> Optional[xr.Dataset]:
        """Merge specified data variables from the dataset."""
        try:
            merged_data = [self.dataset[var] for var in self.data_vars if var in self.dataset]
            return xr.merge(merged_data, compat='override') if merged_data else None
        except KeyError as e:
            print(f"Variable not found in dataset: {e}")
            return None
        except Exception as e:
            print(f"An error occurred during merging: {e}")
            return None

    @staticmethod
    def convert_to_dataframe(data_array: xr.DataArray) -> pd.DataFrame:
        """Convert an xarray DataArray to a pandas DataFrame."""
        return data_array.to_dataframe()

    @staticmethod
    def remove_oceans(df: pd.DataFrame) -> pd.DataFrame:
        """Remove ocean data points from the dataframe."""
        return df[df['lsm'] == 1]

    @staticmethod
    def encode_time(df: pd.DataFrame) -> pd.DataFrame:
        """Encode time in the dataframe into cyclical features."""
        df = df.reset_index()
        df['month'] = df['time'].dt.month
        df['week'] = df['time'].dt.isocalendar().week
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 53)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 53)
        df = df.drop(columns=['month', 'week', "time"])
        return df

    @staticmethod
    def manage_indexes(df: pd.DataFrame) -> pd.DataFrame:
        """Reset the index of the dataframe and encode time."""
        df = df.reset_index()
        return ClimateDataProcessor.encode_time(df)

    @staticmethod
    def normalize_data(data_dict: Dict[str, pd.DataFrame], imputer: SimpleImputer, scaler: StandardScaler) -> Dict[str, pd.DataFrame]:
        """Normalize and impute the data in the given dictionary."""
        for key, df in data_dict.items():
            original_index = df.index
            df_imputed = imputer.fit_transform(df) if key == 'X' else imputer.transform(df)
            df_normalized = scaler.fit_transform(df_imputed) if key == 'X' else scaler.transform(df_imputed)
            data_dict[key] = pd.DataFrame(df_normalized, columns=df.columns, index=original_index)
        return data_dict

    @staticmethod
    def get_features_and_target(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Split the dataframe into features and target columns."""
        X = df[feature_cols]
        y = df[target_col]
        return X, y

class LinearBlock(nn.Module):
    """A Linear Block consisting of Linear -> LeakyReLU -> Dropout."""
    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu", dropout: float = 0.0, slope: float = -0.01):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU(slope) if activation == "leaky_relu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.fc(X)))

class MLP(nn.Module):
    """Multilayer Perceptron for classification tasks."""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        layers = [LinearBlock(input_dim, hidden_dims[0], activation, dropout)]
        layers += [LinearBlock(h_dim, hidden_dims[i + 1], activation, dropout) for i, h_dim in enumerate(hidden_dims[:-1])]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the model."""
        return self.layers(x)

    def _init_weights(self) -> None:
        """Initialize weights of the MLP."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

class ClimateModelTrainer:
    """
    Trainer class for the climate prediction model.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []

    def train(self, num_epochs: int) -> None:
        """Train the model for a specified number of epochs."""
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
                self.optimizer.zero_grad()
                inputs, targets = self._prepare_batch(batch)
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            avg_val_loss = self._validate()
            self.val_losses.append(avg_val_loss)
            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

    def _validate(self) -> None:
        """Validate the model on the validation dataset."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = self._prepare_batch(batch)
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)
                total_loss += loss.item()
            print(f"Validation Loss: {total_loss / len(self.val_loader)}")

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch data for training."""
        inputs = batch['X'].to(self.device, dtype=torch.float)
        targets = batch['y'].to(self.device, dtype=torch.float).unsqueeze(1)
        return inputs, targets

    def plot_losses(self) -> None:
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 4))
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        # Plot validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.show()

# Functions for Logistic Regression and Random Forest
def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train a logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a random forest classifier."""
    model = RandomForestClassifier(n_estimators=

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluate the given model."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='binary')
    print(f"Model Accuracy: {accuracy}, F1-Score: {f1}")

# Usage
dataset_path = '../../../../scratch/ssd004/scratch/tsepaole/SeasFireCube_v3.zarr'
dataset_processor = ClimateDataProcessor(dataset_path, data_vars)
dataset_processor.preprocess_dataset()
merged_data = dataset_processor.merge_data_vars()
if merged_data is not None:
    X = merged_data.sel(time=slice('2011-01-01', '2012-01-01')).load()
    y = merged_data.sel(time=slice('2011-01-09', '2012-01-09')).load()
    x_val = merged_data.sel(time=slice('2012-02-01', '2012-03-01')).load()
    y_val = merged_data.sel(time=slice('2012-02-09', '2012-03-09')).load()
    
    # Convert to DataFrame
    X_df = ClimateDataProcessor.convert_to_dataframe(X)
    y_df = ClimateDataProcessor.convert_to_dataframe(y)
    x_val_df = ClimateDataProcessor.convert_to_dataframe(x_val)
    y_val_df = ClimateDataProcessor.convert_to_dataframe(y_val)
    
    # Preprocess DataFrames
    X_df = ClimateDataProcessor.remove_oceans(X_df)
    y_df = ClimateDataProcessor.remove_oceans(y_df)
    x_val_df = ClimateDataProcessor.remove_oceans(x_val_df)
    y_val_df = ClimateDataProcessor.remove_oceans(y_val_df)
    
    # Manage indexes and encode time
    X_df = ClimateDataProcessor.manage_indexes(X_df)
    y_df = ClimateDataProcessor.manage_indexes(y_df)
    x_val_df = ClimateDataProcessor.manage_indexes(x_val_df)
    y_val_df = ClimateDataProcessor.manage_indexes(y_val_df)
    
    # Normalize the data
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    scaler = StandardScaler()
    data_dict = {'X': X_df, 'y': y_df, 'x_val': x_val_df, 'y_val': y_val_df}
    normalized_data = ClimateDataProcessor.normalize_data(data_dict, imputer, scaler)
    
    # Split data into features and target
    feature_cols = ["month_sin", "month_cos", "week_sin", "week_cos", "latitude", "longitude", 'lst_day', 'ndvi', 'rel_hum', 'ssrd', 'sst', "t2m_min", "tp", "vpd"]
    target_col = 'gfed_ba' # Assuming 'gfed_ba' is a categorical variable for classification
    X_train, y_train = ClimateDataProcessor.get_features_and_target(normalized_data['X'], feature_cols, target_col)
    X_test, y_test = ClimateDataProcessor.get_features_and_target(normalized_data['x_val'], feature_cols, target_col)
    
    # Train Logistic Regression
    log_reg_model = train_logistic_regression(X_train, y_train)
    evaluate_model(log_reg_model, X_test, y_test)
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)
    
    # Prepare data for PyTorch models
    X_train_torch = torch.from_numpy(X_train.values).float()
    y_train_torch = torch.from_numpy(y_train.values).float().unsqueeze(1)
    X_test_torch = torch.from_numpy(X_test.values).float()
    y_test_torch = torch.from_numpy(y_test.values).float().unsqueeze(1)
    
    train_dataset = ClimateDataset(X_train_torch, y_train_torch)
    test_dataset = ClimateDataset(X_test_torch, y_test_torch)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Define and train MLP
    mlp = MLP(input_dim=len(feature_cols), hidden_dims=[64, 32], output_dim=1)
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    trainer = ClimateModelTrainer(mlp, train_loader, test_loader, optimizer, loss_fn)
    trainer.train(num_epochs=5)
    trainer.plot_losses()
