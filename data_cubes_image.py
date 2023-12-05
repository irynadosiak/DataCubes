import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr
import pandas as pd
import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader, TensorDataset

class TensorOperations:
    """ Class to handle operations related to PyTorch tensors. """
    @staticmethod
    def save_tensors(tensors: List[torch.Tensor], filename: str) -> None:
        torch.save(tensors, filename)

    @staticmethod
    def load_tensors(filename: str) -> List[torch.Tensor]:
        return torch.load(filename)

class DatasetManipulator:
    """ Class to handle operations on xarray datasets. """
    def __init__(self, dataset_path: str):
        self.dataset = xr.open_dataset(dataset_path, chunks=None)
        self.dataset.load()

    @staticmethod
    def get_dates(dataset: xr.Dataset, start_day: str, num_days: int) -> List[str]:
        time_index = pd.to_datetime(dataset.time.values)
        start_index = time_index.get_loc(pd.to_datetime(start_day))
        time_slice = time_index[start_index: start_index + num_days]
        return [f"{date.year:04d}-{date.month:02d}-{date.day:02d}T{date.hour:02d}:{date.minute:02d}:{date.second:02d}.{date.nanosecond:09d}" for date in time_slice]

    def normalize_features(self, res_maxes: Dict[str, float]) -> None:
        for variable, max_val in res_maxes.items():
            self.dataset[variable] /= max_val

class UNet(nn.Module):
    """ U-Net model for semantic segmentation. """
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.encoder_conv1 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.up_transpose_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = F.relu(self.encoder_conv1(x))
        x2 = self.pool(x1)
        x3 = F.relu(self.encoder_conv2(x2))
        x4 = self.pool(x3)

        # Decoder
        x5 = self.up_transpose_conv1(x4)
        x6 = torch.cat([x5, x3], dim=1)
        x7 = F.relu(self.decoder_conv1(x6))
        x8 = F.relu(self.decoder_conv2(x7))
        return x8

class TensorDataPreparation:
    """ Class for preparing tensor data from xarray datasets. """
    def __init__(self, variable_dataset: xr.Dataset, target_dataset: xr.Dataset, division_size: int = 128, threshold: int = 80):
        self.variable_dataset = variable_dataset
        self.target_dataset = target_dataset
        self.division_size = division_size
        self.threshold = threshold
        self.lat_division_size = self.variable_dataset.dims["latitude"] // division_size
        self.lon_division_size = self.variable_dataset.dims["longitude"] // division_size

    def prepare_data(self, days: List[str], is_target: bool = False) -> List[torch.Tensor]:
        datasets = []
        dataset = self.target_dataset if is_target else self.variable_dataset

        for day in days:
            for i in range(self.lat_division_size):
                for j in range(self.lon_division_size):
                    lat_slice = slice(i * self.division_size, (i + 1) * self.division_size)
                    lon_slice = slice(j * self.division_size, (j + 1) * self.division_size)
                    subset = dataset.sel(time=day).isel(latitude=lat_slice, longitude=lon_slice).to_array(dim='variable')
                    null_values = subset.isnull().sum()
                    total_values = subset.count()
                    null_ratio = (null_values / (null_values + total_values)) * 100
                    if null_ratio.values < self.threshold:
                        numpy_array = subset.values
                        reshaped_array = numpy_array.reshape((subset.shape[0], subset.shape[1], subset.shape[2]))
                        torch_tensor = torch.from_numpy(reshaped_array)
                        torch_tensor[torch.isnan(torch_tensor)] = 0
                        datasets.append(torch_tensor)
        return datasets

# Initialize DatasetManipulator
dataset_path = '../../../../scratch/ssd004/scratch/tsepaole/SeasFireCube_v3.zarr'
manipulator = DatasetManipulator(dataset_path)

# Get dates and normalize features
start_day = "2011-01-01T00:00:00.000000000"
num_days = 139
days = manipulator.get_dates(manipulator.dataset, start_day, num_days)

# Prepare variables for normalization
true_variables = ["lst_day", "ndvi", "rel_hum", "ssrd", "sst", "t2m_min", "tp", "vpd"]
target_variables = ["gfed_ba"]  # Add target variables here
max_values = manipulator.get_max_values(true_variables)
manipulator.normalize_features(max_values)

# Initialize TensorDataPreparation with both variable and target datasets
data_preparer = TensorDataPreparation(manipulator.dataset, manipulator.dataset)

# Prepare tensor data for both input and target
input_datasets = data_preparer.prepare_data(days, is_target=False)
target_datasets = data_preparer.prepare_data(days, is_target=True)  # Assuming target data uses the same days

# Save prepared datasets
TensorOperations.save_tensors(input_datasets, "input_train_2011_2014.pt")
TensorOperations.save_tensors(target_datasets, "target_train_2011_2014.pt")

# Load prepared datasets
input_datasets_loaded = TensorOperations.load_tensors("input_train_2011_2014.pt")
target_datasets_loaded = TensorOperations.load_tensors("target_train_2011_2014.pt")

# Create DataLoader
tensor_dataset = TensorDataset(torch.stack(input_datasets_loaded), torch.stack(target_datasets_loaded))
data_loader = DataLoader(tensor_dataset, batch_size=8)

# Train UNet model
model = UNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch_number = 10  # Number of training epochs

for epoch in range(epoch_number):
    model.train()
    epoch_loss = 0.0
    for data, target in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(data_loader)}')
