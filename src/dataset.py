import torch
from torch.utils.data import Dataset
import pandas as pd
import joblib

class TitanicDataset(Dataset):
    def __init__(self, csv_path, preprocessor_path, has_label=True):
        self.preprocessor = joblib.load(preprocessor_path)
        df = pd.read_csv(csv_path)
        self.X = self.preprocessor.transform(df.drop(columns=['Survived'], errors='ignore'))
        self.X = torch.tensor(self.X, dtype=torch.float32)
        if has_label:
            self.y = torch.tensor(df['Survived'].values, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]