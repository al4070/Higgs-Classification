import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class HiggsDataset(Dataset):
    def __init__(self, file_path, chunk_size=1000000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.offsets = self._get_offsets()

    def _get_offsets(self):
        # Just counts how many chunks you'll need
        total_lines = sum(1 for _ in open(self.file_path)) - 1  # skip header
        return list(range(1, total_lines, self.chunk_size))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        skip_rows = self.offsets[idx]
        df = pd.read_csv(self.file_path, skiprows=skip_rows, nrows=self.chunk_size, header=None)
        X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
        y = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32)
        return X, y

dataset = HiggsDataset('/Users/tribrid/Desktop/Machine Learning/Higgs-Classification/HIGGS.csv')
loader = DataLoader(dataset, batch_size=1, shuffle=False)


column_names = [
    "label",  # 0
    # Low-level features (1–21)
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_btag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_btag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_btag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_btag",
    # High-level features (22–28)
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
]


df = pd.read_csv('/Users/tribrid/Desktop/Machine Learning/Higgs-Classification/HIGGS.csv', nrows=5, header=None, names=column_names)
print(df)

