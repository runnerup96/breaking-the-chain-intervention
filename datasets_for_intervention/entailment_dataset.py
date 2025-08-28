from datasets import Dataset


class EntailmentDataset(Dataset):
    def __init__(self, file_path):
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]