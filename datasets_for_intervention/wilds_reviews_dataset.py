import json
from copy import deepcopy
from torch.utils.data import Dataset


class WildsReviewsDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_data = None
        self.binary_data = None
        self.load_and_prepare()
        
    def load_data(self):
        with open(self.data_path, 'r') as f:
            self.raw_data = json.load(f)

    def convert_to_binary(self):
        data = self.raw_data
            
        binary_data = []
        for item in data:
            new_item = deepcopy(item)
            if item["label"] in [0, 1]:
                new_item["label"] = 0  # Negative
                binary_data.append(new_item)
            elif item["label"] in [3, 4]:
                new_item["label"] = 1  # Positive
                binary_data.append(new_item)
            # Skip label 2 (neutral)
        
        self.binary_data = binary_data
    
    def load_and_prepare(self):
        self.load_data()
        self.convert_to_binary()

    def __len__(self):
        return len(self.binary_data)

    def __getitem__(self, i):
        return {"text": self.binary_data[i]['review'], "idx": self.binary_data[i]['sample_idx']}
    
