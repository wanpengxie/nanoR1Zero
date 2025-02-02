import json
import numpy as np

class DataLoader:
    def __init__(self, data_path, batch_size=16):
        self.data_path = data_path
        self.data = self.load_data()
        print ("data size: ", len(self.data))
        self.batch_size = batch_size

    def load_data(self):
        with open(self.data_path, 'r') as f:
            return json.load(f)
        
    def __iter__(self):
        np.random.shuffle(self.data)
        n = len(self.data)
        for i in range(0, n, self.batch_size):
            yield self.data[i:i+self.batch_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]