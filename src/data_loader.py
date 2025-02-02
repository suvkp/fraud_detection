import pandas as pd
from functools import cached_property

class DataLoader:
    def __init__(self, path):
        self.path = path
    
    @cached_property
    def dataset(self):
        return pd.read_csv(self.path)