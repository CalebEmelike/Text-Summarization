# Class that loads the data and preprocesses it in this format tmp = pd.DataFrame() tmp['text'] = train_filtered['text_short'] tmp['labels'] = train_filtered['label']

import pandas as pd
from textSummarizer.entity import ModelTrainerConfig

class DataLoaders:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def load_data(self):
        data = pd.read_csv(self.config.data_path)
        return data
    
    def preprocess_data(self, data):
        tmp = pd.DataFrame()
        tmp['text'] = data['text_short']
        tmp['labels'] = data['label']
        tmp['labels'] = tmp['labels'].map({'NOT': 0, 'TIN': 1, 'UNT': 2})
        # Sample sizes for each class
        # Filter by category and sample
        not_samples = tmp[tmp['labels'] == 0].sample(n=200, random_state=100)  # 'NOT' class
        tin_samples = tmp[tmp['labels'] == 1].sample(n=200, random_state=100)  # 'TIN' class
        unt_samples = tmp[tmp['labels'] == 2].sample(n=100, random_state=100)  
  
        # Concatenate the samples
        balanced_sample = pd.concat([not_samples, tin_samples, unt_samples]).reset_index(drop=True)

        return balanced_sample
    
    def load_and_preprocess_data(self):
        data = self.load_data()
        data = self.preprocess_data(data)
        return data

