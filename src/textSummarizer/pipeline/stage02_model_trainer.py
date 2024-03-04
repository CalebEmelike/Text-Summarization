from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.components.data_loader import DataLoaders
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        data_loader = DataLoaders(model_trainer_config)
        model_trainer = ModelTrainer(model_trainer_config, data_loader)
        model_trainer.train_model()