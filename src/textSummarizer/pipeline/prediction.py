from simpletransformers.classification import ClassificationModel
from textSummarizer.config.configuration import ConfigurationManager
import torch

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_predict_config()
        self.model = ClassificationModel(
            "roberta",
            str(self.config.model_dir),
            use_cuda=False
        )
        
    def predict(self, text):
        predictions, raw_outputs = self.model.predict([text])
        return predictions[0]