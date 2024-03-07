from simpletransformers.classification import ClassificationModel
import torch
from src.textSummarizer.components.data_loader import DataLoaders
from textSummarizer.entity import ModelTrainerConfig
from sklearn.utils.class_weight import compute_class_weight

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, data_loader: DataLoaders):
        self.config = config
        self.model = None
        self.data_loader = data_loader
        
    def train_model(self):
        train_df = self.data_loader.load_and_preprocess_data()
        class_weights = compute_class_weight('balanced', classes=train_df['labels'].unique(), y=train_df['labels'])
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ClassificationModel(
            self.config.model_type,
            self.config.model_name,
            num_labels=3,
            use_cuda = False,
            args={
                'output_dir': f"{self.config.output_dir}/outputs",
                'cache_dir': f"{self.config.output_dir}/cache",
                'tensorboard_dir': f"{self.config.output_dir}/runs",
                'reprocess_input_data': self.config.reprocess_input_data,
                'overwrite_output_dir': self.config.overwrite_output_dir,
                'fp16': self.config.fp16,
                'weight': class_weights_dict,
                'do_lower_case': self.config.do_lower_case,
                'manual_seed': self.config.manual_seed,
                'use_multiprocessing': self.config.use_multiprocessing,
                'use_multiprocessing_for_evaluation': self.config.use_multiprocessing_for_evaluation,
                'thread_count': self.config.thread_count,
                'save_eval_checkpoints': self.config.save_eval_checkpoints,
                'save_model_every_epoch': self.config.save_model_every_epoch,
                'early_stopping_metric': self.config.early_stopping_metric,
                'early_stopping_metric_minimize': self.config.early_stopping_metric_minimize,
                'early_stopping_patience': self.config.early_stopping_patience,
                
            }
        )
        
        self.model.train_model(train_df)
        
    def evaluate_model(self, eval_df):
        result, model_outputs, wrong_predictions = self.model.eval_model(eval_df)
        return result, model_outputs, wrong_predictions
    
    def predict(self, data):
        predictions, raw_outputs = self.model.predict(data['text'])
        return predictions, raw_outputs