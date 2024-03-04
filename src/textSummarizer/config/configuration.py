# Configuration Manager class
from textSummarizer.constants import *
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity import (DataIngestionConfig, 
                                  ModelTrainerConfig, PredictConfig)

class ConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        
        return data_ingestion_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingParams
        
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            output_dir = config.output_dir,
            model_type = params.model_type,
            model_name = params.model_name,
            n_epochs = params.n_epochs,
            train_batch_size = params.train_batch_size,
            eval_batch_size = params.eval_batch_size,
            lr = params.lr,
            class_weights = params.class_weights,
            reprocess_input_data = params.reprocess_input_data,
            overwrite_output_dir = params.overwrite_output_dir,
            fp16 = params.fp16,
            do_lower_case = params.do_lower_case,
            manual_seed = params.manual_seed,
            use_multiprocessing = params.use_multiprocessing,
            use_multiprocessing_for_evaluation = params.use_multiprocessing_for_evaluation,
            thread_count = params.thread_count,
            save_eval_checkpoints = params.save_eval_checkpoints,
            save_model_every_epoch = params.save_model_every_epoch,
            early_stopping_metric = params.early_stopping_metric,
            early_stopping_metric_minimize = params.early_stopping_metric_minimize,
            early_stopping_patience = params.early_stopping_patience,
            use_cuda = params.use_cuda
        )
        
        return model_trainer_config
    
    def get_predict_config(self) -> PredictConfig:
        config = self.config.model_predict
        prediction_config = PredictConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_dir = config.model_dir
        )
        
        return prediction_config
        