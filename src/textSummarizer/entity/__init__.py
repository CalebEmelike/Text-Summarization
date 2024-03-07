# Entity class for data ingestion configuration
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    output_dir: Path
    model_type: str
    model_name: str
    n_epochs: int
    train_batch_size: int
    eval_batch_size: int
    lr: float
    # class_weights: None
    reprocess_input_data: bool
    overwrite_output_dir: bool
    fp16: bool
    do_lower_case: bool
    manual_seed: int
    use_multiprocessing: bool
    use_multiprocessing_for_evaluation: bool
    thread_count: int
    save_eval_checkpoints: bool
    save_model_every_epoch: bool
    early_stopping_metric: str
    early_stopping_metric_minimize: bool
    early_stopping_patience: int
    use_cuda: bool
    
@dataclass(frozen=True)
class PredictConfig:
    root_dir: Path
    data_path: Path
    model_dir: Path