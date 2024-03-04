from textSummarizer.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage02_model_trainer import ModelTrainingPipeline
from textSummarizer.logging import logger

STAGE_NAME = "Stage 01: Data Ingestion Stage"
try:
    logger.info(f"Starting {STAGE_NAME}")
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
    logger.info(f"Completed {STAGE_NAME}")
except Exception as e:
    logger.exception(f"Failed {STAGE_NAME} with exception: {e}")
    raise e

STAGE_NAME = "Stage 02: Model Training Stage"
try:
    logger.info(f"Starting {STAGE_NAME}")
    pipeline = ModelTrainingPipeline()
    pipeline.main()
    logger.info(f"Completed {STAGE_NAME}")
except Exception as e:
    logger.exception(f"Failed {STAGE_NAME} with exception: {e}")
    raise e

