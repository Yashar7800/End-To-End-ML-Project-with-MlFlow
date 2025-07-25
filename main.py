from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformationPipeline
from mlProject.pipeline.stage04_model_trainer import ModelTrainerTrainingPipeline
from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainigPipeline



STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"-----> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"----->stage {STAGE_NAME} completed <<<<<< \n\nx============x")

except Exception as e:
    
        logger.exception(e)
        raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f"-----> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f"----->stage {STAGE_NAME} completed <<<<<< \n\nx============x")

except Exception as e:
    
        logger.exception(e)
        raise e


STAGE_NAME=  "Data Transformation Stage"
try:
    logger.info(f'>>>> stage {STAGE_NAME} started <<<<<<')
    obj = DataTransformationPipeline()
    obj.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<< \n\nx============x')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME=  "Model Trainer Stage"
try:
    logger.info(f'>>>> stage {STAGE_NAME} started <<<<<<')
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<< \n\nx============x')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = 'Model Evaluation Stage'
try:
    logger.info(f'>>>> stage {STAGE_NAME} started <<<<<<')
    obj = ModelEvaluationTrainigPipeline()
    obj.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<< \n\nx============x')
except Exception as e:
    logger.exception(e)
    raise e