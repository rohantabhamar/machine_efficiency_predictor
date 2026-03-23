from src.model_training import ModelTraining
from src.data_preprocessing import DataProcessing

if __name__ == "__main__":

    prepocessing = DataProcessing("artifacts/raw/data.csv","artifacts/processed")
    prepocessing.run()
    trainer = ModelTraining("artifacts/processed/","artifacts/models")
    trainer.run()