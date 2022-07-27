from src.ETL import load_data, save_forecasts
from src.model import ForecastModel

class Forecaster:
    def __init__(self, model_path="", input_path="", output_path=""):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        
        self.model = ForecastModel(model_path)
        
        
    def forecast(self):
        ids, X = load_data(self.input_path)
        
        predictions = self.model.predict(X)
        
        save_forecasts(ids, predictions, self.output_path)