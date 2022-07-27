import xgboost as xgb

class ForecastModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.__load_model()
        

    def __load_model(self):
        model = xgb.Booster()
        model.load_model(self.model_path)

        return model
        
        
    def predict(self, X):
        d_matrix_x = xgb.DMatrix(X)
        return self.model.predict(d_matrix_x)