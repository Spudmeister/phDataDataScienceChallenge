import click
from src.forecaster import Forecaster


@click.command
@click.option('--input_data_path', default ='./data.csv',
        help ='Path to input csv data')
@click.option('--model_path', default ='./xgb_model.json',
        help ='Path to forecast model')
@click.option('--output_path', default ='./forecasts.csv',
        help ='Path for forecast output')
def run(model_path, input_data_path, output_path):
   Forecaster(model_path, input_data_path, output_path).forecast()
  
if __name__=="__main__":
    run()