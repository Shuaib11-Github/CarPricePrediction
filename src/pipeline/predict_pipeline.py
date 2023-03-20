import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                km_driven: int,
                fuel: str,
                seller_type: str,
                transmission: str,
                owner: str,
                seats: float,
                torque_rpm: float,
                mil_kmpl: float,
                engine_cc: float,
                max_power_new: float,
                No_of_years: float):
        
        self.km_driven = km_driven
        self.fuel = fuel
        self.seller_type = seller_type
        self.transmission = transmission
        self.owner = owner
        self.seats = seats
        self.torque_rpm = torque_rpm
        self.mil_kmpl = mil_kmpl
        self.engine_cc = engine_cc
        self.max_power_new = max_power_new
        self.No_of_years = No_of_years

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "km_driven" : [self.km_driven],
                "fuel" : [self.fuel],
                "seller_type" : [self.seller_type],
                "transmission" : [self.transmission],
                "owner" : [self.owner],
                "seats" : [self.seats],
                "torque_rpm" : [self.torque_rpm],
                "mil_kmpl" : [self.mil_kmpl],
                "engine_cc" : [self.engine_cc],
                "max_power_new" : [self.max_power_new],
                "No_of_years" : [self.No_of_years]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)