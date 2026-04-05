import json
import numpy as np
import pickle
import os

__location = None
__data_columns = None
__model = None

def get_location_names():
   return __location

def get_estimate_price(location, sqft, bhk, bath):
    try:
       loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
         x[loc_index] = 1

    return round(__model.predict([x])[0])

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
def load_saved_artifacts():
    print("Loading saved artifacts start")

    global __data_columns
    global __location 

    with open(os.path.join(ARTIFACTS_DIR, "columns.json"), "r") as f:
       __data_columns =  json.load(f)['data_columns']
       __location = __data_columns[3:]

    global __model
    with open(os.path.join(ARTIFACTS_DIR, "Real_estate_price_prediction.pickle"), "rb") as f:
        __model = pickle.load(f)

    print("Loading saved artifacts is done")

if __name__ == "__main__":
    load_saved_artifacts()
    # print(get_location_names())
    print(get_estimate_price('1st phase jp nagar', 500, 2, 1))
    print(get_estimate_price('yeshwanthpur', 1000, 3, 2))
    print(get_estimate_price('sarjapur', 700, 2, 2))

    print("Current working dir:", os.getcwd())
    print("File exists?", os.path.exists(os.path.join(ARTIFACTS_DIR, "columns.json")))
    print("Model exists?", os.path.exists(os.path.join(ARTIFACTS_DIR, "Real_estate_price_prediction.pickle")))