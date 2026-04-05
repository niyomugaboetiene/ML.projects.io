# import json
# import numpy as np
# import pickle

# __location = None
# __data_columns = None
# __model = None

# def get_location_names():
#    return __location

# def get_estimate_price(location, sqft, bhk, bath):
#     try:
#        loc_index = __data_columns.index(location.lower())
#     except:
#         loc_index = -1

#     x = np.zeros(len(__data_columns))
#     x[0] = sqft
#     x[1] = bath
#     x[2] = bhk

#     if loc_index >= 0:
#          x[loc_index] = 1

#     return round(__model.predict([x])[0])

# def load_saved_artifacts():
#     print("Loading saved artifacts start")

#     global __data_columns
#     global __location 

#     with open("./artifacts/columns.json", "r") as f:
#        __data_columns =  json.load(f)['data_columns']
#        __location = __data_columns[3:]

#     global __model
#     with open("./artifacts/Real_estate_price_prediction.pickle", "rb") as f:
#         __model = pickle.load(f)

#     print("Loading saved artifacts is done")

# if __name__ == "__main__":
#     load_saved_artifacts()
#     # print(get_location_names())
#     print(get_estimate_price('1st phase jp nagar', 500, 2, 1))
#     print(get_estimate_price('yeshwanthpur', 1000, 3, 2))
#     print(get_estimate_price('sarjapur', 700, 2, 2))
# util.py
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
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0])

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

def load_saved_artifacts():
    global __data_columns, __location, __model
    print("Loading saved artifacts...")

    columns_file = os.path.join(ARTIFACTS_DIR, "columns.json")
    model_file = os.path.join(ARTIFACTS_DIR, "Real_estate_price_prediction.pickle")

    if not os.path.exists(columns_file):
        raise FileNotFoundError(f"{columns_file} not found on server!")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"{model_file} not found on server!")

    with open(columns_file, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __location = __data_columns[3:]  

    with open(model_file, "rb") as f:
        __model = pickle.load(f)

    print(f"Artifacts loaded: {len(__location)} locations, model loaded.")