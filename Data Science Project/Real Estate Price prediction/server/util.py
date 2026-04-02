import json

__location = None
__data_columns = None
__model = None

def get_location_names():
   pass


def load_saved_artifacts():
    print("Loading saved artifacts")

    with open("./artifacts/columns.json", "r") as f:
        json.load(f)
        
if __name__ == "__main__":
    print(get_location_names())