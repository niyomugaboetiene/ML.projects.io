# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import util

# app = Flask(__name__)
# CORS(app)

# # @app.route("/hello")
# # def say_hello():
# #     return "Hi"


# @app.route("/get_location_names")
# def get_location_names():
#     response = jsonify({
#         'locations': util.get_location_names()
#     })

#     response.headers.add("Access-Control-Allow-Origin", "*")

#     return response


# @app.route('/predict_home_price', methods=['POST'])
# def predict_home_price():

#     total_sqft = float(request.json['total_sqft'])
#     location = request.json['location']
#     bath = int(request.json['bath'])
#     bhk = int(request.json['bhk'])

#     response = jsonify({
#         'estimated_price': util.get_estimate_price(location, total_sqft, bhk, bath)
#     })

#     response.headers.add("Access-Control-Allow-Origin", "*")
#     return response

# if __name__ == "__main__":
#     print("Flask server started successfully")
#     util.load_saved_artifacts()
#     app.run()

from flask import Flask, request, jsonify
from flask_cors import CORS
import util

app = Flask(__name__)
CORS(app)

# Load artifacts immediately
util.load_saved_artifacts()

@app.route("/get_location_names")
def get_location_names():
    locations = util.get_location_names()
    if not locations:
        print("Warning: No locations loaded!")
    response = jsonify({'locations': locations})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.json['total_sqft'])
    location = request.json['location']
    bath = int(request.json['bath'])
    bhk = int(request.json['bhk'])

    estimated_price = util.get_estimate_price(location, total_sqft, bhk, bath)
    response = jsonify({'estimated_price': estimated_price})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    print("Flask server starting...")
    app.run(host="0.0.0.0", port=5000)