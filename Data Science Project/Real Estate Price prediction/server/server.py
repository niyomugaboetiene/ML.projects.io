from flask import Flask, request, jsonify
import util

app = Flask(__name__)

# @app.route("/hello")
# def say_hello():
#     return "Hi"


@app.route("/get_location_names")
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

if __name__ == "__main__":
    print("Flask server started successfully")
    app.run()