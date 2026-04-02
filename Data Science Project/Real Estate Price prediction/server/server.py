from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route("/hello")
def say_hello():
    return "Hi"


if __name__ == "__main__":
    print("Flask server started successfully")
    app.run()