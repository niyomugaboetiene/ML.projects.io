from flask import Flask, request, jsonify

app = Flask(__name__)


if __name__ == "__main__":
    print("Flask server started successfully")
    app.run()