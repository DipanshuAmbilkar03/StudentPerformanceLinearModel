from flask import Flask, render_template, request
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.pkl")

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

model = pickle.load(open(MODEL_PATH, "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        previous = float(request.form["previous"])
        extra = int(request.form["extra"])
        sleep = float(request.form["sleep"])
        papers = float(request.form["papers"])

        input_data = np.array([[hours, previous, extra, sleep, papers]])
        prediction = round(model.predict(input_data)[0], 2)

    return render_template("index.html", prediction=prediction)

def handler(environ, start_response):
    return app(environ, start_response)
