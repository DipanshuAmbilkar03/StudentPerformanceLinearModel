from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

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

        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()
