from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(app.root_path, "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.get('feature', '')

    # remove empty values + strip spaces
    features_lst = [f.strip() for f in features.split(',') if f.strip()]

    if len(features_lst) == 0:
        return render_template('index.html', messages=["Please enter comma-separated numeric features."])

    # convert to float
    try:
        np_features = np.array(features_lst, dtype=np.float32)
    except ValueError:
        return render_template('index.html', messages=["Conversion error: make sure all inputs are numbers."])

    # reshape and predict
    try:
        pred = model.predict(np_features.reshape(1, -1))
    except Exception as e:
        # show helpful message in UI (but also check server log)
        print("Prediction error:", e)
        return render_template('index.html', messages=[f"Prediction error: {e}"])

    output = ["cancrous" if int(pred[0]) == 1 else "non-cancrous"]
    # pass as "messages" because your template uses that name
    return render_template('index.html', messages=output)


if __name__ == "__main__":
    app.run(debug=True)
