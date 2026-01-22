from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('model/titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        fare = float(request.form['fare'])

        features = np.array([[pclass, sex, age, fare]])
        result = model.predict(features)[0]

        prediction = 'Survived' if result == 1 else 'Did Not Survive'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
