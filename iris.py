from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("iris.pkl")

# ชนิดของดอกไม้ในชุดข้อมูล Iris
iris_species = ['setosa', 'versicolor', 'virginica']

@app.route('/api/iris', methods=['POST'])
def predict_species():
    # รับค่า input จากแบบฟอร์ม
    sepal_length = float(request.form.get('sepal_length')) 
    sepal_width = float(request.form.get('sepal_width')) 
    petal_length = float(request.form.get('petal_length')) 
    petal_width = float(request.form.get('petal_width')) 

    # Prepare the input for the model
    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict using the model
    prediction = model.predict(x)

    # แปลงผลลัพธ์การพยากรณ์เป็นชนิดของดอกไม้
    species = iris_species[prediction[0]]

    # Return the result
    return {'species': species}, 200    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)