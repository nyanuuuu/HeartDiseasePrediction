from flask import Flask, request, render_template
from sklearn.pipeline import Pipeline
import pickle
import numpy as np

app = Flask(__name__, template_folder="C:\\Users\\Dnyanu Fegade\\OneDrive\\Desktop\\HEART DISEASE PREDICTION (6th sem)\\template", static_folder="C:\\Users\\Dnyanu Fegade\\OneDrive\\Desktop\\HEART DISEASE PREDICTION (6th sem)\\static")

with open('Heart_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
    

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/form.html')
def form():
    return render_template("form.html")

@app.route('/info.html')
def info():
    return render_template("info.html")

@app.route("/predict", methods=["POST"])  
def predict():  
   age = request.form["age"]  
   sex = request.form["sex"]  
   trestbps = request.form["trestbps"]  
   chol = request.form["chol"]  
   oldpeak = request.form["oldpeak"]  
   thalach = request.form["thalach"]  
   fbs = request.form["fbs"]  
   exang = request.form["exang"]  
   slope = request.form["slope"]  
   cp = request.form["cp"]  
   thal = request.form["thal"]  
   ca = request.form["ca"]  
   restecg = request.form["restecg"]  
   arr = np.array([[age, sex, cp, trestbps,  
            chol, fbs, restecg, thalach,  
            exang, oldpeak, slope, ca,  
            thal]])  
   prediction = model.predict(arr)  

   if prediction == 0:
       res_val = "No heart disease"
   elif prediction == 1:
       res_val = "Mild heart disease"
   elif prediction == 2:
       res_val = "Moderate heart disease"
   elif prediction == 3:
       res_val = "Severe heart disease"
   elif prediction == 4:
       res_val = "Critical heart disease"
   else:
       res_val = "Invalid prediction"

     # Pass all form data back to the template along with the prediction text
   return render_template('form.html', prediction_text=res_val)

if __name__ == "__main__":
    app.run(debug=True)
